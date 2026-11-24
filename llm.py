"""LLM integration. The LLM only EXPLAINS results already computed by recon.py.

It never receives raw files, never calculates, and cannot approve anything.
Its output is validated with Pydantic and number-checked against the payload.
"""
from __future__ import annotations

import json
import re
from typing import List

from pydantic import BaseModel, Field, ValidationError

from config import PROMPT_VERSION, SIGN_CONVENTION
from recon import ReconResult

SYSTEM_PROMPT = f"""You are a finance reconciliation explainer supporting a human reviewer.
Prompt version: {PROMPT_VERSION}

STRICT RULES
1. Every number you are given was computed by deterministic code. Do NOT calculate, re-compute,
   round, convert, or change any figure. Quote figures exactly as given. Do not introduce new numbers.
2. Do NOT approve, reject, or recommend posting any adjustment. Only a human reviewer decides.
3. Possible reasons MUST be labelled as hypotheses, never stated as facts.
4. Write in plain business language for a finance reviewer.
5. Respond with ONLY a JSON object with exactly these keys:
   "executive_summary": string,
   "discrepancy_explanations": [ {{"account_code": string, "explanation": string,
        "hypotheses": [string, ...]}} ],
   "reviewer_questions": [string, ...],
   "reviewer_report": string   (concise, reviewer-ready, plain text; may use line breaks)
Sign convention used by the code: {SIGN_CONVENTION}
"""


class DiscrepancyExplanation(BaseModel):
    account_code: str
    explanation: str
    hypotheses: List[str] = Field(min_length=1)


class AIReport(BaseModel):
    executive_summary: str
    discrepancy_explanations: List[DiscrepancyExplanation]
    reviewer_questions: List[str] = Field(min_length=1)
    reviewer_report: str


def build_payload(result: ReconResult) -> dict:
    """Only already-computed results leave the machine. No file names, hashes or raw CSVs."""
    t = result.totals
    return {
        "materiality_threshold_inr": result.threshold,
        "sign_convention": result.sign_convention,
        "summary": {k: t[k] for k in (
            "accounts_reconciled", "matching_accounts", "within_tolerance_accounts",
            "mismatching_accounts", "net_variance", "gross_absolute_variance",
            "ledger_net_profit", "reported_net_profit", "profit_difference",
            "missing_accounts", "duplicate_accounts")},
        "deterministic_notes": result.notes,
        "non_matching_accounts": result.to_payload_rows(),
    }


_NUM = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(text):
        try:
            out.add(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def find_ungrounded_numbers(report: AIReport, payload: dict) -> list[float]:
    """Numbers in the LLM text that do not appear in the payload (possible invented/changed figures)."""
    allowed = _numbers(json.dumps(payload))
    allowed |= {abs(x) for x in allowed}
    allowed |= {1.0, 2.0, 3.0}  # list counts / ordinals
    found = _numbers(report.model_dump_json())
    return sorted(x for x in found if x not in allowed)


def generate_report(result: ReconResult, api_key: str, model: str) -> dict:
    """Returns dict(status, report|None, ungrounded_numbers, error|None). Never raises."""
    if not api_key:
        return {"status": "SKIPPED_NO_API_KEY", "report": None, "ungrounded_numbers": [],
                "error": "No OpenAI API key provided."}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        payload = build_payload(result)
        last_err = None
        for _ in range(2):  # one retry if the JSON fails schema validation
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Computed reconciliation results:\n" + json.dumps(payload, indent=2)},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            try:
                report = AIReport.model_validate_json(raw)
                bad = find_ungrounded_numbers(report, payload)
                return {"status": "OK" if not bad else "OK_WITH_NUMBER_WARNING",
                        "report": report, "ungrounded_numbers": bad, "error": None}
            except ValidationError as e:
                last_err = f"LLM output failed schema validation: {e.errors()[:3]}"
        return {"status": "SCHEMA_VALIDATION_FAILED", "report": None,
                "ungrounded_numbers": [], "error": last_err}
    except Exception as e:  # network, auth, rate limit ...
        return {"status": "LLM_ERROR", "report": None, "ungrounded_numbers": [],
                "error": f"{type(e).__name__}: {e}"}

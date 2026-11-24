"""Audit log + human review. Nothing here posts or approves adjustments automatically."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone

from config import PROMPT_VERSION, RULES_VERSION, RUNS_DIR
from recon import ReconResult

ALLOWED_DECISIONS = ("APPROVED", "REJECTED")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_run_id() -> str:
    return "RUN-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6].upper()


def derive_review_status(flagged: list[str], reviews: list[dict]) -> str:
    """NOT_REQUIRED / PENDING_REVIEW / PARTIALLY_REVIEWED / APPROVED / REJECTED (latest decision per item wins)."""
    if not flagged:
        return "NOT_REQUIRED"
    latest = {r["account_code"]: r["decision"] for r in reviews}
    decided = [latest.get(a) for a in flagged]
    if not any(decided):
        return "PENDING_REVIEW"
    if any(d == "REJECTED" for d in decided):
        return "REJECTED" if all(decided) else "PARTIALLY_REVIEWED"
    return "APPROVED" if all(decided) else "PARTIALLY_REVIEWED"


def build_audit_log(run_id: str, files: dict[str, tuple[str, bytes]], result: ReconResult,
                    llm_outcome: dict, model: str) -> dict:
    flagged = result.table[result.table["requires_human_review"]]["account_code"].tolist()
    return {
        "run_id": run_id,
        "timestamp_utc": now_utc(),
        "input_files": [{"role": role, "file_name": name, "sha256": sha256_bytes(data)}
                        for role, (name, data) in files.items()],
        "rules": {"rules_version": RULES_VERSION,
                  "materiality_threshold_inr": result.threshold,
                  "sign_convention": result.sign_convention},
        "llm": {"provider": "openai", "model_name": model, "prompt_version": PROMPT_VERSION,
                "status": llm_outcome["status"], "error": llm_outcome["error"],
                "ungrounded_numbers_flagged": llm_outcome["ungrounded_numbers"],
                "data_sent_to_llm": "computed reconciliation results only (no raw files)"},
        "results": {**{k: result.totals[k] for k in (
            "accounts_reconciled", "matching_accounts", "within_tolerance_accounts",
            "mismatching_accounts", "net_variance", "ledger_net_profit",
            "reported_net_profit", "profit_difference")},
            "number_of_discrepancies": result.totals["accounts_requiring_review"]},
        "reconciliation_status": result.totals["reconciliation_status"],
        "accounts_requiring_human_review": flagged,
        "human_review_status": derive_review_status(flagged, []),
        "reviews": [],
        "adjustments_posted": False,
        "controls": ["Calculations & threshold checks performed by Pandas, not LLM",
                     "LLM output validated by Pydantic schema + numeric grounding check",
                     "No automatic approval or adjustment posting"],
    }


def save_log(log: dict, runs_dir: str = RUNS_DIR) -> str:
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{log['run_id']}_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return path


def record_review(log: dict, account_code: str, decision: str, reviewer: str,
                  role: str, comment: str = "") -> dict:
    """Append a human decision. Raises ValueError on bad input; never posts any adjustment."""
    if account_code not in log["accounts_requiring_human_review"]:
        raise ValueError(f"{account_code} is not flagged for review in this run.")
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"Decision must be one of {ALLOWED_DECISIONS}.")
    if not reviewer.strip() or not role.strip():
        raise ValueError("Reviewer name and role are required.")
    log["reviews"].append({"account_code": account_code, "decision": decision,
                           "reviewer": reviewer.strip(), "reviewer_role": role.strip(),
                           "comment": comment.strip(), "timestamp_utc": now_utc()})
    log["human_review_status"] = derive_review_status(log["accounts_requiring_human_review"], log["reviews"])
    return log

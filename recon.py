"""Deterministic reconciliation engine. NO LLM is used in this module.

Every financial calculation and threshold check happens here with Pandas.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

from config import (
    REQUIRED_PL_COLS,
    REQUIRED_RULE_COLS,
    REQUIRED_TB_COLS,
    SIGN_CONVENTION,
)

REVIEW_STATUSES = {
    "DUPLICATE", "MISSING_IN_PL", "MISSING_IN_TB",
    "MISMATCH", "NAME_MISMATCH", "NOT_MAPPED", "BALANCE_TYPE_CONFLICT",
}


class InputValidationError(Exception):
    """Raised when input files fail validation. Holds a list of readable problems."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


# --------------------------------------------------------------------------- #
# Part 1 - reading and validating input
# --------------------------------------------------------------------------- #
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read CSV bytes. utf-8-sig strips the BOM present in the supplied files."""
    df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def validate_frame(df: pd.DataFrame, required: list[str], label: str,
                   amount_col: str | None = None) -> pd.DataFrame:
    """Check required columns, non-empty codes and numeric amounts."""
    problems: list[str] = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InputValidationError([f"{label}: missing required column(s): {', '.join(missing)}"])

    blank = df[df["account_code"] == ""]
    if len(blank):
        problems.append(f"{label}: {len(blank)} row(s) have an empty account_code "
                        f"(file rows {', '.join(str(i + 2) for i in blank.index)})")

    out = df.copy()
    if amount_col:
        numeric = pd.to_numeric(out[amount_col], errors="coerce")
        bad = out[numeric.isna()]
        if len(bad):
            problems.append(
                f"{label}: non-numeric {amount_col} in file row(s) "
                f"{', '.join(str(i + 2) for i in bad.index)} "
                f"(values: {', '.join(repr(v) for v in bad[amount_col])})")
        out[amount_col] = numeric
    if problems:
        raise InputValidationError(problems)
    return out


# --------------------------------------------------------------------------- #
# Part 2 - reconciliation
# --------------------------------------------------------------------------- #
@dataclass
class ReconResult:
    table: pd.DataFrame
    exceptions: pd.DataFrame
    totals: dict
    threshold: float
    sign_convention: str = SIGN_CONVENTION
    notes: list[str] = field(default_factory=list)

    def to_payload_rows(self) -> list[dict]:
        cols = ["account_code", "account_name", "pl_category", "ledger_amount",
                "reported_amount", "variance", "profit_impact", "status", "issues"]
        return self.table[self.table["status"] != "MATCH"][cols].to_dict(orient="records")


def _sign(normal_balance: str | None, category: str | None) -> int:
    """+1 for income/credit accounts, -1 for debit/expense accounts."""
    if normal_balance:
        return 1 if normal_balance.lower() == "credit" else -1
    return 1 if (category or "").lower() == "income" else -1


def reconcile(tb_raw: pd.DataFrame, pl_raw: pd.DataFrame, rules_raw: pd.DataFrame,
              threshold: float) -> ReconResult:
    if threshold < 0:
        raise InputValidationError(["Materiality threshold cannot be negative."])

    tb = validate_frame(tb_raw, REQUIRED_TB_COLS, "Trial Balance", "amount_inr")
    pl = validate_frame(pl_raw, REQUIRED_PL_COLS, "Reported P&L", "reported_amount_inr")
    rules = validate_frame(rules_raw, REQUIRED_RULE_COLS, "Mapping rules")

    tb_dups = set(tb[tb.duplicated("account_code", keep=False)]["account_code"])
    pl_dups = set(pl[pl.duplicated("account_code", keep=False)]["account_code"])
    rule_dups = set(rules[rules.duplicated("account_code", keep=False)]["account_code"])
    if rule_dups:
        raise InputValidationError(
            [f"Mapping rules contain duplicate account codes: {', '.join(sorted(rule_dups))}"])

    tb_agg = tb.groupby("account_code", sort=False).agg(
        tb_name=("account_name", "first"), ledger_amount=("amount_inr", "sum"),
        tb_balance=("normal_balance", "first"))
    pl_agg = pl.groupby("account_code", sort=False).agg(
        pl_name=("account_name", "first"), reported_amount=("reported_amount_inr", "sum"))
    rules_idx = rules.set_index("account_code")

    # Preserve ledger order first, then anything only in the P&L / rules.
    codes = list(dict.fromkeys(list(tb_agg.index) + list(pl_agg.index)))

    rows = []
    for code in codes:
        in_tb, in_pl, in_rules = code in tb_agg.index, code in pl_agg.index, code in rules_idx.index
        tb_name = tb_agg.at[code, "tb_name"] if in_tb else None
        pl_name = pl_agg.at[code, "pl_name"] if in_pl else None
        ledger = float(tb_agg.at[code, "ledger_amount"]) if in_tb else None
        reported = float(pl_agg.at[code, "reported_amount"]) if in_pl else None
        rule = rules_idx.loc[code] if in_rules else None
        tb_bal = tb_agg.at[code, "tb_balance"] if in_tb else None

        issues: list[str] = []
        category = rule["pl_category"] if rule is not None else None
        include = (rule["include_in_pl"].lower() == "include") if rule is not None else True
        normal = rule["normal_balance"] if rule is not None else tb_bal

        if rule is None:
            issues.append("Account has no mapping rule")
        elif tb_bal and rule["normal_balance"].lower() != tb_bal.lower():
            issues.append(f"Normal balance differs: Trial Balance={tb_bal}, mapping rule={rule['normal_balance']}")
        if code in tb_dups:
            issues.append("Duplicate account code in Trial Balance (amounts summed for comparison)")
        if code in pl_dups:
            issues.append("Duplicate account code in Reported P&L (amounts summed for comparison)")
        if in_tb and in_pl and tb_name.lower() != pl_name.lower():
            issues.append(f"Account name differs: TB='{tb_name}' vs P&L='{pl_name}'")
        if not in_pl:
            issues.append("Account missing from Reported P&L")
        if not in_tb:
            issues.append("Account missing from Trial Balance")

        variance = round((reported or 0.0) - (ledger or 0.0), 2)
        abs_var = abs(variance)
        sign = _sign(normal, category)
        profit_impact = round(variance * sign, 2) if include else 0.0

        # Status precedence: structural problems first, then amount-based.
        if code in tb_dups or code in pl_dups:
            status = "DUPLICATE"
        elif not in_pl:
            status = "MISSING_IN_PL"
        elif not in_tb:
            status = "MISSING_IN_TB"
        elif rule is None:
            status = "NOT_MAPPED"
        elif abs_var > threshold:
            status = "MISMATCH"
        elif any(i.startswith("Normal balance differs") for i in issues):
            status = "BALANCE_TYPE_CONFLICT"
        elif any(i.startswith("Account name differs") for i in issues):
            status = "NAME_MISMATCH"
        elif abs_var > 0:
            status = "WITHIN_TOLERANCE"
        else:
            status = "MATCH"

        if status == "WITHIN_TOLERANCE":
            issues.append(f"Variance {abs_var:,.2f} is within the threshold ({threshold:,.2f})")

        rows.append({
            "account_code": code,
            "account_name": tb_name or pl_name,
            "pl_category": category or "UNMAPPED",
            "normal_balance": normal or "UNKNOWN",
            "included_in_pl": include,
            "ledger_amount": ledger,
            "reported_amount": reported,
            "variance": variance,
            "abs_variance": abs_var,
            "profit_impact": profit_impact,
            "status": status,
            "requires_human_review": status in REVIEW_STATUSES,
            "issues": "; ".join(issues),
        })

    table = pd.DataFrame(rows)

    # Net profit using the stated sign convention.
    inc = table[table["included_in_pl"]].copy()
    inc["sign"] = [_sign(n if n != "UNKNOWN" else None, c) for n, c in zip(inc["normal_balance"], inc["pl_category"])]
    ledger_profit = round(float((inc["ledger_amount"].fillna(0) * inc["sign"]).sum()), 2)
    reported_profit = round(float((inc["reported_amount"].fillna(0) * inc["sign"]).sum()), 2)
    diff = round(reported_profit - ledger_profit, 2)

    exceptions = table[table["status"] != "MATCH"].reset_index(drop=True)
    review_n = int(table["requires_human_review"].sum())
    notes = []
    gross_abs = round(float(table["abs_variance"].sum()), 2)
    if diff == 0 and gross_abs > 0:
        notes.append(
            "Net profit difference is zero, but individual account variances exist and offset each other "
            f"(gross absolute variance = {gross_abs:,.2f}). A matching bottom line does NOT mean the P&L is correct.")

    totals = {
        "accounts_reconciled": int(len(table)),
        "matching_accounts": int((table["status"] == "MATCH").sum()),
        "within_tolerance_accounts": int((table["status"] == "WITHIN_TOLERANCE").sum()),
        "mismatching_accounts": int((~table["status"].isin(["MATCH", "WITHIN_TOLERANCE"])).sum()),
        "accounts_requiring_review": review_n,
        "net_variance": round(float(table["variance"].sum()), 2),
        "gross_absolute_variance": gross_abs,
        "ledger_net_profit": ledger_profit,
        "reported_net_profit": reported_profit,
        "profit_difference": diff,
        "missing_accounts": table[table["status"].isin(["MISSING_IN_PL", "MISSING_IN_TB"])]["account_code"].tolist(),
        "duplicate_accounts": table[table["status"] == "DUPLICATE"]["account_code"].tolist(),
        "reconciliation_status": "RECONCILED" if review_n == 0 and diff == 0 else "DISCREPANCIES_FOUND",
    }
    return ReconResult(table=table, exceptions=exceptions, totals=totals,
                       threshold=threshold, notes=notes)

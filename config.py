"""Central configuration. Change values here or override via environment variables."""
import os

MATERIALITY_THRESHOLD_INR = float(os.getenv("MATERIALITY_THRESHOLD_INR", "100"))
RULES_VERSION = "mapping-rules-v1.0"
PROMPT_VERSION = "pl-recon-prompt-v1.0"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

REQUIRED_TB_COLS = ["account_code", "account_name", "amount_inr", "normal_balance"]
REQUIRED_PL_COLS = ["account_code", "account_name", "reported_amount_inr"]
REQUIRED_RULE_COLS = ["account_code", "account_name", "pl_category", "normal_balance", "include_in_pl"]

SIGN_CONVENTION = (
    "All amounts are stored as positive balances in their normal direction. "
    "Credit-balance accounts (Income) add to profit (+1); Debit-balance accounts "
    "(expenses) reduce profit (-1). Net profit = Income - Expenses. "
    "Variance = Reported P&L amount - Trial Balance amount. "
    "Profit difference = Reported net profit - Ledger net profit "
    "(positive means the reported P&L shows MORE profit than the ledger supports)."
)

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

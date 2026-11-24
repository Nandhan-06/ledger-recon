# AI-Powered P&L Reconciliation Assistant (Assignment 1 – Finance)

Compares a Trial Balance with a reported P&L, flags discrepancies above a materiality threshold,
asks OpenAI to *explain* (never calculate) the results, writes a JSON audit log and supports human review.
Synthetic data only.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."                          # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
streamlit run app.py
```
You can also paste the key into the sidebar (password field, never stored). Copy `.env.example` to `.env` if you prefer
(`.env` is git-ignored). Optional: `OPENAI_MODEL`, `MATERIALITY_THRESHOLD_INR`.

## Run tests
`pytest -q tests`

## Using the UI
1. Sidebar: pick sample files (in `data/`) or upload your own, set the threshold (default ₹100), click **Run reconciliation**.
2. Tabs: Reconciliation table + chart · Exceptions & profit · AI explanation · Human review · Audit log.
3. In **Human review**, record APPROVED/REJECTED with reviewer name + role. This only updates the audit log; nothing is posted.

## Architecture – who does what
| Stage | Component | Notes |
|---|---|---|
| Read + validate CSVs | `recon.py` (Pandas) | required columns, numeric amounts; bad input **blocks** the run |
| Variance, status, threshold, profit | `recon.py` (Pandas) | deterministic; **no LLM** |
| Explanation, hypotheses, reviewer questions, report | `llm.py` (OpenAI) | receives computed results only; JSON validated with Pydantic; numbers not in the payload are flagged |
| Audit log | `audit.py` | run ID, SHA-256 of inputs, rules/prompt versions, model, counts, status, timestamps |
| Approval | **Human** via UI | flagged accounts stay `PENDING_REVIEW` until a person decides |

### Sign convention
Credit accounts (Income) = +1, debit accounts (expenses) = −1. Net profit = Income − Expenses.
Variance = Reported − Trial Balance. Profit difference = Reported profit − Ledger profit (positive = reported profit is higher).

### Sample result (supplied data, threshold ₹100)
Utilities +500, Marketing −300, Depreciation −200 are flagged. Net profit is ₹6,000 in both sources, so the profit difference is 0 –
the variances offset each other, and the app warns that a matching bottom line does not mean the P&L is correct.
See `sample_audit_log.json` (generated with the AI step off, plus two example review decisions).

## Failure scenarios to demo
- Upload a Trial Balance with `abc` as an amount → run blocked with row number.
- Delete a column / a P&L row → validation error / `MISSING_IN_PL`.
- Duplicate a Trial Balance row → `DUPLICATE`.
- Raise the threshold to 300 → only Utilities stays flagged.
- Remove the API key → reconciliation still works; AI tab explains why it is unavailable.

## Files
`app.py` UI · `recon.py` engine · `llm.py` OpenAI · `audit.py` log + review · `config.py` settings · `data/` inputs · `runs/` saved logs · `tests/`

"""Streamlit UI - AI-Powered P&L Reconciliation Assistant.  Run:  streamlit run app.py"""
import json
import os

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import audit
import config
import llm
import recon

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAMPLE = {"trial_balance": "finance_trial_balance.csv",
          "reported_pl": "finance_reported_pl.csv",
          "mapping_rules": "finance_mapping_rules.csv"}

st.set_page_config(page_title="P&L Reconciliation Assistant", page_icon="📊", layout="wide")
st.title("📊 AI-Powered P&L Reconciliation Assistant")
st.caption("Python/Pandas does all the maths. The LLM only explains. A human decides. Synthetic data only.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API key", type="password",
                            value=os.getenv("OPENAI_API_KEY", ""),
                            help="Read from OPENAI_API_KEY if set. Never stored or logged.")
    model = st.text_input("Model", value=config.DEFAULT_MODEL)
    threshold = st.number_input("Materiality threshold (₹)", min_value=0.0,
                                value=config.MATERIALITY_THRESHOLD_INR, step=50.0)
    use_ai = st.checkbox("Generate AI explanation", value=True)
    st.divider()
    st.subheader("Input files")
    source = st.radio("Source", ["Use sample files", "Upload my own"])
    uploads = {}
    if source == "Upload my own":
        uploads["trial_balance"] = st.file_uploader("Trial Balance CSV", type="csv")
        uploads["reported_pl"] = st.file_uploader("Reported P&L CSV", type="csv")
        uploads["mapping_rules"] = st.file_uploader("Mapping rules CSV", type="csv")
    run_clicked = st.button("▶ Run reconciliation", type="primary", use_container_width=True)


def load_inputs() -> dict:
    files = {}
    for role, fname in SAMPLE.items():
        if source == "Use sample files":
            with open(os.path.join(DATA_DIR, fname), "rb") as f:
                files[role] = (fname, f.read())
        else:
            up = uploads.get(role)
            if up is None:
                raise recon.InputValidationError([f"Please upload the {role.replace('_', ' ')} file."])
            files[role] = (up.name, up.getvalue())
    return files


# ------------------------------------------------------------------ run
if run_clicked:
    st.session_state.pop("error", None)
    try:
        files = load_inputs()
        frames = {r: recon.read_csv_bytes(d) for r, (_, d) in files.items()}
        result = recon.reconcile(frames["trial_balance"], frames["reported_pl"],
                                 frames["mapping_rules"], threshold)
        if use_ai:
            with st.spinner("Asking the LLM to explain the computed results…"):
                outcome = llm.generate_report(result, api_key, model)
        else:
            outcome = {"status": "SKIPPED_BY_USER", "report": None, "ungrounded_numbers": [], "error": None}
        run_id = audit.new_run_id()
        log = audit.build_audit_log(run_id, files, result, outcome, model)
        path = audit.save_log(log)
        st.session_state.update(result=result, outcome=outcome, log=log, log_path=path)
    except recon.InputValidationError as e:
        st.session_state["error"] = e.problems
        st.session_state.pop("result", None)
    except Exception as e:  # unreadable CSV etc.
        st.session_state["error"] = [f"Could not process files: {type(e).__name__}: {e}"]
        st.session_state.pop("result", None)

if st.session_state.get("error"):
    st.error("⛔ Input validation failed - reconciliation blocked.")
    for p in st.session_state["error"]:
        st.write(f"- {p}")
    st.stop()

if "result" not in st.session_state:
    st.info("Choose your inputs in the sidebar and click **Run reconciliation**.")
    st.stop()

result: recon.ReconResult = st.session_state["result"]
outcome = st.session_state["outcome"]
log = st.session_state["log"]
t = result.totals

# ------------------------------------------------------------------ dashboard
status_icon = "✅" if t["reconciliation_status"] == "RECONCILED" else "⚠️"
st.subheader(f"{status_icon} {t['reconciliation_status'].replace('_', ' ')}  ·  Review: {log['human_review_status'].replace('_', ' ')}")
c = st.columns(5)
c[0].metric("Accounts reconciled", t["accounts_reconciled"])
c[1].metric("Matching", t["matching_accounts"])
c[2].metric("Mismatching", t["mismatching_accounts"])
c[3].metric("Gross abs. variance (₹)", f"{t['gross_absolute_variance']:,.2f}")
c[4].metric("Profit difference (₹)", f"{t['profit_difference']:,.2f}")
for n in result.notes:
    st.warning(n)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Reconciliation", "Exceptions & profit", "AI explanation", "Human review", "Audit log"])

STATUS_COLOURS = {"MATCH": "#d4edda", "WITHIN_TOLERANCE": "#fff3cd"}


def colour(row):
    bg = STATUS_COLOURS.get(row["status"], "#f8d7da")
    return [f"background-color: {bg}; color: #111"] * len(row)


with tab1:
    show = result.table[["account_code", "account_name", "ledger_amount", "reported_amount",
                         "variance", "status", "requires_human_review", "issues"]]
    st.dataframe(show.style.apply(colour, axis=1).format(
        {"ledger_amount": "{:,.2f}", "reported_amount": "{:,.2f}", "variance": "{:+,.2f}"}, na_rep="—"),
        use_container_width=True, hide_index=True)
    st.caption(f"Flag rule: |variance| > ₹{result.threshold:,.2f}. Variance = Reported − Trial Balance.")
    chart = pd.DataFrame({"accounts": [t["matching_accounts"], t["within_tolerance_accounts"], t["mismatching_accounts"]]},
                         index=["Matching", "Within tolerance", "Mismatching"])
    st.bar_chart(chart)

with tab2:
    st.markdown("**Missing / duplicate / mismatched accounts**")
    if result.exceptions.empty:
        st.success("No exceptions.")
    else:
        st.dataframe(result.exceptions[["account_code", "account_name", "status", "variance", "profit_impact", "issues"]],
                     use_container_width=True, hide_index=True)
    st.markdown("**Net profit**")
    p = st.columns(3)
    p[0].metric("Ledger net profit (₹)", f"{t['ledger_net_profit']:,.2f}")
    p[1].metric("Reported net profit (₹)", f"{t['reported_net_profit']:,.2f}")
    p[2].metric("Difference (₹)", f"{t['profit_difference']:,.2f}")
    st.info(result.sign_convention)

with tab3:
    s = outcome["status"]
    if outcome["report"] is None:
        if s == "SKIPPED_NO_API_KEY":
            st.warning("No OpenAI API key entered. Add it in the sidebar and re-run to get the AI explanation.")
        elif s == "SKIPPED_BY_USER":
            st.info("AI explanation was turned off.")
        else:
            st.error(f"AI explanation unavailable ({s}): {outcome['error']}")
        st.caption("The reconciliation above is unaffected - it never depends on the LLM.")
    else:
        rep = outcome["report"]
        st.caption(f"Generated by `{log['llm']['model_name']}` · prompt {config.PROMPT_VERSION} · "
                   "AI-generated text, not a financial approval. Reasons are hypotheses.")
        if outcome["ungrounded_numbers"]:
            st.warning("⚠️ The AI text contains figure(s) not present in the computed results: "
                       f"{outcome['ungrounded_numbers']}. Do not rely on them - verify against the tables.")
        st.markdown("#### Summary")
        st.write(rep.executive_summary)
        st.markdown("#### Discrepancy explanations")
        for d in rep.discrepancy_explanations:
            with st.expander(d.account_code, expanded=True):
                st.write(d.explanation)
                st.markdown("*Hypotheses (unverified):*")
                for h in d.hypotheses:
                    st.markdown(f"- {h}")
        st.markdown("#### Questions for the reviewer")
        for q in rep.reviewer_questions:
            st.markdown(f"- {q}")
        st.markdown("#### Reviewer-ready report")
        st.text(rep.reviewer_report)

with tab4:
    flagged = log["accounts_requiring_human_review"]
    st.markdown(f"**Human review status: `{log['human_review_status']}`**")
    st.caption("Recording a decision here only updates the audit log. No adjustment is ever posted by this tool.")
    if not flagged:
        st.success("No accounts exceed the threshold - review not required.")
    else:
        with st.form("review_form", clear_on_submit=True):
            acct = st.selectbox("Account", flagged)
            decision = st.radio("Decision", list(audit.ALLOWED_DECISIONS), horizontal=True)
            name = st.text_input("Reviewer name")
            role = st.text_input("Reviewer role (e.g. Finance Controller)")
            comment = st.text_area("Comment (optional)")
            if st.form_submit_button("Record decision"):
                try:
                    audit.record_review(log, acct, decision, name, role, comment)
                    audit.save_log(log)
                    st.session_state["log"] = log
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        if log["reviews"]:
            st.dataframe(pd.DataFrame(log["reviews"]), use_container_width=True, hide_index=True)

with tab5:
    st.json(log)
    st.download_button("⬇ Download audit log (JSON)", json.dumps(log, indent=2),
                       file_name=f"{log['run_id']}_audit.json", mime="application/json")
    st.caption(f"Saved to {st.session_state['log_path']}")

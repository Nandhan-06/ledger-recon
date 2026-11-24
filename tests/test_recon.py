import io, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, pytest
import audit, llm, recon

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
def rd(n): return recon.read_csv_bytes(open(f"{D}/{n}", "rb").read())
def run(thr=100, tb=None, pl=None):
    return recon.reconcile(tb if tb is not None else rd("finance_trial_balance.csv"),
                           pl if pl is not None else rd("finance_reported_pl.csv"),
                           rd("finance_mapping_rules.csv"), thr)

def test_three_expected_mismatches():
    r = run()
    bad = set(r.table[r.table.status == "MISMATCH"].account_code)
    assert bad == {"UTIL001", "MKT001", "DEP001"}
    assert r.table.set_index("account_code").variance.loc[["UTIL001", "MKT001", "DEP001"]].tolist() == [500, -300, -200]

def test_profit_and_offset_note():
    r = run()
    assert r.totals["ledger_net_profit"] == 6000 and r.totals["profit_difference"] == 0
    assert r.totals["gross_absolute_variance"] == 1000 and r.notes

def test_threshold_configurable():
    r = run(thr=300)  # 500 flagged; 300 and 200 are within tolerance
    assert r.totals["accounts_requiring_review"] == 1
    assert (r.table.status == "WITHIN_TOLERANCE").sum() == 2

def test_missing_and_duplicate():
    tb = rd("finance_trial_balance.csv")
    pl = rd("finance_reported_pl.csv")
    pl = pl[pl.account_code != "TRV001"]
    tb = pd.concat([tb, tb[tb.account_code == "RENT001"]], ignore_index=True)
    r = run(tb=tb, pl=pl)
    st = r.table.set_index("account_code").status
    assert st["TRV001"] == "MISSING_IN_PL" and st["RENT001"] == "DUPLICATE"

def test_non_numeric_blocked():
    tb = rd("finance_trial_balance.csv"); tb.loc[0, "amount_inr"] = "abc"
    with pytest.raises(recon.InputValidationError): run(tb=tb)

def test_missing_column_blocked():
    pl = rd("finance_reported_pl.csv").drop(columns=["reported_amount_inr"])
    with pytest.raises(recon.InputValidationError): run(pl=pl)

def test_review_flow_and_validation():
    r = run()
    outcome = {"status": "SKIPPED_BY_USER", "report": None, "ungrounded_numbers": [], "error": None}
    files = {"trial_balance": ("tb.csv", b"x")}
    log = audit.build_audit_log("RUN-T", files, r, outcome, "gpt-4o-mini")
    assert log["human_review_status"] == "PENDING_REVIEW" and log["adjustments_posted"] is False
    with pytest.raises(ValueError): audit.record_review(log, "UTIL001", "APPROVED", "", "")
    with pytest.raises(ValueError): audit.record_review(log, "REV001", "APPROVED", "A", "B")
    audit.record_review(log, "UTIL001", "APPROVED", "Asha", "Controller")
    assert log["human_review_status"] == "PARTIALLY_REVIEWED"
    audit.record_review(log, "MKT001", "APPROVED", "Asha", "Controller")
    audit.record_review(log, "DEP001", "REJECTED", "Asha", "Controller", "needs support")
    assert log["human_review_status"] == "REJECTED"

def test_ungrounded_number_detection():
    r = run(); p = llm.build_payload(r)
    good = llm.AIReport(executive_summary="Utilities differ by 500.00.",
        discrepancy_explanations=[llm.DiscrepancyExplanation(account_code="UTIL001", explanation="x", hypotheses=["h"])],
        reviewer_questions=["q"], reviewer_report="Net 0.00")
    assert llm.find_ungrounded_numbers(good, p) == []
    bad = good.model_copy(update={"executive_summary": "Utilities differ by 777."})
    assert llm.find_ungrounded_numbers(bad, p) == [777.0]

def test_no_key_skips_llm_gracefully():
    assert llm.generate_report(run(), "", "m")["status"] == "SKIPPED_NO_API_KEY"

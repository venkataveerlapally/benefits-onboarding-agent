"""Demo UI.   streamlit run ui/streamlit_app.py"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import llm  # noqa: E402
from app.outputs import issues_dataframe  # noqa: E402
from app.pipeline import run_onboarding  # noqa: E402

DATA = ROOT / "data" / "synthetic"
STATUS_ICON = {"done": "✅", "todo": "⬜", "needs_review": "🟡", "blocked": "🔴"}
SEV_COLOR = {"high": "#fde2e1", "medium": "#fff4d6", "low": "#eef2f7"}

st.set_page_config(page_title="Benefits Onboarding Agent", page_icon="🩺", layout="wide")
st.title("Benefits Client Onboarding Agent")
st.caption("Upload a client's census, payroll export and plan document. The agent maps the columns, "
           "checks eligibility and compliance, and drafts the follow-up to the client. All demo data is synthetic.")

# ------------------------------------------------------------------ sidebar: choose input
with st.sidebar:
    st.subheader("Input")
    mode = st.radio("Source", ["Demo client", "Upload files"], horizontal=True)
    if mode == "Demo client":
        slugs = sorted(p.name for p in DATA.iterdir() if p.is_dir()) if DATA.exists() else []
        if not slugs:
            st.error("No demo data. Run `python scripts/generate_data.py` first.")
            st.stop()
        slug = st.selectbox("Client", slugs, format_func=lambda s: s.replace("_", " ").title())
        client_name = slug.replace("_", " ").title()
        census_path = next((DATA / slug).glob("census.*"))
        plan_path, payroll_path = DATA / slug / "plan_summary.pdf", DATA / slug / "payroll.csv"
    else:
        client_name = st.text_input("Client name", "New Client")
        census_up = st.file_uploader("Employee census", type=["xlsx", "csv"])
        payroll_up = st.file_uploader("Payroll export (optional)", type=["xlsx", "csv"])
        plan_up = st.file_uploader("Plan document (optional)", type=["pdf"])
        tmp = Path(tempfile.mkdtemp())

        def save(up):
            if up is None:
                return None
            p = tmp / up.name
            p.write_bytes(up.getvalue())
            return p

        census_path, payroll_path, plan_path = save(census_up), save(payroll_up), save(plan_up)

    run = st.button("Run onboarding", type="primary", use_container_width=True, disabled=census_path is None)
    st.divider()
    st.caption(f"LLM: {'🟢 ' + llm.model_label() if llm.available() else '⚪ off, using free deterministic fallbacks'}")

if run:
    with st.spinner("Reading files, mapping columns, running checks..."):
        st.session_state.result = run_onboarding(client_name, census_path, plan_path, payroll_path)
        st.session_state.chat = []

r = st.session_state.get("result")
if r is None:
    st.info("Pick a demo client in the sidebar and click **Run onboarding**.")
    st.stop()

# ------------------------------------------------------------------ KPIs
s = r.stats
c = st.columns(6)
c[0].metric("Employees", s["employees"])
c[1].metric("Dependents", s["dependents"])
c[2].metric("Ready to enroll", s["ready_to_enroll"])
c[3].metric("Blocking issues", s["issues_high"])
c[4].metric("Needs review", s["issues_medium"])
c[5].metric("FTEs", s["fte"], "ACA ALE" if s["aca_ale"] else "Not ALE", delta_color="off")

tabs = st.tabs(["🚩 Issues", "☑️ Checklist", "✉️ Client email", "💬 Plan Q&A", "👥 Employees", "⚙️ Under the hood"])

# ------------------------------------------------------------------ Issues
with tabs[0]:
    df = issues_dataframe(r.issues)
    f1, f2 = st.columns(2)
    sev = f1.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium"])
    cats = f2.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
    view = df[df["severity"].isin(sev) & df["category"].isin(cats)]
    st.dataframe(
        view[["severity", "category", "employee_id", "message"]].style.apply(
                        lambda row: [f"background-color: {SEV_COLOR[row['severity']]}; color: #1a1a1a"] * len(row), axis=1),
        use_container_width=True, hide_index=True, height=480)
    st.download_button("Download issues (CSV)", df.to_csv(index=False), f"{r.client_name}_issues.csv", "text/csv")

# ------------------------------------------------------------------ Checklist
with tabs[1]:
    for item in r.checklist:
        st.markdown(f"{STATUS_ICON[item['status']]} **{item['task']}**: {item['detail']}")

# ------------------------------------------------------------------ Email
with tabs[2]:
    st.caption(f"Drafted by: {r.email_method}. Review and edit before sending. Nothing is sent automatically.")
    edited = st.text_area("Draft", r.email, height=520)
    st.download_button("Download .txt", edited, f"{r.client_name}_followup.txt")

# ------------------------------------------------------------------ Q&A
with tabs[3]:
    if r.qa is None:
        st.warning("Upload a plan document to enable Q&A.")
    else:
        st.caption(f"Answers come only from **{r.plan.plan_name}** documents, with citations. "
                   "Low-confidence questions are routed to a human.")
        examples = ["Can my 27 year old son stay on my plan?", "When does my coverage start?",
                    "How much are generic meds?", "I just had a baby, how do I add her?"]
        ex_cols = st.columns(len(examples))
        clicked = next((q for col, q in zip(ex_cols, examples) if col.button(q, use_container_width=True)), None)
        with st.form("ask", clear_on_submit=True):
            typed = st.text_input("Ask a benefits question")
            submitted = st.form_submit_button("Ask")
        q = clicked or (typed if submitted and typed else None)
        if q:
            st.session_state.chat.append((q, r.qa.ask(q)))
        for q, a in reversed(st.session_state.chat):
            st.markdown(f"**🙋 {q}**")
            if a["handoff"]:
                st.warning(a["answer"])
            else:
                st.success(a["answer"])
            st.caption(f"confidence {a['confidence']} · {a['method']} · sources: "
                       + ", ".join(f"{c['section']} p.{c['page']}" for c in a["citations"][:2]))

# ------------------------------------------------------------------ Employees
with tabs[4]:
    issue_count = pd.Series([i.employee_id for i in r.issues if i.client_action]).value_counts()
    rows = [{
        "ID": e.employee_id, "Name": e.full_name, "Hire date": e.hire_date, "Hrs/wk": e.hours_per_week,
        "Dependents": len(e.dependents), "Coverage starts": r.eligibility.get(e.employee_id),
        "Open issues": int(issue_count.get(e.employee_id, 0)),
    } for e in r.employees]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=520)

# ------------------------------------------------------------------ Under the hood
with tabs[5]:
    a, b = st.columns(2)
    a.subheader(f"Column mapping ({r.mapping.method})")
    a.dataframe(pd.DataFrame(r.mapping.mapping.items(), columns=["Client header", "Canonical field"]),
                hide_index=True, use_container_width=True)
    if r.mapping.unmapped:
        a.warning(f"Unmapped columns: {', '.join(r.mapping.unmapped)}")
    b.subheader(f"Plan rules ({r.plan.extraction_method})")
    b.json(r.plan.model_dump(mode="json"))
    b.subheader("Timings (sec)")
    b.json(r.timings)

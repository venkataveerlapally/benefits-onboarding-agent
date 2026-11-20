"""Human-facing outputs: onboarding checklist, client email draft, issues export."""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from . import llm
from .models import Issue, PlanRules, Severity

SEV_ORDER = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}


def issues_dataframe(issues: list[Issue]) -> pd.DataFrame:
    rows = [i.model_dump() for i in sorted(issues, key=lambda i: (SEV_ORDER[i.severity], i.category))]
    df = pd.DataFrame(rows, columns=["severity", "category", "code", "employee_id", "message", "field", "client_action"])
    df["severity"] = df["severity"].map(lambda s: s.value if hasattr(s, "value") else s)
    return df


def build_checklist(issues: list[Issue], stats: dict, plan: PlanRules, has_payroll: bool, mapping_method: str) -> list[dict]:
    by_cat = defaultdict(list)
    for i in issues:
        by_cat[i.category].append(i)
    high = [i for i in issues if i.severity == Severity.HIGH]
    data_problems = by_cat["Missing data"] + by_cat["Data format"]

    def item(task, status, detail):
        return {"task": task, "status": status, "detail": detail}

    return [
        item("Receive & parse employee census", "done",
             f"{stats['employees']} employees, {stats['dependents']} dependents (columns mapped by {mapping_method})"),
        item("Review plan document", "done",
             f"{plan.plan_name}: {plan.waiting_period_days}-day waiting period, {plan.min_hours_per_week:g} hr minimum, "
             f"dependents to age {plan.dependent_age_limit}"),
        item("Reconcile census against payroll",
             "todo" if not has_payroll else ("needs_review" if by_cat["Payroll reconciliation"] else "done"),
             "No payroll file provided" if not has_payroll else f"{len(by_cat['Payroll reconciliation'])} discrepancies"),
        item("Remove duplicate records", "blocked" if by_cat["Duplicates"] else "done",
             f"{len(by_cat['Duplicates'])} duplicates found" if by_cat["Duplicates"] else "No duplicates"),
        item("Resolve missing / invalid employee data",
             "blocked" if any(i.severity == Severity.HIGH for i in data_problems) else ("needs_review" if data_problems else "done"),
             f"{len(data_problems)} items"),
        item("Verify dependent eligibility", "needs_review" if by_cat["Dependents"] else "done",
             f"{len(by_cat['Dependents'])} items"),
        item("Calculate eligibility & waiting periods", "done",
             f"{stats['eligible_on_effective_date']} eligible on day one, {stats['in_waiting_period']} in waiting period, "
             f"{stats['part_time']} part-time"),
        item("Determine ACA status", "done",
             f"{stats['fte']} FTEs: {'Applicable Large Employer (1094-C/1095-C required)' if stats['aca_ale'] else 'not an ALE'}"),
        item("Send data request to client", "todo" if any(i.client_action for i in issues) else "done",
             f"{sum(1 for i in issues if i.client_action)} items need the client's input"),
        item("Submit enrollment file to carrier", "blocked" if high else "todo",
             f"Waiting on {len(high)} high-severity fixes" if high else "Ready to submit"),
        item("Send employee enrollment invites", "todo", f"{stats['ready_to_enroll']} employees ready now"),
    ]


def _template_email(client_name: str, plan: PlanRules, issues: list[Issue], stats: dict) -> str:
    actionable = sorted([i for i in issues if i.client_action], key=lambda i: (SEV_ORDER[i.severity], i.category))
    grouped = defaultdict(list)
    for i in actionable:
        grouped[i.category].append(i)

    lines = [
        f"Subject: {client_name} benefits onboarding: {len(actionable)} items to confirm before enrollment",
        "",
        f"Hi {client_name} team,",
        "",
        f"Thanks for sending over your census and payroll files. We've processed {stats['employees']} employees and "
        f"{stats['dependents']} dependents for the {plan.plan_name} plan"
        + (f" starting {plan.effective_date:%B %-d, %Y}." if plan.effective_date else "."),
        "",
        f"Good news: {stats['ready_to_enroll']} employees are ready to enroll. Before we can submit the full file to the "
        f"carrier, we need your help with the items below:",
        "",
    ]
    for cat, items in grouped.items():
        lines.append(f"{cat} ({len(items)})")
        for i in items:
            flag = " [required]" if i.severity == Severity.HIGH else ""
            lines.append(f"  - {i.message}{flag}")
        lines.append("")
    lines += [
        "Items marked [required] block carrier submission, so those are the priority. The easiest way to send "
        "corrections is to reply to this email or update the attached issues spreadsheet. Please don't email full "
        "SSNs; use the secure upload link instead.",
        "",
        "Thanks!",
        "Benefits Onboarding Team",
    ]
    return "\n".join(lines)


def draft_client_email(client_name: str, plan: PlanRules, issues: list[Issue], stats: dict) -> tuple[str, str]:
    """Return (email_text, method). The LLM only rewrites tone; it gets a fixed list of issues and must not add any."""
    template = _template_email(client_name, plan, issues, stats)
    if llm.available():
        actionable = [{"category": i.category, "severity": i.severity.value, "message": i.message}
                      for i in issues if i.client_action]
        text = llm.complete(
            "You write clear, warm, concise emails for a benefits administration company. "
            "Never invent issues or numbers. Include every issue you are given, grouped by category. "
            "Mark high-severity items as required. Start with a 'Subject:' line. Never ask for SSNs by email.",
            f"Client: {client_name}\nPlan: {plan.plan_name}, effective {plan.effective_date}\n"
            f"Stats: {json.dumps(stats)}\nIssues: {json.dumps(actionable)}\n\n"
            f"Here is a correct but plain draft to improve on:\n{template}",
            max_tokens=3000,
        )
        if text and text.strip().lower().startswith("subject"):
            return text.strip(), "llm"
    return template, "template"

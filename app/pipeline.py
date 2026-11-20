"""One function that runs the whole onboarding flow. Used by the CLI, API and Streamlit UI."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .extract import build_employees, extract_plan_rules, map_columns
from .ingest import read_pdf_pages, read_table
from .models import ColumnMapping, Employee, Issue, PlanRules
from .outputs import build_checklist, draft_client_email
from .qa import PlanQA
from .rules import run_rules


@dataclass
class OnboardingResult:
    client_name: str
    mapping: ColumnMapping
    plan: PlanRules
    employees: list[Employee]
    issues: list[Issue]
    stats: dict
    checklist: list[dict]
    email: str
    email_method: str
    eligibility: dict
    qa: PlanQA | None = None
    timings: dict = field(default_factory=dict)


def run_onboarding(client_name: str, census_path: str | Path, plan_pdf_path: str | Path | None = None,
                   payroll_path: str | Path | None = None, as_of: date | None = None) -> OnboardingResult:
    t, timings = time.perf_counter(), {}

    def lap(name):
        nonlocal t
        now = time.perf_counter(); timings[name] = round(now - t, 3); t = now

    census_df = read_table(census_path)
    mapping = map_columns(census_df)
    employees, parse_issues = build_employees(census_df, mapping)
    lap("ingest_extract")

    pages = read_pdf_pages(plan_pdf_path) if plan_pdf_path else []
    plan = extract_plan_rules(pages) if pages else PlanRules()
    lap("plan_extract")

    payroll_df = read_table(payroll_path) if payroll_path else None
    result = run_rules(employees, plan, payroll_df, as_of=as_of)
    issues = parse_issues + result.issues
    lap("rules")

    checklist = build_checklist(issues, result.stats, plan, payroll_df is not None, mapping.method)
    email, email_method = draft_client_email(client_name, plan, issues, result.stats)
    qa = PlanQA(pages, plan.plan_name) if pages else None
    lap("outputs")

    return OnboardingResult(client_name, mapping, plan, employees, issues, result.stats, checklist,
                            email, email_method, result.eligibility, qa, timings)

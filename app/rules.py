"""Deterministic eligibility & compliance checks.

No LLM in here on purpose: every flag must be explainable, reproducible and unit-tested.
Each check is a small function so new rules are easy to add (and to test in isolation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from .extract import map_columns, parse_number
from .models import Employee, Issue, PlanRules, Relationship, Severity

SSN_OK = re.compile(r"^\d{3}-\d{2}-\d{4}$")
EMAIL_OK = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SALARY_TOLERANCE = 0.05       # flag census vs payroll differences over 5%
ACA_FT_HOURS = 30             # ACA full-time threshold
ACA_ALE_THRESHOLD = 50        # Applicable Large Employer: 50+ full-time equivalents
FUTURE_HIRE_GRACE_DAYS = 60   # start dates within 60 days are normal upcoming hires, beyond that likely a typo


def age_on(dob: date, on: date) -> int:
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


def first_of_month_on_or_after(d: date) -> date:
    if d.day == 1:
        return d
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def eligibility_date(hire: date, plan: PlanRules) -> date:
    """Coverage starts the first of the month after the waiting period (never before plan start)."""
    start = first_of_month_on_or_after(hire + timedelta(days=plan.waiting_period_days))
    return max(start, plan.effective_date) if plan.effective_date else start


@dataclass
class RuleResult:
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    eligibility: dict[str, date | None] = field(default_factory=dict)  # employee_id -> coverage start


# --------------------------------------------------------------------------- per-employee checks

def check_required_fields(e: Employee) -> list[Issue]:
    out = []
    if not e.first_name or not e.last_name:
        out.append(Issue(code="MISSING_NAME", severity=Severity.HIGH, category="Missing data",
                         employee_id=e.employee_id, field="name", message="Employee name is incomplete"))
    if not e.ssn:
        out.append(Issue(code="MISSING_SSN", severity=Severity.HIGH, category="Missing data",
                         employee_id=e.employee_id, field="ssn",
                         message=f"{e.full_name}: SSN is missing (required for carrier enrollment and ACA reporting)"))
    elif not SSN_OK.match(e.ssn):
        out.append(Issue(code="INVALID_SSN", severity=Severity.HIGH, category="Data format",
                         employee_id=e.employee_id, field="ssn",
                         message=f"{e.full_name}: SSN doesn't look valid (should be 9 digits)"))
    if not e.dob:
        out.append(Issue(code="MISSING_DOB", severity=Severity.HIGH, category="Missing data",
                         employee_id=e.employee_id, field="dob",
                         message=f"{e.full_name}: date of birth is missing (needed for age-rated premiums)"))
    if not e.hire_date:
        out.append(Issue(code="MISSING_HIRE_DATE", severity=Severity.HIGH, category="Missing data",
                         employee_id=e.employee_id, field="hire_date",
                         message=f"{e.full_name}: hire date is missing, so eligibility can't be determined"))
    if e.hours_per_week is None:
        out.append(Issue(code="MISSING_HOURS", severity=Severity.MEDIUM, category="Missing data",
                         employee_id=e.employee_id, field="hours_per_week",
                         message=f"{e.full_name}: scheduled weekly hours missing"))
    if e.email and not EMAIL_OK.match(e.email):
        out.append(Issue(code="INVALID_EMAIL", severity=Severity.LOW, category="Data format",
                         employee_id=e.employee_id, field="email",
                         message=f"{e.full_name}: email '{e.email}' is invalid (enrollment invites will bounce)"))
    return out


def check_dates(e: Employee, as_of: date) -> list[Issue]:
    out = []
    if e.hire_date and e.hire_date > as_of + timedelta(days=FUTURE_HIRE_GRACE_DAYS):
        out.append(Issue(code="FUTURE_HIRE_DATE", severity=Severity.MEDIUM, category="Data format",
                         employee_id=e.employee_id, field="hire_date",
                         message=f"{e.full_name}: hire date {e.hire_date:%m/%d/%Y} is far in the future (typo?)"))
    elif e.hire_date and e.hire_date > as_of:
        out.append(Issue(code="UPCOMING_HIRE", severity=Severity.LOW, category="Eligibility",
                         employee_id=e.employee_id, client_action=False,
                         message=f"{e.full_name}: starts {e.hire_date:%m/%d/%Y} (upcoming new hire)"))
    if e.dob:
        a = age_on(e.dob, as_of)
        if a < 16 or a > 85:
            out.append(Issue(code="IMPLAUSIBLE_AGE", severity=Severity.MEDIUM, category="Data format",
                             employee_id=e.employee_id, field="dob",
                             message=f"{e.full_name}: date of birth gives age {a}, please confirm"))
    return out


def check_dependents(e: Employee, plan: PlanRules, on: date) -> list[Issue]:
    out = []
    spouses = [d for d in e.dependents if d.relationship == Relationship.SPOUSE]
    if len(spouses) > 1:
        out.append(Issue(code="MULTIPLE_SPOUSES", severity=Severity.MEDIUM, category="Dependents",
                         employee_id=e.employee_id, message=f"{e.full_name}: {len(spouses)} spouses listed"))
    for d in e.dependents:
        name = f"{d.first_name} {d.last_name}"
        if not d.dob:
            out.append(Issue(code="DEPENDENT_MISSING_DOB", severity=Severity.MEDIUM, category="Dependents",
                             employee_id=e.employee_id,
                             message=f"{e.full_name}: dependent {name} is missing date of birth"))
            continue
        if d.relationship == Relationship.CHILD and age_on(d.dob, on) >= plan.dependent_age_limit:
            out.append(Issue(code="DEPENDENT_OVER_AGE", severity=Severity.HIGH, category="Dependents",
                             employee_id=e.employee_id,
                             message=f"{e.full_name}: child {name} is {age_on(d.dob, on)} on {on:%m/%d/%Y}, over the "
                                     f"plan's age-{plan.dependent_age_limit} limit (eligible only with disability certification)"))
        if not d.ssn:
            out.append(Issue(code="DEPENDENT_MISSING_SSN", severity=Severity.LOW, category="Dependents",
                             employee_id=e.employee_id,
                             message=f"{e.full_name}: dependent {name} has no SSN (needed for Form 1095-C)"))
    return out


def check_eligibility(e: Employee, plan: PlanRules) -> tuple[list[Issue], date | None]:
    """Informational: who is eligible, and when. Not sent to the client as a 'problem'."""
    out = []
    if e.hours_per_week is not None and e.hours_per_week < plan.min_hours_per_week:
        out.append(Issue(code="INELIGIBLE_PART_TIME", severity=Severity.LOW, category="Eligibility",
                         employee_id=e.employee_id, client_action=False,
                         message=f"{e.full_name}: {e.hours_per_week:g} hrs/wk is below the {plan.min_hours_per_week:g}-hour minimum"))
        return out, None
    if not e.hire_date:
        return out, None
    start = eligibility_date(e.hire_date, plan)
    if plan.effective_date and start > plan.effective_date:
        out.append(Issue(code="IN_WAITING_PERIOD", severity=Severity.LOW, category="Eligibility",
                         employee_id=e.employee_id, client_action=False,
                         message=f"{e.full_name}: in waiting period, coverage starts {start:%m/%d/%Y}"))
    return out, start


# --------------------------------------------------------------------------- cross-record checks

def find_duplicates(employees: list[Employee]) -> tuple[list[Issue], set[int]]:
    """Return issues + indexes of rows that are duplicate copies (to exclude downstream)."""
    issues, dup_idx = [], set()
    seen_ssn, seen_person, seen_id = {}, {}, {}
    for i, e in enumerate(employees):
        if e.employee_id in seen_id:
            issues.append(Issue(code="DUPLICATE_ID", severity=Severity.HIGH, category="Duplicates",
                                employee_id=e.employee_id,
                                message=f"Employee ID {e.employee_id} is used on more than one employee row"))
        seen_id.setdefault(e.employee_id, i)

        person_key = (e.first_name.lower(), e.last_name.lower(), e.dob)
        orig = None
        if e.ssn and SSN_OK.match(e.ssn) and e.ssn in seen_ssn:
            orig = seen_ssn[e.ssn]
        elif e.dob and person_key in seen_person:
            orig = seen_person[person_key]
        if orig is not None:
            o = employees[orig]
            issues.append(Issue(code="DUPLICATE_EMPLOYEE", severity=Severity.HIGH, category="Duplicates",
                                employee_id=o.employee_id,
                                message=f"{e.full_name} appears twice (IDs {o.employee_id} and {e.employee_id}, "
                                        f"rows {o.source_row} and {e.source_row})"))
            dup_idx.add(i)
            continue
        if e.ssn and SSN_OK.match(e.ssn):
            seen_ssn[e.ssn] = i
        if e.dob:
            seen_person[person_key] = i
    return issues, dup_idx


def load_payroll(payroll_df: pd.DataFrame) -> dict[str, dict]:
    cm = map_columns(payroll_df)
    inv = {f: h for h, f in cm.mapping.items()}
    if "employee_id" not in inv:
        return {}
    out = {}
    for row in payroll_df.to_dict("records"):
        eid = str(row[inv["employee_id"]]).strip()
        out[eid] = {
            "annual_salary": parse_number(row.get(inv.get("annual_salary", ""), None)),
            "hours_per_week": parse_number(row.get(inv.get("hours_per_week", ""), None)),
        }
    return out


def reconcile_payroll(employees: list[Employee], payroll: dict[str, dict]) -> list[Issue]:
    out = []
    census_ids = {e.employee_id for e in employees}
    for e in employees:
        p = payroll.get(e.employee_id)
        if p is None:
            out.append(Issue(code="NOT_IN_PAYROLL", severity=Severity.MEDIUM, category="Payroll reconciliation",
                             employee_id=e.employee_id,
                             message=f"{e.full_name} ({e.employee_id}) is on the census but not in payroll (terminated?)"))
            continue
        if e.annual_salary and p["annual_salary"]:
            diff = abs(e.annual_salary - p["annual_salary"]) / p["annual_salary"]
            if diff > SALARY_TOLERANCE:
                out.append(Issue(code="SALARY_MISMATCH", severity=Severity.MEDIUM, category="Payroll reconciliation",
                                 employee_id=e.employee_id,
                                 message=f"{e.full_name}: census salary ${e.annual_salary:,.0f} vs payroll "
                                         f"${p['annual_salary']:,.0f} ({diff:.0%} difference, affects salary-based benefits)"))
        if e.hours_per_week and p["hours_per_week"] and abs(e.hours_per_week - p["hours_per_week"]) >= 5:
            out.append(Issue(code="HOURS_MISMATCH", severity=Severity.MEDIUM, category="Payroll reconciliation",
                             employee_id=e.employee_id,
                             message=f"{e.full_name}: census says {e.hours_per_week:g} hrs/wk, payroll says {p['hours_per_week']:g}"))
    for eid in payroll:
        if eid not in census_ids:
            out.append(Issue(code="NOT_IN_CENSUS", severity=Severity.MEDIUM, category="Payroll reconciliation",
                             employee_id=eid,
                             message=f"Payroll employee {eid} is missing from the census (new hire not listed?)"))
    return out


# --------------------------------------------------------------------------- orchestrator

def run_rules(employees: list[Employee], plan: PlanRules, payroll_df: pd.DataFrame | None = None,
              as_of: date | None = None) -> RuleResult:
    as_of = as_of or date.today()
    coverage_date = plan.effective_date or as_of
    res = RuleResult()

    dup_issues, dup_idx = find_duplicates(employees)
    res.issues += dup_issues
    unique = [e for i, e in enumerate(employees) if i not in dup_idx]

    for e in unique:
        res.issues += check_required_fields(e)
        res.issues += check_dates(e, as_of)
        res.issues += check_dependents(e, plan, coverage_date)
        elig_issues, start = check_eligibility(e, plan)
        res.issues += elig_issues
        res.eligibility[e.employee_id] = start

    if payroll_df is not None:
        res.issues += reconcile_payroll(unique, load_payroll(payroll_df))

    # ---- ACA employer-mandate math (simplified monthly FTE calculation)
    ft = [e for e in unique if (e.hours_per_week or 0) >= ACA_FT_HOURS]
    pt_monthly_hours = sum((e.hours_per_week or 0) * 52 / 12 for e in unique if 0 < (e.hours_per_week or 0) < ACA_FT_HOURS)
    fte = len(ft) + pt_monthly_hours / 120
    is_ale = fte >= ACA_ALE_THRESHOLD
    if is_ale:
        res.issues.append(Issue(code="ACA_ALE", severity=Severity.LOW, category="Compliance", client_action=False,
                                message=f"~{fte:.0f} full-time equivalents: Applicable Large Employer. Employer mandate "
                                        f"applies; Forms 1094-C/1095-C must be filed."))

    high_ids = {i.employee_id for i in res.issues if i.severity == Severity.HIGH and i.employee_id}
    eligible_now = [eid for eid, d in res.eligibility.items() if d and plan.effective_date and d <= plan.effective_date]
    res.stats = {
        "employees": len(unique),
        "dependents": sum(len(e.dependents) for e in unique),
        "full_time": len(ft),
        "part_time": len(unique) - len(ft),
        "fte": round(fte, 1),
        "aca_ale": is_ale,
        "eligible_on_effective_date": len(eligible_now),
        "in_waiting_period": sum(1 for i in res.issues if i.code == "IN_WAITING_PERIOD"),
        "ready_to_enroll": len([eid for eid in eligible_now if eid not in high_ids]),
        "issues_high": sum(1 for i in res.issues if i.severity == Severity.HIGH),
        "issues_medium": sum(1 for i in res.issues if i.severity == Severity.MEDIUM),
        "issues_low": sum(1 for i in res.issues if i.severity == Severity.LOW),
    }
    return res

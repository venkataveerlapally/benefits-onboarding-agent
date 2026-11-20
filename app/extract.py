"""Turn messy client files into clean Pydantic models.

Design principle: the LLM is used only for the *fuzzy* part (figuring out that "Hire Dt"
means hire_date, or reading plan terms out of prose). Parsing values and every compliance
decision stays in deterministic Python so results are testable and auditable.
"""
from __future__ import annotations

import difflib
import json
import re
from datetime import date, datetime, timedelta

import pandas as pd

from . import llm
from .models import ColumnMapping, Dependent, Employee, Issue, PlanRules, Relationship, Severity

CANONICAL_FIELDS = {
    "employee_id": ["empid", "employeeid", "employeenumber", "empno", "employeeno", "memberid", "eeid", "id", "badge"],
    "first_name": ["firstname", "fname", "givenname", "first", "forename"],
    "last_name": ["lastname", "lname", "surname", "last", "familyname"],
    "ssn": ["ssn", "socialsecurity", "socialsecuritynumber", "taxid", "tin", "ssnumber", "social"],
    "dob": ["dob", "birthdate", "dateofbirth", "birthday", "bdate", "birthdt"],
    "hire_date": ["hiredate", "dateofhire", "startdate", "hiredt", "doh", "employmentdate", "datehired"],
    "hours_per_week": ["hrswk", "hoursperweek", "weeklyhours", "scheduledhours", "hours", "hrs", "stdhours"],
    "annual_salary": ["annualsalary", "salary", "basepay", "comp", "compensation", "annualpay", "wages", "annualcomp"],
    "email": ["email", "emailaddress", "workemail", "mail"],
    "relationship": ["relationship", "reltype", "membertype", "relation", "rel", "relcode", "dependenttype"],
}
REQUIRED_COLUMNS = {"employee_id", "first_name", "last_name"}

SSN_RE = re.compile(r"\b\d{3}-?\d{2}-?(\d{4})\b")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# --------------------------------------------------------------------------- column mapping

def heuristic_map(headers: list[str]) -> dict[str, str]:
    mapping, taken = {}, set()
    lookup = {syn: field for field, syns in CANONICAL_FIELDS.items() for syn in syns}
    # pass 1: exact synonym match
    for h in headers:
        f = lookup.get(_norm(h))
        if f and f not in taken:
            mapping[h] = f; taken.add(f)
    # pass 2: fuzzy match for whatever is left
    for h in headers:
        if h in mapping:
            continue
        best = difflib.get_close_matches(_norm(h), list(lookup), n=1, cutoff=0.75)
        if best and lookup[best[0]] not in taken:
            mapping[h] = lookup[best[0]]; taken.add(lookup[best[0]])
    return mapping


def _mask_pii(value) -> str:
    """Never send raw SSNs to an external API - mask all but the last 4."""
    return SSN_RE.sub(lambda m: f"XXX-XX-{m.group(1)}", str(value))


def map_columns(df: pd.DataFrame) -> ColumnMapping:
    headers = list(df.columns)
    heuristic = heuristic_map(headers)
    method = "heuristic"
    mapping = dict(heuristic)

    if llm.available():
        sample = [{h: _mask_pii(v) for h, v in row.items()} for row in df.head(3).fillna("").to_dict("records")]
        prompt = (
            "Map each spreadsheet header from a benefits census file to one of these canonical fields, "
            f"or null if none apply: {list(CANONICAL_FIELDS)}.\n"
            f"Headers: {json.dumps(headers)}\nSample rows (PII masked): {json.dumps(sample, default=str)}\n"
            'Return {"mapping": {"<header>": "<field or null>"}}. Each canonical field at most once.'
        )
        result = llm.complete_json("You are a meticulous benefits data analyst.", prompt)
        if result and isinstance(result.get("mapping"), dict):
            llm_map = {h: f for h, f in result["mapping"].items() if h in headers and f in CANONICAL_FIELDS}
            if len(set(llm_map.values())) == len(llm_map):  # reject if it double-assigned a field
                # LLM wins, heuristic fills any gaps it left
                used = set(llm_map.values())
                for h, f in heuristic.items():
                    if h not in llm_map and f not in used:
                        llm_map[h] = f
                mapping, method = llm_map, "llm"

    return ColumnMapping(mapping=mapping, unmapped=[h for h in headers if h not in mapping], method=method)


# --------------------------------------------------------------------------- value parsing

DATE_FORMATS = ["%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%d-%b-%Y", "%b %d, %Y", "%m-%d-%Y", "%Y/%m/%d"]


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and not v.strip()) or v is pd.NaT


def parse_date(v) -> tuple[date | None, bool]:
    """Return (date, ok). ok=False means a value was present but couldn't be parsed."""
    if _blank(v):
        return None, True
    if isinstance(v, pd.Timestamp):
        return (None, True) if pd.isna(v) else (v.date(), True)
    if isinstance(v, datetime):
        return v.date(), True
    if isinstance(v, date):
        return v, True
    if isinstance(v, (int, float)) and 20000 < float(v) < 80000:  # Excel serial date
        return date(1899, 12, 30) + timedelta(days=int(v)), True
    s = str(v).strip()
    if s.endswith(" 00:00:00"):
        s = s[:-9]
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date(), True
        except ValueError:
            pass
    return None, False


def parse_number(v) -> float | None:
    if _blank(v):
        return None
    try:
        return float(re.sub(r"[$,\s]", "", str(v)))
    except ValueError:
        return None


def normalize_ssn(v) -> str | None:
    if _blank(v):
        return None
    s = str(v).strip()
    if s.endswith(".0"):  # Excel turned it into a float
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    if len(digits) == 9:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    return s  # keep the bad value so the rules engine can flag it


REL_MAP = {
    Relationship.EMPLOYEE: {"ee", "employee", "self", "emp", "subscriber", "member", "e"},
    Relationship.SPOUSE: {"sp", "spouse", "wife", "husband", "domesticpartner", "dp", "s"},
    Relationship.CHILD: {"ch", "child", "dependent", "dep", "son", "daughter", "c", "stepchild"},
}


def parse_relationship(v) -> Relationship | None:
    if _blank(v):
        return Relationship.EMPLOYEE
    n = _norm(str(v))
    for rel, codes in REL_MAP.items():
        if n in codes:
            return rel
    return None


def _str(v) -> str:
    if _blank(v):
        return ""
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


# --------------------------------------------------------------------------- build models

def build_employees(df: pd.DataFrame, cm: ColumnMapping) -> tuple[list[Employee], list[Issue]]:
    issues: list[Issue] = []
    missing_cols = REQUIRED_COLUMNS - set(cm.mapping.values())
    if missing_cols:
        issues.append(Issue(code="MISSING_COLUMN", severity=Severity.HIGH, category="File structure",
                            message=f"Census is missing required column(s): {', '.join(sorted(missing_cols))}"))
        return [], issues

    inv = {field: header for header, field in cm.mapping.items()}

    def get(row, field):
        h = inv.get(field)
        return row.get(h) if h else None

    employees: list[Employee] = []
    by_id: dict[str, Employee] = {}
    last_emp: Employee | None = None

    for idx, row in enumerate(df.to_dict("records"), start=2):  # start=2: header is row 1 in Excel
        rel = parse_relationship(get(row, "relationship"))
        eid = _str(get(row, "employee_id"))
        dob, dob_ok = parse_date(get(row, "dob"))
        if not dob_ok:
            issues.append(Issue(code="INVALID_DATE", severity=Severity.MEDIUM, category="Data format",
                                employee_id=eid or None, field="dob",
                                message=f"Row {idx}: couldn't read date of birth '{get(row, 'dob')}'"))
        if rel is None:
            issues.append(Issue(code="UNKNOWN_RELATIONSHIP", severity=Severity.MEDIUM, category="Data format",
                                employee_id=eid or None,
                                message=f"Row {idx}: unrecognized relationship code '{get(row, 'relationship')}'"))
            continue

        if rel == Relationship.EMPLOYEE:
            hire, hire_ok = parse_date(get(row, "hire_date"))
            if not hire_ok:
                issues.append(Issue(code="INVALID_DATE", severity=Severity.MEDIUM, category="Data format",
                                    employee_id=eid, field="hire_date",
                                    message=f"Row {idx}: couldn't read hire date '{get(row, 'hire_date')}'"))
            email = _str(get(row, "email")) or None
            emp = Employee(
                employee_id=eid or f"ROW{idx}", first_name=_str(get(row, "first_name")),
                last_name=_str(get(row, "last_name")), ssn=normalize_ssn(get(row, "ssn")), dob=dob,
                hire_date=hire, hours_per_week=parse_number(get(row, "hours_per_week")),
                annual_salary=parse_number(get(row, "annual_salary")), email=email, source_row=idx,
            )
            employees.append(emp)
            by_id.setdefault(emp.employee_id, emp)
            last_emp = emp
        else:
            parent = by_id.get(eid) or last_emp
            if parent is None:
                issues.append(Issue(code="ORPHAN_DEPENDENT", severity=Severity.HIGH, category="Dependents",
                                    message=f"Row {idx}: dependent {_str(get(row, 'first_name'))} has no matching employee"))
                continue
            parent.dependents.append(Dependent(
                first_name=_str(get(row, "first_name")), last_name=_str(get(row, "last_name")),
                relationship=rel, dob=dob, ssn=normalize_ssn(get(row, "ssn")),
            ))
    return employees, issues


# --------------------------------------------------------------------------- plan rules

def _regex_plan_rules(text: str) -> dict:
    out: dict = {}
    flat = re.sub(r"\s+", " ", text)
    if m := re.search(r"^(.*?)\s+-\s+Summary of Benefits", text.strip(), re.M):
        out["plan_name"] = m.group(1).strip()
    if m := re.search(r"effective date:?\s*([A-Z][a-z]+ \d{1,2}, \d{4})", flat, re.I):
        out["effective_date"] = datetime.strptime(m.group(1), "%B %d, %Y").date()
    if m := re.search(r"following (\d+) days", flat, re.I):
        out["waiting_period_days"] = int(m.group(1))
    if m := re.search(r"at least (\d+(?:\.\d+)?) hours per week", flat, re.I):
        out["min_hours_per_week"] = float(m.group(1))
    if m := re.search(r"(?:turn|through) age (\d+)", flat, re.I):
        out["dependent_age_limit"] = int(m.group(1))
    if m := re.search(r"deductible is \$([\d,]+) per individual and \$([\d,]+) per family", flat, re.I):
        out["deductible_individual"] = float(m.group(1).replace(",", ""))
        out["deductible_family"] = float(m.group(2).replace(",", ""))
    if m := re.search(r"out-of-pocket maximum is \$([\d,]+) per individual", flat, re.I):
        out["oop_max_individual"] = float(m.group(1).replace(",", ""))
    return out


def extract_plan_rules(pages: list[tuple[int, str]]) -> PlanRules:
    text = "\n".join(t for _, t in pages)
    regex_vals = _regex_plan_rules(text)

    if llm.available():
        fields = list(PlanRules.model_fields)
        fields.remove("extraction_method")
        result = llm.complete_json(
            "You extract structured terms from health plan documents. Use null for anything not stated.",
            f"Extract these fields: {fields}. Dates as YYYY-MM-DD, money as numbers.\n\nDOCUMENT:\n{text[:15000]}",
        )
        if result:
            merged = {**regex_vals, **{k: v for k, v in result.items() if k in fields and v is not None}}
            try:
                return PlanRules(**merged, extraction_method="llm")
            except Exception:
                pass  # bad LLM output -> fall through to regex

    return PlanRules(**regex_vals, extraction_method="regex")

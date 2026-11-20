"""Generate realistic, intentionally messy synthetic client data.

For each fake client we write:
  census.(xlsx|csv)   - employees + dependents, client-specific column names/date formats
  payroll.csv         - payroll export (should agree with the census... mostly)
  plan_summary.pdf    - a plan summary document
  ground_truth.json   - every error we planted, so evaluate.py can score the agent

100% fake data. No real people, no real SSNs (all start with 9xx, which the SSA never issues).
Deterministic: same seed -> same files.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"

FIRST = ["James", "Maria", "Wei", "Aisha", "Carlos", "Priya", "Liam", "Fatima", "Noah", "Sofia",
         "Ethan", "Chloe", "Mateo", "Zara", "Lucas", "Amara", "Daniel", "Yuki", "Omar", "Elena",
         "Samuel", "Grace", "Arjun", "Hannah", "Diego", "Leila", "Ryan", "Mei", "Isaac", "Nadia",
         "Kofi", "Olivia", "Ravi", "Emma", "Hugo", "Ines", "Jamal", "Clara", "Tariq", "Ava"]
LAST = ["Smith", "Garcia", "Chen", "Khan", "Rodriguez", "Patel", "Johnson", "Nguyen", "Williams",
        "Okafor", "Brown", "Kim", "Martinez", "Ali", "Davis", "Silva", "Miller", "Tanaka", "Wilson",
        "Haddad", "Moore", "Novak", "Taylor", "Reyes", "Anderson", "Mensah", "Thomas", "Lopez",
        "Jackson", "Ivanova", "White", "Sato", "Harris", "Cohen", "Martin", "Rossi", "Clark", "Diaz"]

EFFECTIVE = date(2027, 1, 1)

# Each client has its own "dialect": column names, date format, relationship codes, file type.
CLIENTS = [
    {
        "slug": "northwind_bakery", "name": "Northwind Bakery", "n": 25, "file": "xlsx",
        "waiting_days": 30, "plan": "Evergreen PPO 1500", "ded": (1500, 3000), "oop": 6000,
        "cols": ["Emp ID", "First Name", "Last Name", "SSN", "DOB", "Date of Hire", "Hrs/Wk",
                 "Annual Salary", "Email", "Relationship"],
        "date_fmt": "%m/%d/%Y", "rel": {"employee": "EE", "spouse": "SP", "child": "CH"},
        "money": "plain",
    },
    {
        "slug": "brightline_logistics", "name": "Brightline Logistics", "n": 80, "file": "csv",
        "waiting_days": 60, "plan": "Summit HMO 2500", "ded": (2500, 5000), "oop": 7500,
        "cols": ["employee_number", "fname", "lname", "social_security", "birth_date",
                 "start_date", "weekly_hours", "base_pay", "work_email", "rel_type"],
        "date_fmt": "%Y-%m-%d", "rel": {"employee": "Employee", "spouse": "Spouse", "child": "Child"},
        "money": "dollar",
    },
    {
        "slug": "cedar_health_partners", "name": "Cedar Health Partners", "n": 200, "file": "xlsx",
        "waiting_days": 90, "plan": "Harbor PPO 1000", "ded": (1000, 2000), "oop": 5000,
        "cols": ["Member ID", "Given Name", "Surname", "Tax ID", "Birth Date", "Hire Dt",
                 "Scheduled Hours", "Comp", "E-mail Address", "Member Type"],
        "date_fmt": "excel", "rel": {"employee": "Self", "spouse": "Spouse", "child": "Dependent"},
        "money": "plain",
    },
]


def rand_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


class SSNFactory:
    """Unique fake SSNs in the 900-999 area range (never issued by the SSA)."""
    def __init__(self, rng: random.Random):
        self.rng, self.used = rng, set()

    def __call__(self) -> str:
        while True:
            s = f"9{self.rng.randint(0, 99):02d}-{self.rng.randint(10, 99)}-{self.rng.randint(1000, 9999)}"
            if s not in self.used:
                self.used.add(s)
                return s


def build_people(cfg: dict, rng: random.Random):
    """Return clean employee records (with dependents) and payroll rows."""
    ssn = SSNFactory(rng)
    used_names: set[tuple[str, str]] = set()
    employees = []
    for i in range(cfg["n"]):
        while True:
            fn, ln = rng.choice(FIRST), rng.choice(LAST)
            if (fn, ln) not in used_names:
                used_names.add((fn, ln))
                break
        dob = rand_date(rng, date(1962, 1, 1), date(2002, 12, 31))
        part_time = rng.random() < 0.15
        emp = {
            "id": f"{cfg['slug'][:2].upper()}{1001 + i}",
            "first": fn, "last": ln, "ssn": ssn(), "dob": dob,
            # most hired long ago; ~10% recent hires who will still be in the waiting period
            "hire": rand_date(rng, date(2026, 9, 15), date(2026, 11, 15)) if rng.random() < 0.10
            else rand_date(rng, date(2012, 1, 1), date(2026, 6, 30)),
            "hours": float(rng.choice([16, 20, 24, 25])) if part_time else float(rng.choice([35, 37.5, 40, 40, 40, 45])),
            "salary": round(rng.uniform(32000, 145000), -2),
            "email": f"{fn.lower()}.{ln.lower()}@{cfg['slug'].replace('_', '')}.example.com",
            "deps": [],
        }
        if rng.random() < 0.45:
            emp["deps"].append({"first": rng.choice(FIRST), "last": ln, "rel": "spouse",
                                "dob": rand_date(rng, date(1962, 1, 1), date(2000, 12, 31)), "ssn": ssn()})
        for _ in range(rng.choice([0, 0, 1, 2, 3])):
            emp["deps"].append({"first": rng.choice(FIRST), "last": ln, "rel": "child",
                                "dob": rand_date(rng, date(2003, 6, 1), date(2025, 12, 31)), "ssn": ssn()})
        employees.append(emp)
    return employees


def plant_errors(cfg: dict, employees: list[dict], rng: random.Random):
    """Mutate the clean data to plant realistic errors. Returns (employees, payroll, truth)."""
    truth: list[dict] = []
    scale = max(1, cfg["n"] // 40)  # bigger clients get more errors
    pool = employees[:]
    rng.shuffle(pool)

    def take():
        return pool.pop()

    for _ in range(scale):
        e = take(); e["ssn"] = None
        truth.append({"code": "MISSING_SSN", "employee_id": e["id"]})
    for _ in range(scale):
        e = take(); e["ssn"] = e["ssn"].replace("-", "")[:7]  # truncated SSN
        truth.append({"code": "INVALID_SSN", "employee_id": e["id"]})
    for _ in range(max(1, scale // 2)):
        e = take(); e["dob"] = None
        truth.append({"code": "MISSING_DOB", "employee_id": e["id"]})
    for _ in range(scale):
        e = take()
        e["deps"].append({"first": rng.choice(FIRST), "last": e["last"], "rel": "child",
                          "dob": date(rng.randint(1997, 2000), rng.randint(1, 12), rng.randint(1, 28)), "ssn": None})
        truth.append({"code": "DEPENDENT_OVER_AGE", "employee_id": e["id"]})
    for _ in range(max(1, scale // 2)):
        e = take(); e["hire"] = date(2028, rng.randint(1, 12), 1)
        truth.append({"code": "FUTURE_HIRE_DATE", "employee_id": e["id"]})
    for _ in range(max(1, scale // 2)):
        e = take(); e["email"] = e["email"].replace("@", "_at_")
        truth.append({"code": "INVALID_EMAIL", "employee_id": e["id"]})

    # payroll is built from the (mostly) clean data, then diverges
    payroll = [{"EmployeeID": e["id"], "AnnualSalary": e["salary"], "ScheduledHours": e["hours"]}
               for e in employees]
    by_id = {p["EmployeeID"]: p for p in payroll}

    for _ in range(scale + 1):
        e = take()
        by_id[e["id"]]["AnnualSalary"] = round(e["salary"] * rng.choice([1.12, 0.85, 1.25]), -2)
        truth.append({"code": "SALARY_MISMATCH", "employee_id": e["id"]})
    for _ in range(max(1, scale // 2)):
        e = take()
        payroll.remove(by_id[e["id"]])
        truth.append({"code": "NOT_IN_PAYROLL", "employee_id": e["id"]})
    for k in range(max(1, scale // 2)):
        ghost = f"{cfg['slug'][:2].upper()}9{k:03d}"
        payroll.append({"EmployeeID": ghost, "AnnualSalary": 61000.0, "ScheduledHours": 40.0})
        truth.append({"code": "NOT_IN_CENSUS", "employee_id": ghost})

    # duplicate: same person entered twice (copy/paste error) - inserted later at a random spot
    dup_src = take()
    truth.append({"code": "DUPLICATE_EMPLOYEE", "employee_id": dup_src["id"]})
    return employees, payroll, truth, dup_src


def fmt_date(d: date | None, fmt: str):
    if d is None:
        return None
    return d if fmt == "excel" else d.strftime(fmt)


def fmt_money(v, style: str):
    if v is None:
        return None
    return f"${v:,.2f}" if style == "dollar" else v


def to_census_rows(cfg: dict, employees: list[dict], dup_src: dict, rng: random.Random):
    c, rel = cfg["cols"], cfg["rel"]
    rows = []

    def emp_row(e):
        return {c[0]: e["id"], c[1]: e["first"], c[2]: e["last"], c[3]: e["ssn"],
                c[4]: fmt_date(e["dob"], cfg["date_fmt"]), c[5]: fmt_date(e["hire"], cfg["date_fmt"]),
                c[6]: e["hours"], c[7]: fmt_money(e["salary"], cfg["money"]), c[8]: e["email"],
                c[9]: rel["employee"]}

    for e in employees:
        rows.append(emp_row(e))
        for d in e["deps"]:
            rows.append({c[0]: e["id"], c[1]: d["first"], c[2]: d["last"], c[3]: d["ssn"],
                         c[4]: fmt_date(d["dob"], cfg["date_fmt"]), c[5]: None, c[6]: None,
                         c[7]: None, c[8]: None, c[9]: rel[d["rel"]]})
    # duplicate row goes near the end, with a different (typo'd) ID - like a real re-entry
    dup = emp_row(dup_src)
    dup[c[0]] = dup_src["id"] + "-B"
    rows.insert(len(rows) - rng.randint(1, 5), dup)
    return pd.DataFrame(rows, columns=c)


def write_plan_pdf(cfg: dict, path: Path):
    s = getSampleStyleSheet()
    ded_i, ded_f = cfg["ded"]
    sections = [
        ("Plan Overview",
         f"This Summary of Benefits describes the {cfg['plan']} medical plan offered by {cfg['name']} "
         f"to eligible employees and their dependents. Plan effective date: January 1, 2027. "
         f"This is a fictional plan created for demonstration purposes."),
        ("Employee Eligibility",
         "Employees who are regularly scheduled to work at least 30 hours per week are eligible to enroll. "
         "Part-time, seasonal, and temporary employees scheduled for fewer than 30 hours per week are not eligible."),
        ("Waiting Period",
         f"Coverage begins on the first day of the month following {cfg['waiting_days']} days of continuous employment. "
         f"Employees hired before the plan effective date who have satisfied the waiting period are eligible on January 1, 2027."),
        ("Dependent Eligibility",
         "Eligible dependents include your legal spouse and your children through the end of the month in which they turn age 26, "
         "regardless of student or marital status. Children over age 26 may remain covered only if they are "
         "incapable of self-support due to a disability, with documentation approved by the plan."),
        ("Deductible",
         f"The calendar-year in-network deductible is ${ded_i:,} per individual and ${ded_f:,} per family. "
         f"Preventive care is covered at 100% and is not subject to the deductible."),
        ("Out-of-Pocket Maximum",
         f"The in-network out-of-pocket maximum is ${cfg['oop']:,} per individual and ${cfg['oop'] * 2:,} per family. "
         f"Once reached, the plan pays 100% of covered in-network services for the rest of the calendar year."),
        ("Copays",
         "Primary care office visit: $25 copay. Specialist visit: $50 copay. Urgent care: $75 copay. "
         "Emergency room: $350 copay, waived if admitted. Telehealth visits: $0 copay."),
        ("Prescription Drugs",
         "Generic drugs: $10 copay. Preferred brand drugs: $40 copay. Non-preferred brand drugs: $75 copay. "
         "Specialty drugs: 20% coinsurance up to $250 per prescription. Mail order provides a 90-day supply for two copays."),
        ("Special Enrollment and Qualifying Life Events",
         "Outside of open enrollment you may add or drop coverage only after a qualifying life event, such as marriage, "
         "divorce, birth or adoption of a child, or loss of other coverage. You must request the change within 30 days of the event. "
         "Coverage for a newborn is effective from the date of birth if enrolled within 30 days."),
        ("COBRA Continuation",
         "If coverage ends because of termination of employment or reduction in hours, you and your covered dependents may be "
         "eligible to continue coverage under COBRA for up to 18 months at your own cost, plus a 2% administrative fee."),
    ]
    story = [Paragraph(f"{cfg['plan']} - Summary of Benefits", s["Title"]), Spacer(1, 12)]
    for head, body in sections:
        story += [Paragraph(head, s["Heading2"]), Paragraph(body, s["BodyText"]), Spacer(1, 8)]
    SimpleDocTemplate(str(path), pagesize=letter).build(story)


def main(seed: int = 42):
    for i, cfg in enumerate(CLIENTS):
        rng = random.Random(seed + i)
        d = OUT / cfg["slug"]
        d.mkdir(parents=True, exist_ok=True)
        employees = build_people(cfg, rng)
        employees, payroll, truth, dup_src = plant_errors(cfg, employees, rng)
        census = to_census_rows(cfg, employees, dup_src, rng)
        if cfg["file"] == "xlsx":
            census.to_excel(d / "census.xlsx", index=False)
        else:
            census.to_csv(d / "census.csv", index=False)
        pd.DataFrame(payroll).to_csv(d / "payroll.csv", index=False)
        write_plan_pdf(cfg, d / "plan_summary.pdf")
        meta = {"client": cfg["name"], "plan": cfg["plan"], "effective_date": EFFECTIVE.isoformat(),
                "waiting_period_days": cfg["waiting_days"], "census_columns": cfg["cols"],
                "planted_errors": truth}
        (d / "ground_truth.json").write_text(json.dumps(meta, indent=2))
        print(f"{cfg['name']:<24} {cfg['n']:>4} employees, {len(census):>4} census rows, "
              f"{len(truth):>2} planted errors -> {d.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 42)

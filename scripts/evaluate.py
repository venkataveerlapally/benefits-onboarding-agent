"""Score the agent against the planted errors in ground_truth.json and a labeled Q&A set.

Metrics:
  - Column mapping accuracy (vs the known canonical mapping for each client)
  - Error detection recall + precision, per planted error code
  - Q&A: right section retrieved, and correct handoff on out-of-scope questions
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.extract import CANONICAL_FIELDS  # noqa: E402
from app.pipeline import run_onboarding  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
CANON_ORDER = ["employee_id", "first_name", "last_name", "ssn", "dob", "hire_date",
               "hours_per_week", "annual_salary", "email", "relationship"]

QA_SET = [  # (question, expected section or None = should hand off)
    ("Can my 27 year old son stay on my plan?", "Dependent Eligibility"),
    ("When does my coverage start?", "Waiting Period"),
    ("How much are generic meds?", "Prescription Drugs"),
    ("What's my deductible?", "Deductible"),
    ("How much does a doctor visit cost?", "Copays"),
    ("I just had a baby, how do I add her?", "Special Enrollment and Qualifying Life Events"),
    ("What happens to my insurance if I quit?", "COBRA Continuation"),
    ("What's the most I'd pay in a year?", "Out-of-Pocket Maximum"),
    ("I work 20 hours a week, can I enroll?", "Employee Eligibility"),
    ("Does the plan cover my dog's vet bills?", None),
    ("What's the 401k match?", None),
    ("Can I expense my gym membership?", None),
]


def main():
    totals = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    map_correct = map_total = 0
    qa_correct = qa_total = 0

    for d in sorted(p for p in DATA.iterdir() if p.is_dir()):
        truth = json.loads((d / "ground_truth.json").read_text())
        r = run_onboarding(truth["client"], next(d.glob("census.*")), d / "plan_summary.pdf", d / "payroll.csv")

        # column mapping
        expected = dict(zip(truth["census_columns"], CANON_ORDER))
        for h, f in expected.items():
            map_total += 1
            map_correct += r.mapping.mapping.get(h) == f

        # error detection, only for the codes we planted
        planted = {(t["code"], t["employee_id"]) for t in truth["planted_errors"]}
        codes = {c for c, _ in planted}
        found = {(i.code, i.employee_id) for i in r.issues if i.code in codes}
        for code in codes:
            p = {x for x in planted if x[0] == code}
            f = {x for x in found if x[0] == code}
            totals[code]["tp"] += len(p & f)
            totals[code]["fn"] += len(p - f)
            totals[code]["fp"] += len(f - p)

        # Q&A
        for q, section in QA_SET:
            a = r.qa.ask(q)
            qa_total += 1
            if section is None:
                qa_correct += a["handoff"]
            else:
                qa_correct += (not a["handoff"]) and bool(a["citations"]) and a["citations"][0]["section"] == section

        print(f"{truth['client']:<24} planted={len(planted):>2}  caught={len(planted & found):>2}  "
              f"false_pos={len(found - planted)}")

    print("\nColumn mapping accuracy: "
          f"{map_correct}/{map_total} ({map_correct / map_total:.0%})")
    print("\nError detection by type:")
    print(f"  {'code':<22}{'recall':>8}{'precision':>11}")
    tp = fn = fp = 0
    for code, t in sorted(totals.items()):
        rec = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else 1
        prec = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else 1
        print(f"  {code:<22}{rec:>8.0%}{prec:>11.0%}")
        tp, fn, fp = tp + t["tp"], fn + t["fn"], fp + t["fp"]
    print(f"  {'OVERALL':<22}{tp / (tp + fn):>8.0%}{tp / (tp + fp):>11.0%}   ({tp}/{tp + fn} planted errors caught)")
    print(f"\nPlan Q&A accuracy (right section or correct handoff): {qa_correct}/{qa_total} ({qa_correct / qa_total:.0%})")


if __name__ == "__main__":
    main()

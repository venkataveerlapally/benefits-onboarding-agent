"""Run the full pipeline on a synthetic client and print a report.

    python scripts/run_demo.py                    # all clients
    python scripts/run_demo.py northwind_bakery   # one client
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import run_onboarding  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
ICON = {"done": "[x]", "todo": "[ ]", "needs_review": "[?]", "blocked": "[!]"}


def client_files(slug: str):
    d = DATA / slug
    census = next(d.glob("census.*"))
    return census, d / "plan_summary.pdf", d / "payroll.csv"


def report(slug: str, show_email: bool = True):
    census, plan, payroll = client_files(slug)
    name = slug.replace("_", " ").title()
    r = run_onboarding(name, census, plan, payroll)
    print("=" * 78)
    print(f"{name}  |  plan: {r.plan.plan_name}  |  mapping: {r.mapping.method}  |  {sum(r.timings.values()):.2f}s")
    print("=" * 78)
    print("Stats:", r.stats)
    print("\nIssues by code:", dict(Counter(i.code for i in r.issues).most_common()))
    print("\nChecklist:")
    for c in r.checklist:
        print(f"  {ICON[c['status']]} {c['task']:<42} {c['detail']}")
    if show_email:
        print("\n--- Client email draft (" + r.email_method + ") ---")
        print(r.email)
    if r.qa:
        print("\n--- Plan Q&A ---")
        for q in ["Can my 27 year old son stay on my plan?", "When does my coverage start?",
                  "How much are generic meds?", "Does the plan cover my dog's vet bills?"]:
            a = r.qa.ask(q)
            print(f"Q: {q}\nA: {a['answer']}  [conf={a['confidence']}, {a['method']}]\n")


if __name__ == "__main__":
    slugs = sys.argv[1:] or sorted(p.name for p in DATA.iterdir() if p.is_dir())
    for s in slugs:
        report(s, show_email=len(slugs) == 1)

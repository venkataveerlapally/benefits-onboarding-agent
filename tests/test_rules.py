from datetime import date

import pandas as pd
import pytest

from app.models import Dependent, Employee, PlanRules, Relationship
from app.rules import (age_on, eligibility_date, find_duplicates, reconcile_payroll, run_rules)

PLAN = PlanRules(plan_name="Test PPO", effective_date=date(2027, 1, 1), waiting_period_days=60,
                 min_hours_per_week=30, dependent_age_limit=26)
AS_OF = date(2026, 11, 20)


def emp(**kw):
    base = dict(employee_id="E1", first_name="Ana", last_name="Lee", ssn="900-12-3456",
                dob=date(1990, 5, 1), hire_date=date(2020, 1, 1), hours_per_week=40,
                annual_salary=60000, email="ana@x.com")
    base.update(kw)
    return Employee(**base)


def codes(employees, payroll=None):
    return [i.code for i in run_rules(employees, PLAN, payroll, as_of=AS_OF).issues]


def test_clean_employee_has_no_issues():
    assert codes([emp()]) == []


@pytest.mark.parametrize("field,value,code", [
    ("ssn", None, "MISSING_SSN"),
    ("ssn", "1234567", "INVALID_SSN"),
    ("dob", None, "MISSING_DOB"),
    ("hire_date", None, "MISSING_HIRE_DATE"),
    ("email", "ana_at_x.com", "INVALID_EMAIL"),
    ("hire_date", date(2028, 1, 1), "FUTURE_HIRE_DATE"),
    ("dob", date(2020, 1, 1), "IMPLAUSIBLE_AGE"),
])
def test_field_checks(field, value, code):
    assert code in codes([emp(**{field: value})])


def test_upcoming_hire_is_not_an_error():
    c = codes([emp(hire_date=date(2026, 12, 15))])
    assert "FUTURE_HIRE_DATE" not in c and "UPCOMING_HIRE" in c


def test_age_on_birthday_boundary():
    assert age_on(date(2001, 1, 1), date(2027, 1, 1)) == 26
    assert age_on(date(2001, 1, 2), date(2027, 1, 1)) == 25


def test_child_over_age_limit_flagged_but_25_year_old_is_fine():
    old = Dependent(first_name="Kid", last_name="Lee", relationship=Relationship.CHILD, dob=date(2000, 6, 1), ssn="900-11-1111")
    ok = Dependent(first_name="Kid2", last_name="Lee", relationship=Relationship.CHILD, dob=date(2001, 6, 1), ssn="900-11-2222")
    assert "DEPENDENT_OVER_AGE" in codes([emp(dependents=[old])])
    assert "DEPENDENT_OVER_AGE" not in codes([emp(dependents=[ok])])


def test_spouse_over_26_is_not_flagged():
    sp = Dependent(first_name="Sam", last_name="Lee", relationship=Relationship.SPOUSE, dob=date(1960, 1, 1), ssn="900-11-3333")
    assert "DEPENDENT_OVER_AGE" not in codes([emp(dependents=[sp])])


def test_waiting_period_math():
    # hired Nov 1 + 60 days = Dec 31 -> first of next month = Jan 1
    assert eligibility_date(date(2026, 11, 1), PLAN) == date(2027, 1, 1)
    # hired Nov 15 + 60 days = Jan 14 -> Feb 1
    assert eligibility_date(date(2026, 11, 15), PLAN) == date(2027, 2, 1)
    # long-tenured employees start on plan effective date, not earlier
    assert eligibility_date(date(2015, 3, 3), PLAN) == date(2027, 1, 1)


def test_part_time_is_ineligible():
    assert "INELIGIBLE_PART_TIME" in codes([emp(hours_per_week=20)])


def test_duplicate_by_ssn_and_by_name_dob():
    a, b = emp(), emp(employee_id="E1-B")
    issues, dup_idx = find_duplicates([a, b])
    assert [i.code for i in issues] == ["DUPLICATE_EMPLOYEE"] and dup_idx == {1}
    c = emp(employee_id="E2", ssn=None)
    issues, _ = find_duplicates([a, c])
    assert issues and issues[0].code == "DUPLICATE_EMPLOYEE"


def test_duplicate_copy_not_double_counted_in_payroll():
    payroll = pd.DataFrame([{"EmployeeID": "E1", "AnnualSalary": 60000, "ScheduledHours": 40}])
    c = codes([emp(), emp(employee_id="E1-B")], payroll)
    assert "NOT_IN_PAYROLL" not in c


def test_payroll_reconciliation():
    payroll = {"E1": {"annual_salary": 70000, "hours_per_week": 40}, "GHOST": {"annual_salary": 1, "hours_per_week": 40}}
    c = [i.code for i in reconcile_payroll([emp(), emp(employee_id="E2", ssn="900-99-9999")], payroll)]
    assert sorted(c) == ["NOT_IN_CENSUS", "NOT_IN_PAYROLL", "SALARY_MISMATCH"]


def test_salary_within_tolerance_not_flagged():
    payroll = {"E1": {"annual_salary": 61500, "hours_per_week": 40}}
    assert reconcile_payroll([emp()], payroll) == []


def test_aca_ale_threshold():
    staff = [emp(employee_id=f"E{i}", ssn=f"900-00-{i:04d}", first_name=f"N{i}") for i in range(50)]
    res = run_rules(staff, PLAN, as_of=AS_OF)
    assert res.stats["aca_ale"] is True
    res = run_rules(staff[:49], PLAN, as_of=AS_OF)
    assert res.stats["aca_ale"] is False

from datetime import date

import pandas as pd

from app.extract import (_mask_pii, build_employees, heuristic_map, map_columns, normalize_ssn,
                         parse_date, parse_number, parse_relationship)
from app.models import Relationship
from app.qa import PlanQA


def test_heuristic_maps_messy_headers():
    m = heuristic_map(["Emp #", "Given Name", "Surname", "Soc Sec Number", "Birth Dt", "Hire Dt",
                       "Hrs/Wk", "Comp", "E-mail", "Rel Type"])
    assert m["Given Name"] == "first_name"
    assert m["Surname"] == "last_name"
    assert m["Hire Dt"] == "hire_date"
    assert m["Hrs/Wk"] == "hours_per_week"
    assert m["Comp"] == "annual_salary"


def test_parse_date_formats():
    assert parse_date("03/25/1993") == (date(1993, 3, 25), True)
    assert parse_date("1993-03-25") == (date(1993, 3, 25), True)
    assert parse_date(34053) == (date(1993, 3, 25), True)      # Excel serial
    assert parse_date(pd.Timestamp("1993-03-25")) == (date(1993, 3, 25), True)
    assert parse_date(None) == (None, True)
    assert parse_date("not a date") == (None, False)


def test_parse_number_and_ssn():
    assert parse_number("$52,000.00") == 52000.0
    assert normalize_ssn("900123456") == "900-12-3456"
    assert normalize_ssn(900123456.0) == "900-12-3456"
    assert normalize_ssn("9001234") == "9001234"   # left as-is so rules flag it


def test_relationship_codes():
    assert parse_relationship("EE") == Relationship.EMPLOYEE
    assert parse_relationship("Self") == Relationship.EMPLOYEE
    assert parse_relationship("Dependent") == Relationship.CHILD
    assert parse_relationship("SP") == Relationship.SPOUSE
    assert parse_relationship("Grandparent") is None


def test_pii_is_masked_before_llm():
    assert _mask_pii("ssn 900-12-3456 here") == "ssn XXX-XX-3456 here"
    assert "900123456" not in _mask_pii("900123456")


def test_dependents_attach_to_employee():
    df = pd.DataFrame([
        {"Emp ID": "E1", "First Name": "Ana", "Last Name": "Lee", "SSN": "900-12-3456", "Relationship": "EE"},
        {"Emp ID": "E1", "First Name": "Kai", "Last Name": "Lee", "SSN": None, "Relationship": "CH"},
    ])
    emps, issues = build_employees(df, map_columns(df))
    assert len(emps) == 1 and len(emps[0].dependents) == 1 and not issues


def test_missing_required_column_is_reported():
    df = pd.DataFrame([{"Favorite Color": "blue"}])
    emps, issues = build_employees(df, map_columns(df))
    assert emps == [] and issues[0].code == "MISSING_COLUMN"


def test_qa_hands_off_out_of_scope():
    pages = [(1, "Deductible\nThe deductible is $1,500 per individual.\nCopays\nOffice visit: $25 copay.")]
    qa = PlanQA(pages)
    assert qa.ask("What is my deductible?")["handoff"] is False
    assert qa.ask("What's the 401k match?")["handoff"] is True

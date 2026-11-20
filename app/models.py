"""Core data models. Everything downstream (rules, outputs, API, UI) speaks these types."""
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Relationship(str, Enum):
    EMPLOYEE = "employee"
    SPOUSE = "spouse"
    CHILD = "child"


class Severity(str, Enum):
    HIGH = "high"      # blocks onboarding / compliance risk
    MEDIUM = "medium"  # needs client confirmation
    LOW = "low"        # informational


class Dependent(BaseModel):
    first_name: str
    last_name: str
    relationship: Relationship
    dob: date | None = None
    ssn: str | None = None


class Employee(BaseModel):
    employee_id: str
    first_name: str
    last_name: str
    ssn: str | None = None
    dob: date | None = None
    hire_date: date | None = None
    hours_per_week: float | None = None
    annual_salary: float | None = None
    email: str | None = None
    dependents: list[Dependent] = Field(default_factory=list)
    source_row: int | None = None  # row in the original file, for traceability

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class PlanRules(BaseModel):
    plan_name: str = "Unknown plan"
    effective_date: date | None = None
    waiting_period_days: int = 0
    min_hours_per_week: float = 30.0
    dependent_age_limit: int = 26
    deductible_individual: float | None = None
    deductible_family: float | None = None
    oop_max_individual: float | None = None
    extraction_method: str = "default"


class Issue(BaseModel):
    code: str                 # machine-readable, e.g. MISSING_SSN (used by evals)
    severity: Severity
    category: str             # human grouping: "Missing data", "Eligibility", ...
    message: str
    employee_id: str | None = None
    field: str | None = None
    client_action: bool = True  # should this go in the email to the client?


class ColumnMapping(BaseModel):
    mapping: dict[str, str]          # original header -> canonical field
    unmapped: list[str] = Field(default_factory=list)
    method: str = "heuristic"        # "llm" or "heuristic"

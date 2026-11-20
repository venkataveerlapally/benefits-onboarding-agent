"""REST API so other systems (n8n, Zapier, a Slack bot, an MCP server) can call the agent.

    uvicorn app.api:app --reload
    open http://localhost:8000/docs
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import llm
from .outputs import issues_dataframe
from .pipeline import OnboardingResult, run_onboarding

app = FastAPI(title="Benefits Onboarding Agent", version="0.1.0")
JOBS: dict[str, OnboardingResult] = {}  # in-memory; swap for Postgres/Redis in production


def _save(upload: UploadFile | None, folder: Path) -> Path | None:
    if upload is None or not upload.filename:
        return None
    path = folder / Path(upload.filename).name
    with path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def _summary(job_id: str, r: OnboardingResult) -> dict:
    return {
        "job_id": job_id,
        "client": r.client_name,
        "plan": r.plan.model_dump(mode="json"),
        "column_mapping": r.mapping.model_dump(),
        "stats": r.stats,
        "checklist": r.checklist,
        "issues": [i.model_dump(mode="json") for i in r.issues],
        "email_draft": r.email,
        "email_method": r.email_method,
        "timings_sec": r.timings,
    }


@app.get("/health")
def health():
    return {"ok": True, "llm_enabled": llm.available(), "jobs": len(JOBS)}


@app.post("/onboard")
def onboard(client_name: str = Form(...), census: UploadFile = File(...),
            plan_pdf: UploadFile | None = File(None), payroll: UploadFile | None = File(None)):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        try:
            r = run_onboarding(client_name, _save(census, folder), _save(plan_pdf, folder), _save(payroll, folder))
        except ValueError as e:
            raise HTTPException(400, str(e))
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = r
    return _summary(job_id, r)


def _job(job_id: str) -> OnboardingResult:
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job id")
    return JOBS[job_id]


@app.get("/onboard/{job_id}")
def get_job(job_id: str):
    return _summary(job_id, _job(job_id))


@app.get("/onboard/{job_id}/issues.csv", response_class=PlainTextResponse)
def issues_csv(job_id: str):
    return issues_dataframe(_job(job_id).issues).to_csv(index=False)


class Question(BaseModel):
    question: str


@app.post("/onboard/{job_id}/ask")
def ask(job_id: str, q: Question):
    r = _job(job_id)
    if r.qa is None:
        raise HTTPException(400, "No plan document was uploaded for this client")
    return r.qa.ask(q.question)

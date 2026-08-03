import os
from datetime import datetime
from decimal import Decimal, InvalidOperation

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Project Submission API")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "smart_failure_detection"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "0000"),
}


def get_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {e}")


class ProjectSubmission(BaseModel):
    project_name: str = Field(..., min_length=2, max_length=200)
    industry_sector: str = Field(..., min_length=2, max_length=150)
    business_model: str = Field(..., min_length=2, max_length=150)
    target_market: str = Field(..., min_length=2, max_length=200)
    budget: Decimal = Field(..., gt=0)
    description: str = Field(..., min_length=10)

    @field_validator("budget", mode="before")
    @classmethod
    def parse_budget(cls, v):
        try:
            return Decimal(str(v))
        except (InvalidOperation, TypeError):
            raise ValueError("budget must be a valid number")

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("index.html")
@app.post("/api/projects")
def submit_project(payload: ProjectSubmission):
    """Insert a new project submission. No auth/user layer yet (Week 1-2 scope only)."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO projects
                    (project_name, industry_sector, business_model, target_market, budget, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING project_id, created_at
                """,
                (
                    payload.project_name,
                    payload.industry_sector,
                    payload.business_model,
                    payload.target_market,
                    payload.budget,
                    payload.description,
                ),
            )
            row = cur.fetchone()
            conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()

    return {
        "status": "success",
        "project_id": row["project_id"],
        "created_at": row["created_at"].isoformat(),
    }


@app.get("/api/projects")
def list_projects():
    """Basic listing endpoint — useful for verifying storage during dev/demo."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM projects ORDER BY created_at DESC")
            rows = cur.fetchall()
    finally:
        conn.close()
    return {"count": len(rows), "projects": rows}


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}



import random


def generate_mock_analysis(project: dict) -> dict:
    seed = project["project_id"]
    rng = random.Random(seed)

    failure_risk = rng.randint(35, 85)
    success_prob = max(5, min(95, 100 - failure_risk + rng.randint(-8, 8)))

    categories = ["Market Risk", "Financial Risk", "Technical Risk", "Operational Risk", "Competition Risk"]
    risk_breakdown = [{"name": c, "pct": rng.randint(25, 90)} for c in categories]

    health_score = max(10, min(95, 100 - failure_risk + rng.randint(-5, 10)))
    health_status = "Excellent" if health_score >= 80 else "Good" if health_score >= 55 else "Needs Attention"

    budget = float(project["budget"])

    return {
        "project": {
            "name": project["project_name"],
            "industry": project["industry_sector"],
            "analysisDate": datetime.utcnow().strftime("%Y-%m-%d"),
        },
        "model": {
            "name": "MOCK — Risk Scoring Model (Milestone 2 pending)",
            "confidencePct": rng.randint(70, 95),
            "confidenceLabel": "High Confidence" if failure_risk < 60 else "Moderate Confidence",
        },
        "overview": {
            "failureRiskPct": failure_risk,
            "riskLevel": "High" if failure_risk >= 60 else "Moderate" if failure_risk >= 35 else "Low",
            "successProbabilityPct": success_prob,
        },
        "riskBreakdown": risk_breakdown,
        "insights": [
            f"Budget of ₹{budget:,.0f} relative to the stated target market may be under- or over-scoped — flagged for review.",
            f"Industry sector '{project['industry_sector']}' shows variable competitive intensity in comparable ventures.",
            "Business model and target market alignment should be validated with early customer feedback.",
            "Placeholder insight — replace once the Market & Competitor Intelligence Engine (Milestone 1-2) is wired in.",
        ],
        "aiSummary": (
            f"This is placeholder analysis for '{project['project_name']}'. Once the Risk Scoring Model and "
            f"LLM strategic reasoning layer are implemented, this section will contain a real AI-generated "
            f"assessment of failure risk, market position, and recommended next steps."
        ),
        "mitigations": [
            {"icon": "trend", "title": "Validate market fit early", "desc": "Run a small pilot before committing full budget."},
            {"icon": "shield", "title": "Extend financial runway", "desc": "Build in buffer beyond initial projections."},
            {"icon": "target", "title": "Clarify competitive differentiation", "desc": "Identify a specific underserved niche."},
        ],
        "whatif": [
            {"label": "Increase Budget", "from": f"₹{budget:,.0f}", "to": f"₹{budget * 1.5:,.0f}",
             "metric": "Failure Risk", "fromPct": failure_risk, "toPct": max(10, failure_risk - 13), "delta": -13},
        ],
        "swot": {
            "strengths": ["Clear initial value proposition", "Founder domain familiarity (placeholder)"],
            "weaknesses": ["Limited working capital (placeholder)", "Undefined operational plan (placeholder)"],
            "opportunities": ["Market demand trend (placeholder)", "Underserved segment (placeholder)"],
            "threats": ["Established competitors (placeholder)", "Thin margins (placeholder)"],
        },
        "timeline": [
            {"phase": "Immediate", "text": "Validate core assumptions with a small pilot before further spend."},
            {"phase": "30 Days", "text": "Reassess budget allocation based on early data."},
            {"phase": "60 Days", "text": "Finalize operational and go-to-market plan."},
            {"phase": "90 Days", "text": "Evaluate expansion only if unit economics are positive."},
        ],
        "healthScore": {"score": health_score, "max": 100, "status": health_status},
        "similarStartups": [
            {"name": "Placeholder Startup A", "category": project["industry_sector"]},
            {"name": "Placeholder Startup B", "category": project["industry_sector"]},
            {"name": "Placeholder Startup C", "category": project["industry_sector"]},
        ],
    }


@app.get("/api/analysis/{project_id}")
def get_analysis(project_id: int):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM projects WHERE project_id = %s", (project_id,))
            project = cur.fetchone()
    finally:
        conn.close()

    if project is None:
        raise HTTPException(status_code=404, detail=f"No project found with id {project_id}")

    return generate_mock_analysis(project)

@app.get("/analysis-results.html", include_in_schema=False)
def analysis_results():
    return FileResponse("analysis-results.html")
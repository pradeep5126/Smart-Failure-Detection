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


def build_competitor_assessment(project: dict) -> list:
    """
    Milestone 2 competition assessment.

    This is a rule-based assessment using the submitted industry, target market,
    business model, and description. It does not claim live market-research data.
    The structure is designed for the current Competition Assessment UI.
    """
    industry = (project.get("industry_sector") or "Startup").strip()
    target_market = (project.get("target_market") or "").strip()
    business_model = (project.get("business_model") or "").strip()
    description = (project.get("description") or "").lower()

    # Keep the names neutral until a live market-research layer is connected.
    competitors = [
        {
            "name": f"{industry} Established Players",
            "marketPosition": "Established market presence",
            "competitiveRisk": "High",
        },
        {
            "name": f"{industry} Growth Startups",
            "marketPosition": "Fast-growing challenger segment",
            "competitiveRisk": "Moderate",
        },
        {
            "name": f"{industry} Niche Providers",
            "marketPosition": "Focused / specialized competitors",
            "competitiveRisk": "Moderate",
        },
    ]

    # Make the assessment respond to the submitted project rather than being
    # identical for every project.
    if len(target_market.split()) >= 6:
        competitors[1]["marketPosition"] = "Challengers serving a similar target market"

    if any(word in description for word in
           ["unique", "differentiation", "niche", "underserved", "advantage"]):
        competitors[2]["competitiveRisk"] = "Low"
        competitors[2]["marketPosition"] = "Specialized competitors with narrower focus"

    if business_model.lower() in {"marketplace", "transactional"}:
        competitors[1]["competitiveRisk"] = "High"

    return competitors


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
            {"name": "Comparable Venture A", "category": project["industry_sector"]},
            {"name": "Comparable Venture B", "category": project["industry_sector"]},
            {"name": "Comparable Venture C", "category": project["industry_sector"]},
        ],
        "competitors": build_competitor_assessment(project),
    }


# =========================
# Milestone 2: Risk Assessment, SWOT & Feasibility
# =========================

def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> int:
    return int(round(max(low, min(high, value))))


def _word_count(text: str) -> int:
    return len((text or "").split())


def _matches_any(text: str, keywords) -> bool:
    text = (text or "").lower()
    return any(keyword.lower() in text for keyword in keywords)


def _financial_risk(project: dict) -> int:
    budget = float(project["budget"])
    description = (project.get("description") or "").lower()
    business_model = (project.get("business_model") or "").lower()

    risk = 55.0

    if budget < 100000:
        risk += 15
    elif budget < 500000:
        risk += 5
    elif budget >= 2000000:
        risk -= 8

    if _matches_any(description, ["revenue", "profit", "subscription", "recurring", "margin", "cash flow"]):
        risk -= 6
    if _matches_any(description, ["high cost", "capital intensive", "burn", "debt"]):
        risk += 10
    if _matches_any(business_model, ["subscription", "saas", "service"]):
        risk -= 5

    return _clamp(risk)


def _market_risk(project: dict) -> int:
    market = project.get("target_market") or ""
    description = project.get("description") or ""
    risk = 55.0

    if _word_count(market) < 3:
        risk += 8
    elif _word_count(market) >= 8:
        risk -= 5

    if _matches_any(description, ["customer", "demand", "market research", "validation", "pilot", "traction"]):
        risk -= 8
    if _matches_any(description, ["global", "everyone", "all customers", "mass market"]):
        risk += 6

    return _clamp(risk)


def _competition_risk(project: dict) -> int:
    description = project.get("description") or ""
    industry = project.get("industry_sector") or ""
    risk = 55.0

    if _matches_any(description, ["competitor", "differentiation", "unique", "niche", "advantage", "moat"]):
        risk -= 10
    if _matches_any(description, ["crowded", "saturated", "many competitors", "competitive"]):
        risk += 10
    if _matches_any(industry, ["fintech", "e-commerce", "social media", "food delivery", "marketplace"]):
        risk += 5

    return _clamp(risk)


def _technical_risk(project: dict) -> int:
    description = project.get("description") or ""
    risk = 50.0

    if _matches_any(description, ["ai", "machine learning", "hardware", "iot", "blockchain", "deep learning", "real-time"]):
        risk += 10
    if _matches_any(description, ["prototype", "mvp", "tested", "working prototype", "technology validated"]):
        risk -= 8
    if _word_count(description) < 30:
        risk += 8

    return _clamp(risk)


def _operational_risk(project: dict) -> int:
    description = project.get("description") or ""
    business_model = project.get("business_model") or ""
    risk = 52.0

    if _matches_any(description, ["team", "operations", "supply chain", "logistics", "process", "partnership"]):
        risk -= 7
    if _matches_any(business_model, ["marketplace", "delivery", "manufacturing", "logistics"]):
        risk += 8
    if _word_count(description) < 30:
        risk += 6

    return _clamp(risk)


def _confidence_score(project: dict, risks: dict) -> int:
    description_quality = min(20, _word_count(project.get("description") or "") // 4)
    market_quality = min(10, _word_count(project.get("target_market") or ""))
    risk_spread = max(risks.values()) - min(risks.values())
    consistency = max(0, 10 - int(risk_spread / 10))

    return _clamp(55 + description_quality + market_quality + consistency, 0, 95)


def compute_milestone2_analysis(project: dict) -> dict:
    risks = {
        "Financial Risk": _financial_risk(project),
        "Market Risk": _market_risk(project),
        "Competition Risk": _competition_risk(project),
        "Technical Risk": _technical_risk(project),
        "Operational Risk": _operational_risk(project),
    }

    weights = {
        "Financial Risk": 0.25,
        "Market Risk": 0.25,
        "Competition Risk": 0.15,
        "Technical Risk": 0.15,
        "Operational Risk": 0.20,
    }

    overall_risk = _clamp(sum(risks[name] * weights[name] for name in risks))

    if overall_risk >= 81:
        risk_level = "Critical Risk"
    elif overall_risk >= 61:
        risk_level = "High Risk"
    elif overall_risk >= 31:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # Lower risk means higher feasibility.
    feasibility = {
        "Financial Feasibility": 100 - risks["Financial Risk"],
        "Market Feasibility": 100 - risks["Market Risk"],
        "Competitive Feasibility": 100 - risks["Competition Risk"],
        "Technical Feasibility": 100 - risks["Technical Risk"],
        "Operational Feasibility": 100 - risks["Operational Risk"],
    }
    overall_feasibility = _clamp(sum(feasibility.values()) / len(feasibility))

    if overall_feasibility >= 80:
        feasibility_level = "Highly Feasible"
    elif overall_feasibility >= 60:
        feasibility_level = "Feasible"
    elif overall_feasibility >= 40:
        feasibility_level = "Needs Attention"
    else:
        feasibility_level = "Low Feasibility"

    description = project.get("description") or ""
    business_model = project.get("business_model") or ""
    market = project.get("target_market") or ""

    strengths = []
    weaknesses = []
    opportunities = []
    threats = []

    if risks["Market Risk"] <= 45:
        strengths.append("The target market is described with useful specificity.")
    else:
        weaknesses.append("The target market needs clearer definition and validation.")

    if risks["Financial Risk"] <= 45:
        strengths.append("The available budget profile supports the stated direction.")
    else:
        weaknesses.append("Financial assumptions or funding requirements need closer review.")

    if risks["Competition Risk"] <= 45:
        strengths.append("The description indicates attention to differentiation or market positioning.")
    else:
        threats.append("Competitive pressure may make customer acquisition or differentiation difficult.")

    if risks["Technical Risk"] <= 45:
        strengths.append("The proposed solution appears technically manageable from the information provided.")
    else:
        weaknesses.append("Technical complexity may increase development time, cost, or execution risk.")

    if _matches_any(description, ["customer", "demand", "validation", "pilot", "traction"]):
        opportunities.append("Early customer validation can strengthen the opportunity and reduce uncertainty.")
    else:
        opportunities.append("Customer validation can reveal demand and improve the opportunity assessment.")

    if _matches_any(description, ["niche", "underserved", "unique", "differentiation", "advantage"]):
        opportunities.append("A focused differentiation strategy may create room in the target market.")
    else:
        opportunities.append("A focused niche or clearer differentiation could strengthen market positioning.")

    if risks["Operational Risk"] >= 61:
        threats.append("Operational complexity could make scaling difficult.")
    else:
        threats.append("Execution quality and operational discipline will remain important as the project scales.")

    if not strengths:
        strengths.append("The project has a defined concept that can be evaluated further.")
    if not weaknesses:
        weaknesses.append("More evidence is needed before treating the current assumptions as validated.")
    if not threats:
        threats.append("Changes in customer demand or competitive conditions could affect the project.")

    confidence = _confidence_score(project, risks)

    positive_factors = [
        name for name, score in feasibility.items() if score >= 60
    ]
    attention_areas = [
        name for name, score in feasibility.items() if score < 60
    ]

    if overall_feasibility >= 80:
        feasibility_summary = "The project appears highly feasible based on the information provided."
    elif overall_feasibility >= 60:
        feasibility_summary = "The project appears feasible, with some areas requiring continued validation."
    elif overall_feasibility >= 40:
        feasibility_summary = "The project has potential but several feasibility areas require attention."
    else:
        feasibility_summary = "The project currently shows low feasibility and requires significant validation before proceeding."

    return {
        "riskScoring": {
            "overallScore": overall_risk,
            "riskLevel": risk_level,
            "riskBreakdown": [
                {"name": name, "score": score}
                for name, score in risks.items()
            ],
            "confidenceScore": confidence,
        },
        "swot": {
            "strengths": strengths[:4],
            "weaknesses": weaknesses[:4],
            "opportunities": opportunities[:4],
            "threats": threats[:4],
        },
        "feasibility": {
            "overallScore": overall_feasibility,
            "level": feasibility_level,
            "breakdown": [
                {"name": name, "score": score}
                for name, score in feasibility.items()
            ],
            "positiveFactors": positive_factors,
            "attentionAreas": attention_areas,
            "summary": feasibility_summary,
        },
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

    result = generate_mock_analysis(project)
    result["milestone2"] = compute_milestone2_analysis(project)
    return result

@app.get("/analysis-results.html", include_in_schema=False)
def analysis_results():
    return FileResponse("analysis-results.html")
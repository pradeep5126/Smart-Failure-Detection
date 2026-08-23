import os
import re
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from strategy_engine import get_or_generate_strategy

# In-memory per-project rate limiter for strategy regeneration
REGENERATION_RATE_LIMIT_WINDOW_SECONDS = 600  # 10 minutes rolling window
REGENERATION_RATE_LIMIT_MAX_REQUESTS = 3
_regeneration_timestamps: dict[int, list[float]] = {}


def clear_regeneration_rate_limits():
    """Clear all in-memory regeneration rate limit timestamps."""
    _regeneration_timestamps.clear()


def check_and_record_regeneration_rate_limit(project_id: int) -> None:
    """
    Enforces a maximum of 3 regeneration requests per project within a rolling 10-minute window.
    Evicts timestamps older than 10 minutes before evaluating the limit.
    Raises HTTPException(429) if the limit is exceeded.
    """
    now = time.time()
    cutoff = now - REGENERATION_RATE_LIMIT_WINDOW_SECONDS
    timestamps = [t for t in _regeneration_timestamps.get(project_id, []) if t > cutoff]
    if len(timestamps) >= REGENERATION_RATE_LIMIT_MAX_REQUESTS:
        _regeneration_timestamps[project_id] = timestamps
        raise HTTPException(
            status_code=429,
            detail="Regeneration limit reached. Please try again later.",
        )
    timestamps.append(now)
    _regeneration_timestamps[project_id] = timestamps

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
    return FileResponse("overview.html")

@app.get("/overview.html", include_in_schema=False)
def overview():
    return FileResponse("overview.html")

@app.get("/index.html", include_in_schema=False)
def project_input():
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

# =========================
# Milestone 2: Risk Assessment, SWOT & Feasibility
#
# Each _xxx_risk() function returns (score, reasons) where reasons is a list
# of {"text": ..., "positive": bool} explaining WHAT specifically drove that
# score. SWOT is then built directly from those reasons — not from generic
# "risk is high/low" templates — so two projects with the same overall score
# but different underlying signals get genuinely different SWOT text.
# =========================

import math


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> int:
    return int(round(max(low, min(high, value))))


def _word_count(text: str) -> int:
    return len((text or "").split())


def _keyword_present(text: str, keyword: str) -> bool:
    """Case-insensitive whole-word/phrase matching; avoids substring false positives."""
    text = (text or "").lower()
    keyword = (keyword or "").strip().lower()
    if not text or not keyword:
        return False

    pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in keyword.split()) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _count_matches(text: str, keywords) -> int:
    """Count distinct keyword signals, not repeated occurrences."""
    return sum(1 for keyword in keywords if _keyword_present(text, keyword))


def _matches_any(text: str, keywords) -> bool:
    return _count_matches(text, keywords) > 0


# ---- Industry tier lookups (shared across market/competition scoring) ----
_HIGH_COMPETITION_INDUSTRIES = [
    "food delivery", "grocery delivery", "grocery", "ride sharing", "ride-hailing",
    "hyperlocal delivery", "e-commerce", "ecommerce", "d2c fashion", "fashion",
    "social media", "quick commerce",
]
_MODERATE_COMPETITION_INDUSTRIES = [
    "fintech", "healthtech", "edtech", "real estate", "travel", "logistics",
    "fitness", "payments", "insurtech", "proptech",
]
_LOW_COMPETITION_INDUSTRIES = [
    "b2b saas", "enterprise software", "developer tools", "developer tool",
    "vertical saas", "niche saas", "api platform", "industrial", "deep tech",
    "biotech", "climate tech", "agritech",
]


def _financial_risk(project: dict):
    budget = float(project["budget"])
    description = (project.get("description") or "").lower()
    business_model = (project.get("business_model") or "").lower()
    reasons = []

    # Budget is one signal, not the definition of financial health.
    budget_clamped = max(20_000, min(budget, 50_000_000))
    log_budget = math.log10(budget_clamped)
    log_min, log_max = math.log10(20_000), math.log10(50_000_000)
    norm = (log_budget - log_min) / (log_max - log_min)
    base = 68 - norm * 30

    if budget < 150_000:
        reasons.append({
            "text": f"Budget of ₹{budget:,.0f} is relatively limited and may constrain execution runway.",
            "positive": False,
        })
    elif budget >= 3_000_000:
        reasons.append({
            "text": f"Budget of ₹{budget:,.0f} provides a stronger execution runway, assuming spending is controlled.",
            "positive": True,
        })

    revenue_kw = [
        "revenue", "profit", "subscription", "recurring revenue", "recurring",
        "margin", "cash flow", "break-even", "profitable", "paying customers",
        "customers", "sales", "traction", "monetization",
    ]
    funding_kw = ["funded", "funding", "investment", "grant", "seed round", "angel", "venture capital"]
    burn_kw = [
        "high cost", "high costs", "capital intensive", "capital-intensive",
        "burn", "debt", "loss-making", "unprofitable", "cash-strapped",
        "high operating cost", "high operating costs",
    ]

    rev_hits = _count_matches(description, revenue_kw)
    fund_hits = _count_matches(description, funding_kw)
    burn_hits = _count_matches(description, burn_kw)

    base -= min(rev_hits, 5) * 4.0
    base -= min(fund_hits, 3) * 3.0
    base += min(burn_hits, 4) * 7.0

    if rev_hits >= 2:
        reasons.append({
            "text": "The description provides evidence of revenue, customers, monetization, or margins, reducing near-term financial uncertainty.",
            "positive": True,
        })
    elif rev_hits == 0:
        reasons.append({
            "text": "No clear revenue, customer, or monetization evidence is stated, increasing financial uncertainty.",
            "positive": False,
        })

    if fund_hits >= 1:
        reasons.append({
            "text": "Funding or investment is mentioned, which can improve available runway.",
            "positive": True,
        })

    if burn_hits >= 1:
        reasons.append({
            "text": "High costs, burn, debt, or capital intensity are mentioned, increasing financial pressure.",
            "positive": False,
        })

    model_modifier = {
        "saas": -5,
        "subscription": -4,
        "service": -2,
        "marketplace": 5,
        "d2c": 4,
        "b2c": 2,
        "hardware": 7,
        "manufacturing": 8,
    }
    original_model = project.get("business_model") or business_model
    for key, val in model_modifier.items():
        if _keyword_present(business_model, key):
            base += val
            if val < 0:
                reasons.append({
                    "text": f"{original_model} model can be relatively capital-efficient, easing some financial pressure.",
                    "positive": True,
                })
            else:
                reasons.append({
                    "text": f"{original_model} model can require higher upfront or operating capital.",
                    "positive": False,
                })
            break

    # Capital-intensive models without stated revenue or funding deserve
    # additional financial caution.
    if (
        any(_keyword_present(business_model, k) for k in ["hardware", "manufacturing", "marketplace", "d2c"])
        and rev_hits == 0
        and fund_hits == 0
    ):
        base += 5
        reasons.append({
            "text": "The business model may require meaningful operating capital without stated revenue or funding support.",
            "positive": False,
        })

    return _clamp(base), reasons

def _market_risk(project: dict):
    market = project.get("target_market") or ""
    description = (project.get("description") or "").lower()
    industry = (project.get("industry_sector") or "").lower()
    reasons = []

    # Continuous curve on target-market specificity (word count) instead of 2 buckets.
    wc = _word_count(market)
    wc_clamped = max(0, min(wc, 14))
    base = 76 - (wc_clamped / 14) * 48  # ~76 (vague) down to ~28 (specific)

    if wc <= 2:
        reasons.append({"text": f"Target market ('{market}') is very broadly defined and needs segmentation.", "positive": False})
    elif wc >= 8:
        reasons.append({"text": f"Target market is specifically defined ('{market}'), supporting focused go-to-market.", "positive": True})

    validation_kw = ["customer interviews", "market research", "survey", "validated", "pilot", "traction", "waitlist", "beta users", "demand validated"]
    mass_kw = ["everyone", "global", "all customers", "mass market", "anyone", "general public", "broad audience"]

    val_hits = _count_matches(description, validation_kw)
    mass_hits = _count_matches(description, mass_kw)

    base -= min(val_hits, 3) * 5
    base += min(mass_hits, 2) * 7

    if val_hits >= 1:
        reasons.append({"text": "Description references market validation activity (research, pilot, or traction).", "positive": True})
    if mass_hits >= 1:
        reasons.append({"text": "Target market is described as broad ('everyone' / 'mass market'), which is hard to serve well early on.", "positive": False})

    if any(k in industry for k in _HIGH_COMPETITION_INDUSTRIES):
        base += 9
        reasons.append({"text": f"'{project.get('industry_sector')}' is a structurally difficult, high-saturation market.", "positive": False})
    elif any(k in industry for k in _LOW_COMPETITION_INDUSTRIES):
        base -= 7
        reasons.append({"text": f"'{project.get('industry_sector')}' is a comparatively under-saturated market segment.", "positive": True})

    return _clamp(base), reasons


def _competition_risk(project: dict):
    description = (project.get("description") or "").lower()
    industry = (project.get("industry_sector") or "").lower()
    business_model = (project.get("business_model") or "").lower()
    reasons = []
    base = 52.0

    if any(k in industry for k in _HIGH_COMPETITION_INDUSTRIES):
        base += 15
        reasons.append({"text": f"'{project.get('industry_sector')}' has intense competitive activity from established players.", "positive": False})
    elif any(k in industry for k in _MODERATE_COMPETITION_INDUSTRIES):
        base += 4
    elif any(k in industry for k in _LOW_COMPETITION_INDUSTRIES):
        base -= 10
        reasons.append({"text": f"'{project.get('industry_sector')}' has relatively fewer direct competitors.", "positive": True})

    diff_kw = ["unique", "differentiation", "niche", "underserved", "advantage", "proprietary", "patent", "moat", "first mover", "exclusive"]
    crowd_kw = ["crowded", "saturated", "many competitors", "competitive", "commoditized", "red ocean"]

    diff_hits = _count_matches(description, diff_kw)
    crowd_hits = _count_matches(description, crowd_kw)

    base -= min(diff_hits, 4) * 4.5
    base += min(crowd_hits, 3) * 6

    if diff_hits >= 2:
        reasons.append({"text": "Description articulates a specific competitive advantage or differentiation angle.", "positive": True})
    elif diff_hits == 0 and crowd_hits == 0:
        reasons.append({"text": "Description doesn't yet name a specific competitive advantage over existing players.", "positive": False})
    if crowd_hits >= 1:
        reasons.append({"text": "Description itself acknowledges a crowded or highly competitive space.", "positive": False})

    model_modifier = {"marketplace": 6, "b2c": 3, "d2c": 4, "saas": -3, "b2b": -3, "enterprise": -5}
    for key, val in model_modifier.items():
        if key in business_model:
            base += val

    return _clamp(base), reasons


def _technical_risk(project: dict):
    description = (project.get("description") or "").lower()
    reasons = []
    base = 42.0

    high_complexity_kw = ["hardware", "iot", "blockchain", "biotech", "genomics", "robotics", "autonomous", "computer vision", "deep learning", "neural network", "chip", "semiconductor"]
    moderate_complexity_kw = ["ai", "machine learning", "real-time", "data pipeline", "recommendation engine", "predictive model", "distributed system", "large scale"]
    low_complexity_kw = ["simple app", "landing page", "content platform", "booking system", "standard web app", "basic website"]

    high_hits = _count_matches(description, high_complexity_kw)
    mod_hits = _count_matches(description, moderate_complexity_kw)
    low_hits = _count_matches(description, low_complexity_kw)

    base += min(high_hits, 3) * 11
    base += min(mod_hits, 3) * 6
    base -= min(low_hits, 2) * 9

    if high_hits >= 1:
        reasons.append({"text": "Description involves hardware, IoT, blockchain, or similarly deep technical complexity.", "positive": False})
    elif mod_hits >= 1:
        reasons.append({"text": "Description involves AI/ML or real-time systems, adding moderate technical complexity.", "positive": False})
    elif low_hits >= 1:
        reasons.append({"text": "Proposed solution is technically straightforward to build.", "positive": True})

    maturity_kw = ["prototype", "mvp", "tested", "working prototype", "technology validated", "beta", "in production", "live product", "deployed"]
    maturity_hits = _count_matches(description, maturity_kw)
    base -= min(maturity_hits, 3) * 7

    if maturity_hits >= 1:
        reasons.append({"text": "An existing prototype, MVP, or deployed version reduces technical uncertainty.", "positive": True})

    wc = _word_count(description)
    if wc < 15:
        base += 12
        reasons.append({"text": "Description is too short to assess the technical approach with confidence.", "positive": False})
    elif wc >= 80:
        base -= 5

    return _clamp(base), reasons


def _operational_risk(project: dict):
    description = (project.get("description") or "").lower()
    business_model = (project.get("business_model") or "").lower()
    reasons = []
    base = 48.0

    complex_model_kw = ["marketplace", "logistics", "manufacturing", "hardware", "delivery", "multi-vendor", "two-sided", "b2b2c"]
    simple_model_kw = ["saas", "subscription", "digital product", "software only", "content"]

    if any(k in business_model for k in complex_model_kw):
        base += 13
        reasons.append({"text": f"{project.get('business_model')} model involves multiple moving parts (logistics, vendors, or delivery), raising execution risk.", "positive": False})
    elif any(k in business_model for k in simple_model_kw):
        base -= 6
        reasons.append({"text": f"{project.get('business_model')} model has a comparatively simple, single-sided operating structure.", "positive": True})

    ops_positive_kw = ["team", "operations", "supply chain", "process", "partnership", "hired", "operational plan", "vendor agreements", "fulfillment", "logistics partner"]
    ops_negative_kw = ["solo founder", "no team", "outsourced", "undefined process", "early stage team"]

    pos_hits = _count_matches(description, ops_positive_kw)
    neg_hits = _count_matches(description, ops_negative_kw)

    base -= min(pos_hits, 4) * 4.5
    base += min(neg_hits, 3) * 7

    if pos_hits >= 2:
        reasons.append({"text": "Description references a defined team, process, or operational plan.", "positive": True})
    if neg_hits >= 1:
        reasons.append({"text": "Description suggests limited team or process maturity at this stage.", "positive": False})

    wc = _word_count(description)
    if wc < 20:
        base += 8
    elif wc >= 60:
        base -= 4

    return _clamp(base), reasons


def _confidence_score(project: dict, risk_scores: dict) -> int:
    """Estimate confidence from information completeness, not score similarity."""
    description = project.get("description") or ""
    target_market = project.get("target_market") or ""
    business_model = project.get("business_model") or ""
    industry = project.get("industry_sector") or ""

    score = 45
    score += min(20, _word_count(description) // 4)
    score += min(10, _word_count(target_market))

    if len(business_model.split()) >= 2:
        score += 5
    if len(industry.split()) >= 2:
        score += 3

    evidence_keywords = [
        "customer", "customers", "revenue", "funding", "prototype", "mvp",
        "pilot", "traction", "validated", "beta", "deployed", "team",
        "partnership", "operational plan",
    ]
    evidence_hits = _count_matches(description, evidence_keywords)
    score += min(12, evidence_hits * 2)

    return _clamp(score, 0, 95)

def compute_milestone2_analysis(project: dict) -> dict:
    financial_score, financial_reasons = _financial_risk(project)
    market_score, market_reasons = _market_risk(project)
    competition_score, competition_reasons = _competition_risk(project)
    technical_score, technical_reasons = _technical_risk(project)
    operational_score, operational_reasons = _operational_risk(project)

    risks = {
        "Financial Risk": financial_score,
        "Market Risk": market_score,
        "Competition Risk": competition_score,
        "Technical Risk": technical_score,
        "Operational Risk": operational_score,
    }
    all_reasons = {
        "Financial Risk": financial_reasons,
        "Market Risk": market_reasons,
        "Competition Risk": competition_reasons,
        "Technical Risk": technical_reasons,
        "Operational Risk": operational_reasons,
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

    # ---- SWOT built directly from the specific reasons each category produced ----
    description = (project.get("description") or "").lower()

    strengths, weaknesses = [], []
    # Sort by how far each category's score is from a neutral 50, so the most
    # decisive signals surface first rather than an arbitrary category order.
    category_order = sorted(risks.keys(), key=lambda name: abs(risks[name] - 50), reverse=True)
    for name in category_order:
        for reason in all_reasons[name]:
            target = strengths if reason["positive"] else weaknesses
            if reason["text"] not in target:
                target.append(reason["text"])

    opportunities = []
    if _matches_any(description, ["customer interviews", "market research", "survey", "validated", "pilot", "traction", "waitlist", "beta users", "demand validated"]):
        opportunities.append("Existing validation signals can be expanded into stronger customer traction and demand evidence.")
    else:
        opportunities.append("Customer interviews, a pilot, or a waitlist would create evidence to validate demand before scaling.")

    if _matches_any(description, ["niche", "underserved", "unique", "differentiation", "advantage", "proprietary", "patent", "moat", "first mover", "exclusive"]):
        opportunities.append("The stated differentiation can be strengthened into a clearer and more defensible market position.")
    else:
        opportunities.append("A clearly defined differentiation angle could reduce competitive exposure and improve positioning.")

    if competition_score <= 40:
        opportunities.append(f"Relatively lower competitive pressure in '{project.get('industry_sector')}' may provide room to establish an early position.")

    if technical_score <= 40 and _matches_any(description, ["prototype", "mvp", "tested", "beta", "deployed", "live product"]):
        opportunities.append("Existing technical maturity creates an opportunity to validate the product with real users and iterate faster.")

    threats = []
    if competition_score >= 60:
        threats.append(f"High competitive intensity in '{project.get('industry_sector')}' may pressure customer acquisition cost and pricing.")
    if financial_score >= 60:
        threats.append("Current budget may not sustain the business through an extended pre-revenue phase.")
    if operational_score >= 60:
        threats.append("Operational complexity of the chosen model raises execution and coordination risk as the project scales.")
    if technical_score >= 60:
        threats.append("Technical complexity could extend development timelines and increase the cost of reaching a working product.")
    if not threats:
        threats.append("Standard market and execution risks apply; no single factor stands out as unusually severe.")

    if not strengths:
        strengths.append("The project has a clearly defined concept that can be evaluated further as it develops.")
    if not weaknesses:
        weaknesses.append("No major weaknesses were flagged from the submission alone — deeper diligence is still recommended.")

    confidence = _confidence_score(project, risks)

    positive_factors = [name for name, score in feasibility.items() if score >= 60]
    attention_areas = [name for name, score in feasibility.items() if score < 60]

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

    milestone2 = compute_milestone2_analysis(project)

    # Milestone 2 is now the authoritative analysis; no random/mock
    # risk values are used by this endpoint.
    confidence = milestone2["riskScoring"]["confidenceScore"]
    attention_areas = milestone2["feasibility"]["attentionAreas"]

    result = {
        "project": {
            "name": project["project_name"],
            "industry": project["industry_sector"],
            "analysisDate": datetime.utcnow().strftime("%Y-%m-%d"),
        },
        "model": {
            "name": "Rule-Based Risk Scoring Engine — Milestone 2",
            "confidencePct": confidence,
            "confidenceLabel": (
                "High Confidence" if confidence >= 75
                else "Moderate Confidence" if confidence >= 55
                else "Limited Confidence"
            ),
        },
        "overview": {
            "failureRiskPct": milestone2["riskScoring"]["overallScore"],
            "riskLevel": milestone2["riskScoring"]["riskLevel"],
            "successProbabilityPct": _clamp(100 - milestone2["riskScoring"]["overallScore"]),
        },
        "riskBreakdown": [
            {"name": item["name"], "score": item["score"], "pct": item["score"]}
            for item in milestone2["riskScoring"]["riskBreakdown"]
        ],
        "insights": milestone2["swot"]["weaknesses"][:3] + milestone2["swot"]["strengths"][:2],
        "aiSummary": milestone2["feasibility"]["summary"],
        "mitigations": [
            {
                "icon": "trend",
                "title": "Address the highest-priority area",
                "desc": area,
            }
            for area in attention_areas[:3]
        ],
        "swot": milestone2["swot"],
        "healthScore": {
            "score": milestone2["feasibility"]["overallScore"],
            "max": 100,
            "status": milestone2["feasibility"]["level"],
        },
        "whatif": [
            {
                "label": "Increase Budget",
                "from": f"₹{float(project['budget']):,.0f}",
                "to": f"₹{float(project['budget']) * 1.5:,.0f}",
                "metric": "Failure Risk",
                "fromPct": milestone2["riskScoring"]["overallScore"],
                "toPct": _clamp(max(0, milestone2["riskScoring"]["overallScore"] - 8)),
                "delta": -8,
            }
        ],
        "similarStartups": [
            {"name": "Comparable Venture A", "category": project["industry_sector"]},
            {"name": "Comparable Venture B", "category": project["industry_sector"]},
            {"name": "Comparable Venture C", "category": project["industry_sector"]},
        ],
        "competitors": build_competitor_assessment(project),
        "milestone2": milestone2,
    }
    return result


@app.get("/api/analysis/{project_id}/strategy")
def get_project_strategy(project_id: int, force_refresh: bool = False):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM projects WHERE project_id = %s", (project_id,))
            project = cur.fetchone()
    finally:
        conn.close()

    if project is None:
        raise HTTPException(status_code=404, detail=f"No project found with id {project_id}")

    milestone2 = compute_milestone2_analysis(project)
    competitors = build_competitor_assessment(project)
    strategy = get_or_generate_strategy(project, milestone2, competitors=competitors, force_refresh=force_refresh)
    return strategy


@app.post("/api/analysis/{project_id}/strategy/regenerate")
def regenerate_project_strategy(project_id: int):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM projects WHERE project_id = %s", (project_id,))
            project = cur.fetchone()
    finally:
        conn.close()

    if project is None:
        raise HTTPException(status_code=404, detail=f"No project found with id {project_id}")

    check_and_record_regeneration_rate_limit(project_id)

    milestone2 = compute_milestone2_analysis(project)
    competitors = build_competitor_assessment(project)
    strategy = get_or_generate_strategy(project, milestone2, competitors=competitors, force_refresh=True)
    return strategy


@app.get("/analysis-results.html", include_in_schema=False)
def analysis_results():
    return FileResponse("analysis-results.html")
"""
strategy_engine.py — Milestone 3 LangGraph-Powered LLM Strategic Advisor & Mitigation Engine

Orchestrates multi-stage strategic reasoning over startup submission data and Milestone 2 risk baselines.
Uses LangGraph for agentic workflow orchestration and pluggable Gemini/OpenAI models for LLM reasoning,
with a robust offline deterministic heuristic engine as an automatic fallback.
"""

import json
import os
from dotenv import load_dotenv
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TypedDict
load_dotenv()

# =====================================================================
# 1. State Schema Definition
# =====================================================================

class StrategyState(TypedDict, total=False):
    project: Dict[str, Any]
    milestone2_analysis: Dict[str, Any]
    provider_info: Dict[str, Any]
    root_causes: List[Dict[str, Any]]
    positioning_analysis: Dict[str, Any]
    mitigation_roadmap: List[Dict[str, Any]]
    executive_synthesis: Dict[str, Any]
    strategic_recommendation_score: int
    confidence_score: int
    generated_at: str
    is_cached: bool


# =====================================================================
# 2. Score Validation Helper
# =====================================================================

def _validate_and_clamp_score(val: Any, fallback: int = 70) -> int:
    """
    Validates and coerces the Strategic Recommendation Score into an integer clamped to [0, 100].
    Prevents malformed LLM outputs (strings, floats, None, out-of-bounds numbers) from breaking the API/UI.
    """
    try:
        if isinstance(val, (int, float, str)):
            clean_str = str(val).strip().split("/")[0].replace("%", "").strip()
            numeric_val = float(clean_str)
            if numeric_val == numeric_val:  # Check for NaN
                return int(round(max(0.0, min(100.0, numeric_val))))
    except (ValueError, TypeError, OverflowError):
        pass
    return max(0, min(100, fallback))


# =====================================================================
# 3. Provider Resolution & Key Management
# =====================================================================

def resolve_llm_provider() -> Tuple[str, str, Optional[str]]:
    """
    Deterministically resolves the LLM provider based on LLM_PROVIDER and available API keys.
    Returns: (provider_name, status_reason, api_key)
    
    Supported provider names: "gemini", "openai", "offline"
    """
    configured_provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if configured_provider == "gemini":
        if gemini_key:
            return "gemini", "Google Gemini configured via GEMINI_API_KEY", gemini_key
        return "offline", "LLM_PROVIDER='gemini' requested but GEMINI_API_KEY is missing; falling back to offline heuristic engine", None

    if configured_provider == "openai":
        if openai_key:
            return "openai", "OpenAI configured via OPENAI_API_KEY", openai_key
        return "offline", "LLM_PROVIDER='openai' requested but OPENAI_API_KEY is missing; falling back to offline heuristic engine", None

    if configured_provider == "offline":
        return "offline", "Explicitly set to offline heuristic reasoning engine", None

    # Default / "auto" mode:
    if gemini_key:
        return "gemini", "Auto-selected Google Gemini (GEMINI_API_KEY detected)", gemini_key
    if openai_key:
        return "openai", "Auto-selected OpenAI (OPENAI_API_KEY detected)", openai_key

    return "offline", "No API key found (GEMINI_API_KEY or OPENAI_API_KEY); using offline heuristic reasoning engine", None


# =====================================================================
# 4. LLM Caller Interface (Gemini / OpenAI / Offline)
# =====================================================================

def _call_gemini(prompt: str, system_instruction: str, api_key: str) -> Optional[str]:
    """Invokes Google Gemini with JSON output expectation."""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return response.text
    except Exception:
        try:
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": f"{system_instruction}\n\nTask:\n{prompt}"}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=12) as res:
                body = json.loads(res.read().decode("utf-8"))
                return body["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return None


def _call_openai(prompt: str, system_instruction: str, api_key: str) -> Optional[str]:
    """Invokes OpenAI with JSON output expectation."""
    try:
        import urllib.request
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=12) as res:
            body = json.loads(res.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except Exception:
        return None


def query_llm_json(prompt: str, system_instruction: str) -> Tuple[Optional[dict], str]:
    """Dispatches to the active provider and returns (parsed_json, actual_provider_used)."""
    provider, _, api_key = resolve_llm_provider()
    
    if provider == "gemini" and api_key:
        raw = _call_gemini(prompt, system_instruction, api_key)
        if raw:
            try:
                clean = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
                return json.loads(clean), "gemini"
            except Exception:
                pass

    if provider == "openai" and api_key:
        raw = _call_openai(prompt, system_instruction, api_key)
        if raw:
            try:
                clean = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
                return json.loads(clean), "openai"
            except Exception:
                pass

    return None, "offline"


# =====================================================================
# 5. Offline Heuristic Engine (High-Fidelity Deterministic Fallback)
# =====================================================================

def _offline_root_causes(project: dict, m2: dict) -> List[dict]:
    causes = []
    risks = {r["name"]: r["score"] for r in m2.get("riskScoring", {}).get("riskBreakdown", [])}
    budget = float(project.get("budget", 0))
    desc = (project.get("description") or "").lower()
    industry = project.get("industry_sector", "Startup")
    model = project.get("business_model", "Standard")

    if risks.get("Financial Risk", 50) >= 55:
        causes.append({
            "category": "Financial Runway & Capital Efficiency",
            "severity": "High" if risks.get("Financial Risk", 0) >= 70 else "Medium",
            "finding": f"Budget of ₹{budget:,.0f} may create runway constraints under a {model} model before sustainable unit margins materialize.",
            "impact": "Risk of running out of capital before reaching product-market fit or completing the first 50 customer validation cycles.",
            "urgency": "Immediate (Days 0-30)"
        })

    if risks.get("Market Risk", 50) >= 50:
        causes.append({
            "category": "Customer Discovery & Demand Validation",
            "severity": "High" if risks.get("Market Risk", 0) >= 65 else "Medium",
            "finding": f"Target market '{project.get('target_market')}' requires tighter persona segmentation and documented willingness-to-pay evidence.",
            "impact": "Customer acquisition costs may escalate rapidly if messaging targets too broad or general an audience.",
            "urgency": "Immediate (Days 0-30)"
        })

    if risks.get("Competition Risk", 50) >= 50:
        causes.append({
            "category": "Competitive Moat & Differentiation",
            "severity": "High" if risks.get("Competition Risk", 0) >= 65 else "Medium",
            "finding": f"Operating in {industry} places the project in proximity to incumbent players with larger capital reserves.",
            "impact": "Vulnerability to fast followers or established players offering feature-matching or subsidized pricing.",
            "urgency": "Medium-Term (Days 30-60)"
        })

    if risks.get("Operational Risk", 50) >= 50 or risks.get("Technical Risk", 50) >= 50:
        causes.append({
            "category": "Execution Velocity & Operational Complexity",
            "severity": "Medium",
            "finding": f"The operational workflow for a {model} venture requires structured coordination between technology, acquisition, and delivery.",
            "impact": "Execution bottlenecks during early iteration cycles could delay customer onboarding.",
            "urgency": "Medium-Term (Days 30-60)"
        })

    if not causes:
        causes.append({
            "category": "Baseline Execution Risk",
            "severity": "Low",
            "finding": "Core quantitative metrics appear balanced, but execution diligence and milestone tracking remain essential.",
            "impact": "Maintain controlled burn and validate weekly retention metrics.",
            "urgency": "Ongoing"
        })

    return causes


def _offline_positioning(project: dict, m2: dict, root_causes: list) -> dict:
    industry = project.get("industry_sector", "Technology")
    model = project.get("business_model", "SaaS")
    market = project.get("target_market", "Target Market")
    
    # Adjust positioning recommendations based on top root cause severity
    top_rc_cat = root_causes[0]["category"] if root_causes else "General"
    
    return {
        "core_value_proposition": f"Focused {model} solution tailored specifically to solve domain bottlenecks for {market} within the {industry} sector.",
        "primary_differentiation_angle": f"Vertical specialization addressing key industry friction in {top_rc_cat}.",
        "defensibility_strategy": "Build proprietary workflow integrations and high-switching-cost data loops early with anchor accounts.",
        "pricing_power_assessment": "Moderate early pricing power; transition from pilot discount pricing to value-metric pricing as retention validates ROI.",
        "barrier_to_entry_recommendations": [
            "Secure 3–5 high-profile case studies with verified ROI figures within 90 days.",
            "Develop proprietary integration or domain workflow automations that raise competitor replication costs.",
            "Implement sticky team-based collaboration or telemetry features to drive daily active usage."
        ]
    }


def _offline_mitigations(project: dict, m2: dict, root_causes: list, positioning: dict) -> List[dict]:
    budget = float(project.get("budget", 0))
    model = project.get("business_model", "Business Model")
    diff_angle = positioning.get("primary_differentiation_angle", "vertical differentiation")
    
    return [
        {
            "phase": "Phase 1: Immediate De-Risking (Days 0–30)",
            "focus": f"Assumptions Testing, Burn Containment & Alignment on {diff_angle}",
            "milestone_goal": "Secure 10 verified customer problem interviews and confirm willingness to pay before committing capital.",
            "actions": [
                {
                    "title": "Establish Zero-Burn Customer Discovery Loop",
                    "detail": "Interview 15 target buyers to test if the problem is a 'hair on fire' priority rather than a 'nice to have'.",
                    "owner": "Founder / Product Lead",
                    "priority": "Critical"
                },
                {
                    "title": "Implement Runway Protection Budget",
                    "detail": f"Cap non-essential infrastructure and operational burn to retain at least 9 months of runway from the ₹{budget:,.0f} budget.",
                    "owner": "Operations / Finance",
                    "priority": "High"
                },
                {
                    "title": "Create Clickable Low-Code / Concierge MVP",
                    "detail": "Test value proposition with interactive prototypes before investing in heavy custom engineering.",
                    "owner": "Engineering / Design",
                    "priority": "High"
                }
            ]
        },
        {
            "phase": "Phase 2: Operational & GTM Stabilization (Days 30–60)",
            "focus": "Product-Market Fit Signal & Repeatable Acquisition",
            "milestone_goal": "Achieve 5 active pilot deployments with measurable weekly engagement.",
            "actions": [
                {
                    "title": "Launch Focused Beta Cohort",
                    "detail": "Onboard 5 reference customers under structured pilot agreements with explicit success criteria.",
                    "owner": "Sales / Growth",
                    "priority": "High"
                },
                {
                    "title": "Instrument Telemetry & Usage Analytics",
                    "detail": "Track daily active usage, task completion rate, and drop-off points to drive weekly product iteration sprints.",
                    "owner": "Engineering",
                    "priority": "Medium"
                },
                {
                    "title": f"Refine {model} Pricing Model",
                    "detail": "Transition from trial to paid pilot contracts based on validated customer value metrics.",
                    "owner": "Leadership",
                    "priority": "High"
                }
            ]
        },
        {
            "phase": "Phase 3: Defensible Scaling & Unit Economics (Days 60–90+)",
            "focus": "Moat Expansion, Expansion Guardrails & Growth",
            "milestone_goal": "Demonstrate positive unit economics (LTV:CAC > 3:1) and strong Net Promoter Score before expanding team spend.",
            "actions": [
                {
                    "title": "Build Defensibility & Workflow Moat",
                    "detail": "Introduce integrations and automated reporting that embed deeply into customer daily workflows.",
                    "owner": "Product Lead",
                    "priority": "Medium"
                },
                {
                    "title": "Formalize Unit Economics Dashboard",
                    "detail": "Calculate fully-loaded Customer Acquisition Cost (CAC) and Payback Period to ensure sustainable expansion.",
                    "owner": "Finance",
                    "priority": "High"
                },
                {
                    "title": "Prepare Investor & Strategic Growth Package",
                    "detail": "Assemble validation metrics, customer testimonials, and unit economics proof for follow-on capitalization.",
                    "owner": "Founder",
                    "priority": "Medium"
                }
            ]
        }
    ]


def _offline_synthesis(project: dict, m2: dict, root_causes: list, positioning: dict, mitigations: list) -> dict:
    overall_m2_risk = m2.get("riskScoring", {}).get("overallScore", 50)
    feasibility_score = m2.get("feasibility", {}).get("overallScore", 50)
    
    # Calculate Strategic Recommendation Score (0-100)
    # Higher score = stronger strategic clarity and execution actionability
    raw_score = int(round(max(20, min(95, feasibility_score * 0.7 + (100 - overall_m2_risk) * 0.3))))
    rec_score = _validate_and_clamp_score(raw_score, fallback=70)
    
    status_label = (
        "High Execution Readiness" if rec_score >= 75
        else "Moderate Execution Readiness" if rec_score >= 55
        else "Validation Required Prior to Scaling"
    )

    return {
        "strategic_recommendation_score": rec_score,
        "recommendation_label": status_label,
        "executive_verdict": (
            f"The venture demonstrates viable fundamentals in '{project.get('industry_sector')}' with strong potential "
            f"if execution focuses on early customer discovery and disciplined capital allocation. "
            f"Execution should prioritize the Phase 1 de-risking roadmap to validate demand assumptions before scaling spend."
        ),
        "bull_case_scenario": (
            "Rapid validation with anchor customers proves high willingness to pay, shortening sales cycles "
            "and allowing self-funded organic expansion or non-dilutive grant funding."
        ),
        "bear_case_scenario": (
            "Customer acquisition costs remain elevated due to broad market positioning, exhausting cash runway "
            "before achieving repeatable retention metrics."
        ),
        "strategic_priorities": [
            "Contain capital expenditure until 5 reference customers are actively engaged.",
            "Narrow initial marketing focus exclusively to the highest-urgency niche segment.",
            "Establish weekly retention and unit economics feedback loops."
        ]
    }


# =====================================================================
# 6. LangGraph Nodes & Analytical Graph Workflow
# =====================================================================

def node_root_cause_analysis(state: StrategyState) -> StrategyState:
    """Node 1: Decomposes primary risk drivers and critical vulnerabilities."""
    project = state["project"]
    m2 = state["milestone2_analysis"]
    provider = state.get("provider_info", {}).get("name", "offline")
    
    root_causes = None
    actual_used = "offline"

    if provider in ("gemini", "openai"):
        system_instruction = (
            "You are a top-tier Venture Capital Partner and Startup Risk Analyst. "
            "Analyze startup submission data and Milestone 2 risk breakdowns to identify root causes and vulnerabilities. "
            "Respond strictly in JSON matching the schema: {\"root_causes\": [{\"category\": str, \"severity\": \"High\"|\"Medium\"|\"Low\", \"finding\": str, \"impact\": str, \"urgency\": str}]}"
        )
        prompt = (
            f"Project: {json.dumps(project, default=str)}\n\n"
            f"Milestone 2 Risk Breakdown: {json.dumps(m2.get('riskScoring', {}), default=str)}\n"
            f"Feasibility Breakdown: {json.dumps(m2.get('feasibility', {}), default=str)}\n\n"
            "Generate 3 to 4 specific root cause diagnostics explaining what could cause this project to fail and what needs immediate attention."
        )
        data, used = query_llm_json(prompt, system_instruction)
        actual_used = used
        if data and "root_causes" in data and isinstance(data["root_causes"], list):
            root_causes = data["root_causes"]

    if not root_causes:
        root_causes = _offline_root_causes(project, m2)

    updated_provider_info = dict(state.get("provider_info", {}))
    updated_provider_info["name"] = actual_used

    return {**state, "root_causes": root_causes, "provider_info": updated_provider_info}


def node_positioning_analysis(state: StrategyState) -> StrategyState:
    """Node 2: Evaluates competitive defensibility, moat construction, and positioning (takes root_causes)."""
    project = state["project"]
    m2 = state["milestone2_analysis"]
    root_causes = state.get("root_causes", [])
    provider = state.get("provider_info", {}).get("name", "offline")

    positioning = None
    actual_used = "offline"

    if provider in ("gemini", "openai"):
        system_instruction = (
            "You are a Competitive Strategy Expert and Product Positioning Advisor. "
            "Evaluate market positioning, moat construction, and defensibility strategies for a startup based on its SWOT and identified root causes. "
            "Respond strictly in JSON matching: {\"core_value_proposition\": str, \"primary_differentiation_angle\": str, "
            "\"defensibility_strategy\": str, \"pricing_power_assessment\": str, \"barrier_to_entry_recommendations\": [str, str, str]}"
        )
        prompt = (
            f"Project: {json.dumps(project, default=str)}\n\n"
            f"SWOT Opportunities & Threats: {json.dumps(m2.get('swot', {}), default=str)}\n\n"
            f"Identified Root Causes: {json.dumps(root_causes, default=str)}\n\n"
            "Provide tailored positioning and defensibility strategies to neutralize these specific vulnerabilities."
        )
        data, used = query_llm_json(prompt, system_instruction)
        actual_used = used
        if data and "defensibility_strategy" in data:
            positioning = data

    if not positioning:
        positioning = _offline_positioning(project, m2, root_causes)

    updated_provider_info = dict(state.get("provider_info", {}))
    updated_provider_info["name"] = actual_used

    return {**state, "positioning_analysis": positioning, "provider_info": updated_provider_info}


def node_mitigation_roadmap(state: StrategyState) -> StrategyState:
    """
    Node 3: Generates concrete 3-phase action plan.
    Explicitly feeds on BOTH Root Causes and Positioning Analysis from upstream nodes.
    """
    project = state["project"]
    m2 = state["milestone2_analysis"]
    root_causes = state.get("root_causes", [])
    positioning = state.get("positioning_analysis", {})
    provider = state.get("provider_info", {}).get("name", "offline")

    roadmap = None
    actual_used = "offline"

    if provider in ("gemini", "openai"):
        system_instruction = (
            "You are an Operating Partner at an early-stage startup accelerator. "
            "Generate a realistic 3-phase strategic mitigation roadmap (Days 0-30, Days 30-60, Days 60-90+) "
            "with prioritized, actionable steps for founders that directly address the identified root causes "
            "and execute the competitive positioning strategy. "
            "Respond strictly in JSON format: {\"mitigation_roadmap\": [{\"phase\": str, \"focus\": str, \"milestone_goal\": str, "
            "\"actions\": [{\"title\": str, \"detail\": str, \"owner\": str, \"priority\": \"Critical\"|\"High\"|\"Medium\"}]}]}"
        )
        prompt = (
            f"Project: {json.dumps(project, default=str)}\n\n"
            f"Identified Root Causes & Vulnerabilities: {json.dumps(root_causes, default=str)}\n\n"
            f"Competitive Positioning & Moat Strategy: {json.dumps(positioning, default=str)}\n\n"
            "Build a 3-phase tactical roadmap for the founding team that leverages this positioning and resolves the root causes."
        )
        data, used = query_llm_json(prompt, system_instruction)
        actual_used = used
        if data and "mitigation_roadmap" in data and isinstance(data["mitigation_roadmap"], list):
            roadmap = data["mitigation_roadmap"]

    if not roadmap:
        roadmap = _offline_mitigations(project, m2, root_causes, positioning)

    updated_provider_info = dict(state.get("provider_info", {}))
    updated_provider_info["name"] = actual_used

    return {**state, "mitigation_roadmap": roadmap, "provider_info": updated_provider_info}


def node_executive_synthesis(state: StrategyState) -> StrategyState:
    """
    Node 4: Produces executive synthesis, scenario modeling, and validated Strategic Recommendation Score.
    Takes Root Causes, Positioning, and Mitigation Roadmap into its synthesis.
    """
    project = state["project"]
    m2 = state["milestone2_analysis"]
    root_causes = state.get("root_causes", [])
    positioning = state.get("positioning_analysis", {})
    roadmap = state.get("mitigation_roadmap", [])
    provider = state.get("provider_info", {}).get("name", "offline")

    synthesis = None
    actual_used = "offline"

    if provider in ("gemini", "openai"):
        system_instruction = (
            "You are an Executive Venture Advisor. Synthesize the findings into an actionable executive summary, "
            "calculate a Strategic Recommendation Score (0-100 evaluating actionability and strategy completeness; "
            "DO NOT override or recalculate the existing Milestone 2 Risk Score), and outline bull/bear scenarios. "
            "Respond strictly in JSON format: {\"strategic_recommendation_score\": int, \"recommendation_label\": str, "
            "\"executive_verdict\": str, \"bull_case_scenario\": str, \"bear_case_scenario\": str, \"strategic_priorities\": [str, str, str]}"
        )
        prompt = (
            f"Project: {json.dumps(project, default=str)}\n\n"
            f"Milestone 2 Feasibility Score: {m2.get('feasibility', {}).get('overallScore')}\n\n"
            f"Root Causes: {json.dumps(root_causes, default=str)}\n\n"
            f"Positioning & Moat Strategy: {json.dumps(positioning, default=str)}\n\n"
            f"Mitigation Roadmap: {json.dumps(roadmap, default=str)}\n\n"
            "Synthesize strategic verdict, scenario forecasts, and strategic recommendation score."
        )
        data, used = query_llm_json(prompt, system_instruction)
        actual_used = used
        if data and "strategic_recommendation_score" in data:
            synthesis = data

    if not synthesis:
        synthesis = _offline_synthesis(project, m2, root_causes, positioning, roadmap)

    # Validate and clamp the score strictly to [0, 100]
    validated_rec_score = _validate_and_clamp_score(
        synthesis.get("strategic_recommendation_score"),
        fallback=70
    )
    synthesis["strategic_recommendation_score"] = validated_rec_score

    updated_provider_info = dict(state.get("provider_info", {}))
    updated_provider_info["name"] = actual_used

    return {
        **state,
        "executive_synthesis": synthesis,
        "strategic_recommendation_score": validated_rec_score,
        "confidence_score": m2.get("riskScoring", {}).get("confidenceScore", 80),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_info": updated_provider_info,
    }


# =====================================================================
# 7. LangGraph Compilation & Execution Wrapper
# =====================================================================

def _build_and_run_langgraph(initial_state: StrategyState) -> StrategyState:
    """
    Constructs and executes the LangGraph StateGraph:
    START -> analyze_root_causes -> evaluate_positioning -> generate_mitigations -> synthesize_strategy -> END
    """
    try:
        from langgraph.graph import StateGraph, START, END
        
        graph = StateGraph(StrategyState)
        graph.add_node("analyze_root_causes", node_root_cause_analysis)
        graph.add_node("evaluate_positioning", node_positioning_analysis)
        graph.add_node("generate_mitigations", node_mitigation_roadmap)
        graph.add_node("synthesize_strategy", node_executive_synthesis)
        
        graph.add_edge(START, "analyze_root_causes")
        graph.add_edge("analyze_root_causes", "evaluate_positioning")
        graph.add_edge("evaluate_positioning", "generate_mitigations")
        graph.add_edge("generate_mitigations", "synthesize_strategy")
        graph.add_edge("synthesize_strategy", END)
        
        app = graph.compile()
        result = app.invoke(initial_state)
        return result
    except ImportError:
        # Graceful, deterministic node-by-node execution matching the exact LangGraph transition flow
        s1 = node_root_cause_analysis(initial_state)
        s2 = node_positioning_analysis(s1)
        s3 = node_mitigation_roadmap(s2)
        s4 = node_executive_synthesis(s3)
        return s4


# =====================================================================
# 8. In-Memory Caching & Public Strategy API
# =====================================================================

_STRATEGY_CACHE: Dict[int, Dict[str, Any]] = {}

def get_or_generate_strategy(
    project: Dict[str, Any],
    milestone2_analysis: Dict[str, Any],
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Returns the cached strategic advisory report or executes the LangGraph workflow afresh.
    Passing force_refresh=True explicitly bypasses and updates the cache.
    
    Guarantees:
    1. Provider reflects the ACTUAL backend that produced the strategy (accurately shows "offline" if fallback was used).
    2. Information flow is strictly Root Causes -> Positioning -> Mitigation -> Executive Strategy.
    3. Strategic Recommendation Score is strictly validated, numeric integer, and clamped to [0, 100].
    4. Milestone 2 Overall Risk Score is never modified.
    """
    project_id = project.get("project_id")
    
    if not force_refresh and project_id is not None and project_id in _STRATEGY_CACHE:
        cached = _STRATEGY_CACHE[project_id]
        cached["is_cached"] = True
        return cached

    provider_name, provider_reason, _ = resolve_llm_provider()

    initial_state: StrategyState = {
        "project": project,
        "milestone2_analysis": milestone2_analysis,
        "provider_info": {
            "name": provider_name,
            "description": provider_reason,
            "orchestrator": "LangGraph StateGraph",
        },
        "is_cached": False,
    }

    final_state = _build_and_run_langgraph(initial_state)

    # Determine the actual provider that completed the run
    actual_provider = final_state.get("provider_info", {}).get("name", "offline")
    if actual_provider == "offline":
        actual_description = "Offline heuristic reasoning engine"
    elif actual_provider == "gemini":
        actual_description = "Google Gemini LLM reasoning layer"
    elif actual_provider == "openai":
        actual_description = "OpenAI LLM reasoning layer"
    else:
        actual_description = f"{actual_provider} reasoning layer"

    final_provider_info = {
        "name": actual_provider,
        "description": actual_description,
        "orchestrator": "LangGraph StateGraph",
    }

    validated_score = _validate_and_clamp_score(
        final_state.get("strategic_recommendation_score", 70),
        fallback=70
    )

    output = {
        "status": "success",
        "project_id": project_id,
        "project_name": project.get("project_name"),
        "industry_sector": project.get("industry_sector"),
        "provider_info": final_provider_info,
        "scores": {
            "baseline_failure_risk_pct": milestone2_analysis.get("riskScoring", {}).get("overallScore"),
            "baseline_risk_level": milestone2_analysis.get("riskScoring", {}).get("riskLevel"),
            "strategic_recommendation_score": validated_score,
            "feasibility_overall_score": milestone2_analysis.get("feasibility", {}).get("overallScore"),
            "confidence_pct": final_state.get("confidence_score", 80),
        },
        "root_causes": final_state.get("root_causes", []),
        "positioning_analysis": final_state.get("positioning_analysis", {}),
        "mitigation_roadmap": final_state.get("mitigation_roadmap", []),
        "executive_synthesis": final_state.get("executive_synthesis", {}),
        "generated_at": final_state.get("generated_at", datetime.now(timezone.utc).isoformat()),
        "is_cached": False,
    }

    if project_id is not None:
        _STRATEGY_CACHE[project_id] = dict(output)

    return output


def clear_strategy_cache(project_id: Optional[int] = None) -> None:
    """Utility to clear cache for testing or maintenance."""
    global _STRATEGY_CACHE
    if project_id is not None:
        _STRATEGY_CACHE.pop(project_id, None)
    else:
        _STRATEGY_CACHE.clear()

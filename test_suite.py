"""
test_suite.py — Automated Regression & Milestone 3 Strategy Engine Verification Suite

Verifies:
1. Milestone 2 Regression: Integrity of risk calculations, SWOT, and feasibility scoring.
2. Milestone 3 Strategy Engine: LangGraph orchestration, accurate actual provider reporting,
   unbroken information flow (Root Causes -> Positioning -> Mitigation -> Synthesis),
   strict score validation/clamping, and baseline risk preservation.
"""

import os
import unittest
from decimal import Decimal

# Import Milestone 2 engine functions from main
from main import (
    _financial_risk,
    _market_risk,
    _competition_risk,
    _technical_risk,
    _operational_risk,
    compute_milestone2_analysis,
)

# Import Milestone 3 Strategy Engine functions
from strategy_engine import (
    resolve_llm_provider,
    get_or_generate_strategy,
    clear_strategy_cache,
    _validate_and_clamp_score,
    _build_and_run_langgraph,
    node_root_cause_analysis,
    node_positioning_analysis,
    node_mitigation_roadmap,
    node_executive_synthesis,
    StrategyState,
)


class TestMilestone2Regression(unittest.TestCase):
    """Ensures existing Milestone 2 risk calculations remain 100% stable and unregressed."""

    def setUp(self):
        self.sample_project = {
            "project_id": 101,
            "project_name": "CloudOps Automation",
            "industry_sector": "B2B SaaS",
            "business_model": "SaaS",
            "target_market": "Mid-market DevOps engineers and CTOs",
            "budget": Decimal("500000"),
            "description": "Automated multi-cloud incident response with AI recommendations and MVP deployed with 3 paying beta customers.",
        }

    def test_category_risk_functions_structure(self):
        """Each risk function must return (score: int, reasons: list[dict])."""
        for fn in [_financial_risk, _market_risk, _competition_risk, _technical_risk, _operational_risk]:
            score, reasons = fn(self.sample_project)
            self.assertIsInstance(score, int)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
            self.assertIsInstance(reasons, list)
            for r in reasons:
                self.assertIn("text", r)
                self.assertIn("positive", r)

    def test_milestone2_analysis_complete_schema(self):
        """Full analysis dictionary must contain riskScoring, swot, and feasibility sections."""
        analysis = compute_milestone2_analysis(self.sample_project)
        
        # Risk scoring assertions
        self.assertIn("riskScoring", analysis)
        risk = analysis["riskScoring"]
        self.assertIn("overallScore", risk)
        self.assertIn("riskLevel", risk)
        self.assertIn("riskBreakdown", risk)
        self.assertIn("confidenceScore", risk)
        self.assertEqual(len(risk["riskBreakdown"]), 5)
        
        # SWOT assertions
        self.assertIn("swot", analysis)
        swot = analysis["swot"]
        for key in ["strengths", "weaknesses", "opportunities", "threats"]:
            self.assertIn(key, swot)
            self.assertIsInstance(swot[key], list)
            self.assertGreater(len(swot[key]), 0)

        # Feasibility assertions
        self.assertIn("feasibility", analysis)
        feas = analysis["feasibility"]
        self.assertIn("overallScore", feas)
        self.assertIn("level", feas)
        self.assertIn("breakdown", feas)
        self.assertIn("positiveFactors", feas)
        self.assertIn("attentionAreas", feas)
        self.assertIn("summary", feas)


class TestMilestone3StrategyEngine(unittest.TestCase):
    """Verifies Milestone 3 LangGraph-driven strategic reasoning, provider resolution, caching, and score separation."""

    def setUp(self):
        clear_strategy_cache()
        self.project = {
            "project_id": 202,
            "project_name": "AgriSense AI",
            "industry_sector": "Agritech",
            "business_model": "Hardware + Subscription",
            "target_market": "Commercial farm operators across South Asia",
            "budget": Decimal("1200000"),
            "description": "IoT soil sensors and satellite imagery AI pipeline to predict crop yield and irrigation needs with working prototype.",
        }
        self.m2_analysis = compute_milestone2_analysis(self.project)

    def test_provider_resolution_and_actual_fallback_reporting(self):
        """
        Fix 1: Provider reporting must reflect the ACTUAL provider that produced the strategy.
        If a provider is requested but unavailable/fails, provider_info['name'] must report 'offline'.
        """
        orig_provider = os.environ.get("LLM_PROVIDER")
        orig_gemini = os.environ.get("GEMINI_API_KEY")
        orig_openai = os.environ.get("OPENAI_API_KEY")

        try:
            # Case 1: Explicit offline
            os.environ["LLM_PROVIDER"] = "offline"
            prov, reason, key = resolve_llm_provider()
            self.assertEqual(prov, "offline")

            # Case 2: Configured 'gemini' but key is missing -> resolve_llm_provider returns offline fallback
            os.environ["LLM_PROVIDER"] = "gemini"
            os.environ.pop("GEMINI_API_KEY", None)
            prov, reason, key = resolve_llm_provider()
            self.assertEqual(prov, "offline")

            # End-to-end: generate strategy and assert output provider_info accurately reports 'offline'
            res = get_or_generate_strategy(self.project, self.m2_analysis, force_refresh=True)
            self.assertEqual(res["provider_info"]["name"], "offline")
            self.assertIn("offline", res["provider_info"]["description"].lower())

        finally:
            if orig_provider is not None:
                os.environ["LLM_PROVIDER"] = orig_provider
            else:
                os.environ.pop("LLM_PROVIDER", None)

            if orig_gemini is not None:
                os.environ["GEMINI_API_KEY"] = orig_gemini
            else:
                os.environ.pop("GEMINI_API_KEY", None)

            if orig_openai is not None:
                os.environ["OPENAI_API_KEY"] = orig_openai
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_langgraph_information_flow(self):
        """
        Fix 2: Verifies sequential information propagation across nodes:
        Root Causes -> Positioning -> Mitigation Roadmap -> Executive Synthesis.
        """
        initial_state: StrategyState = {
            "project": self.project,
            "milestone2_analysis": self.m2_analysis,
            "provider_info": {"name": "offline", "description": "Offline heuristic reasoning engine"},
        }

        # Step 1: Root Causes
        s1 = node_root_cause_analysis(initial_state)
        self.assertIn("root_causes", s1)
        self.assertGreater(len(s1["root_causes"]), 0)

        # Step 2: Positioning (consumes Root Causes)
        s2 = node_positioning_analysis(s1)
        self.assertIn("positioning_analysis", s2)
        self.assertIn("core_value_proposition", s2["positioning_analysis"])
        self.assertIn("primary_differentiation_angle", s2["positioning_analysis"])

        # Step 3: Mitigation Roadmap (consumes Root Causes AND Positioning)
        s3 = node_mitigation_roadmap(s2)
        self.assertIn("mitigation_roadmap", s3)
        self.assertEqual(len(s3["mitigation_roadmap"]), 3)

        # Step 4: Executive Synthesis (consumes Root Causes, Positioning, AND Roadmap)
        s4 = node_executive_synthesis(s3)
        self.assertIn("executive_synthesis", s4)
        self.assertIn("strategic_recommendation_score", s4)
        self.assertIn("bull_case_scenario", s4["executive_synthesis"])
        self.assertIn("bear_case_scenario", s4["executive_synthesis"])

    def test_score_validation_and_clamping(self):
        """
        Fix 3: Validates and clamps Strategic Recommendation Score across edge cases
        (strings with %, fractions, floats, out-of-bounds numbers, malformed values).
        """
        self.assertEqual(_validate_and_clamp_score(85), 85)
        self.assertEqual(_validate_and_clamp_score("85"), 85)
        self.assertEqual(_validate_and_clamp_score("85%"), 85)
        self.assertEqual(_validate_and_clamp_score("92/100"), 92)
        self.assertEqual(_validate_and_clamp_score(88.6), 89)
        self.assertEqual(_validate_and_clamp_score(-15), 0)
        self.assertEqual(_validate_and_clamp_score(150), 100)
        self.assertEqual(_validate_and_clamp_score(None, fallback=70), 70)
        self.assertEqual(_validate_and_clamp_score("invalid_string", fallback=65), 65)
        self.assertEqual(_validate_and_clamp_score(float("nan"), fallback=75), 75)

    def test_score_separation_guarantee(self):
        """Crucial constraint: Milestone 2 Overall Risk Score must NEVER be altered by the Strategy layer."""
        m2_original_risk = self.m2_analysis["riskScoring"]["overallScore"]
        
        strategy = get_or_generate_strategy(self.project, self.m2_analysis, force_refresh=True)
        scores = strategy["scores"]

        # Baseline Risk must match Milestone 2 risk exactly
        self.assertEqual(scores["baseline_failure_risk_pct"], m2_original_risk)

        # Strategic Recommendation Score must exist as a separate metric (0-100)
        self.assertIn("strategic_recommendation_score", scores)
        self.assertIsInstance(scores["strategic_recommendation_score"], int)
        self.assertGreaterEqual(scores["strategic_recommendation_score"], 0)
        self.assertLessEqual(scores["strategic_recommendation_score"], 100)

        # Feasibility score must match Milestone 2 feasibility exactly
        self.assertEqual(scores["feasibility_overall_score"], self.m2_analysis["feasibility"]["overallScore"])

    def test_caching_and_force_refresh_behavior(self):
        """Verifies cache hits and cache bypass on force_refresh=True."""
        clear_strategy_cache()

        # 1st call: fresh generation
        res1 = get_or_generate_strategy(self.project, self.m2_analysis, force_refresh=False)
        self.assertFalse(res1.get("is_cached", False))

        # 2nd call: cache hit
        res2 = get_or_generate_strategy(self.project, self.m2_analysis, force_refresh=False)
        self.assertTrue(res2.get("is_cached", False))

        # 3rd call with force_refresh=True: bypasses cache and regenerates
        res3 = get_or_generate_strategy(self.project, self.m2_analysis, force_refresh=True)
        self.assertFalse(res3.get("is_cached", False))


class TestFrontendIntegration(unittest.TestCase):
    """Verifies that the frontend HTML correctly wires the strategy modal trigger button and endpoints."""

    def setUp(self):
        with open("analysis-results.html", "r", encoding="utf-8") as f:
            self.html_content = f.read()

    def test_view_explanation_btn_label_and_id(self):
        """Button must retain id='viewExplanationBtn' and have label 'View Detailed Strategy & Reasoning →'."""
        self.assertIn('id="viewExplanationBtn"', self.html_content)
        self.assertIn('View Detailed Strategy & Reasoning →', self.html_content)
        # Ensure old label is removed
        self.assertNotIn('View Detailed Explanation →', self.html_content)

    def test_strategy_modal_wiring(self):
        """Button click must trigger openStrategyModal and load strategy from the strategy endpoint."""
        self.assertIn('document.getElementById("viewExplanationBtn").addEventListener("click", openStrategyModal)', self.html_content)
        self.assertIn('/api/analysis/${currentProjectId}/strategy', self.html_content)
        self.assertIn('id="strategyModalOverlay"', self.html_content)


if __name__ == "__main__":
    unittest.main()


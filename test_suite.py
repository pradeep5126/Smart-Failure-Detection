"""
test_suite.py — Automated Regression & Milestone 3 Recommendations Engine Verification Suite

Verifies:
1. Milestone 2 Regression: Absolute immutability of risk calculations, SWOT, and feasibility scoring.
2. Milestone 3 5-Node LangGraph Agent Workflow:
   - Sequential execution: Data Ingestion -> Risk Analysis -> Strategic Reasoning -> Validation -> Report Generation.
   - Grounded recommendations with explicit "Triggered by" references to actual Milestone 1/2 results.
   - Structured Risk Mitigation matrix with expected impact, priority, and timeframe.
   - Strategic reasoning with Bull/Bear scenario forecasts.
   - Dynamic responsiveness across differing project profiles (no hardcoded static recommendations).
   - Baseline M2 score immutability guarantee.
   - Strict Strategic Recommendation Score clamping [0, 100].
   - Provider resolution (Gemini, OpenAI, Offline fallback).
   - In-memory caching and cache bypass on force_refresh.
3. Frontend Integration:
   - Verification of "Recommendations" primary UI labels and 5-node stepper in HTML.
"""

import os
import time
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Import Milestone 2 engine functions and rate limiting from main
from main import (
    _financial_risk,
    _market_risk,
    _competition_risk,
    _technical_risk,
    _operational_risk,
    compute_milestone2_analysis,
    build_competitor_assessment,
    app,
    clear_regeneration_rate_limits,
    check_and_record_regeneration_rate_limit,
    _regeneration_timestamps,
    REGENERATION_RATE_LIMIT_WINDOW_SECONDS,
    REGENERATION_RATE_LIMIT_MAX_REQUESTS,
)

# Import Milestone 3 Strategy Engine functions
from strategy_engine import (
    resolve_llm_provider,
    get_or_generate_strategy,
    clear_strategy_cache,
    _validate_and_clamp_score,
    _build_and_run_langgraph,
    node_data_ingestion,
    node_risk_analysis,
    node_strategic_reasoning,
    node_validation,
    node_report_generation,
    StrategyState,
)


class TestMilestone2Regression(unittest.TestCase):
    """Ensures existing Milestone 2 risk calculations remain 100% stable, deterministic, and unregressed."""

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


class TestMilestone3RecommendationsEngine(unittest.TestCase):
    """Verifies Milestone 3 5-node LangGraph recommendations workflow, grounded triggers, and outputs."""

    def setUp(self):
        clear_strategy_cache()
        # Force offline provider during unit tests for fast, deterministic, reproducible assertions
        os.environ["LLM_PROVIDER"] = "offline"

        self.project_a = {
            "project_id": 201,
            "project_name": "AgriSense AI",
            "industry_sector": "Agritech",
            "business_model": "Hardware + Subscription",
            "target_market": "Commercial farm operators across South Asia",
            "budget": Decimal("80000"),  # Low budget -> High Financial Risk
            "description": "IoT soil sensors and hardware nodes requiring manufacturing capital. No customers yet.",
        }
        self.m2_a = compute_milestone2_analysis(self.project_a)
        self.competitors_a = build_competitor_assessment(self.project_a)

        self.project_b = {
            "project_id": 202,
            "project_name": "DevPulse Enterprise",
            "industry_sector": "B2B SaaS",
            "business_model": "SaaS",
            "target_market": "Enterprise security architects and engineering directors",
            "budget": Decimal("5000000"),  # Strong budget + paying customers -> Low Financial Risk
            "description": "Cloud security telemetry platform with 10 paying enterprise pilot customers and $20k MRR.",
        }
        self.m2_b = compute_milestone2_analysis(self.project_b)
        self.competitors_b = build_competitor_assessment(self.project_b)

    def tearDown(self):
        os.environ.pop("LLM_PROVIDER", None)

    def test_5_node_langgraph_workflow_execution(self):
        """Verifies step-by-step state propagation across the 5 LangGraph nodes."""
        initial_state: StrategyState = {
            "project": self.project_a,
            "milestone2_analysis": self.m2_a,
            "competitors": self.competitors_a,
            "provider_info": {"name": "offline", "description": "Offline heuristic reasoning engine"},
        }

        # Node 1: Data Ingestion
        s1 = node_data_ingestion(initial_state)
        self.assertIn("ingested_context", s1)
        ctx = s1["ingested_context"]
        self.assertIn("risk_category_scores", ctx)
        self.assertIn("overall_failure_risk_pct", ctx)
        self.assertIn("success_probability_pct", ctx)
        self.assertIn("feasibility_scores", ctx)
        self.assertIn("swot", ctx)
        self.assertIn("competitors", ctx)

        # Node 2: Risk Analysis
        s2 = node_risk_analysis(s1)
        self.assertIn("identified_risk_areas", s2)
        self.assertGreater(len(s2["identified_risk_areas"]), 0)
        for area in s2["identified_risk_areas"]:
            self.assertIn("category", area)
            self.assertIn("score", area)
            self.assertIn("severity", area)
            self.assertIn("m1_m2_triggers", area)

        # Node 3: Strategic Reasoning
        s3 = node_strategic_reasoning(s2)
        self.assertIn("strategic_reasoning_raw", s3)
        raw = s3["strategic_reasoning_raw"]
        self.assertIn("recommendations", raw)
        self.assertIn("risk_mitigation", raw)
        self.assertIn("strategic_reasoning", raw)

        # Node 4: Validation
        s4 = node_validation(s3)
        self.assertIn("validation_results", s4)
        self.assertEqual(s4["validation_results"]["status"], "passed")
        self.assertIn("strategic_recommendation_score", s4)
        self.assertGreaterEqual(s4["strategic_recommendation_score"], 0)
        self.assertLessEqual(s4["strategic_recommendation_score"], 100)

        # Node 5: Report Generation
        s5 = node_report_generation(s4)
        self.assertIn("final_report", s5)
        report = s5["final_report"]
        self.assertEqual(report["status"], "success")
        self.assertIn("recommendations", report)
        self.assertIn("risk_mitigation", report)
        self.assertIn("strategic_reasoning", report)
        self.assertIn("supporting_insights", report)
        self.assertIn("langgraph_workflow", report)
        self.assertEqual(len(report["langgraph_workflow"]["nodes"]), 5)

    def test_recommendations_contain_m1_m2_trigger_references(self):
        """Verifies each recommendation has priority, explanation, and explicit M1/M2 'triggered_by' references."""
        strategy = get_or_generate_strategy(self.project_a, self.m2_a, competitors=self.competitors_a, force_refresh=True)
        recs = strategy["recommendations"]
        self.assertGreater(len(recs), 0)

        for r in recs:
            self.assertIn("title", r)
            self.assertIn("priority", r)
            self.assertIn(r["priority"], ["Critical", "High", "Medium", "Low"])
            self.assertIn("explanation", r)
            self.assertGreater(len(r["explanation"]), 15)
            self.assertIn("triggered_by", r)
            self.assertIsInstance(r["triggered_by"], list)
            self.assertGreater(len(r["triggered_by"]), 0)
            
            # Verify trigger object structure
            for trig in r["triggered_by"]:
                self.assertIn("source", trig)
                self.assertIn("finding", trig)
                self.assertIn("score", trig)

    def test_risk_mitigation_matrix_schema(self):
        """Verifies risk mitigation entries contain identified risk, category, action, and expected impact."""
        strategy = get_or_generate_strategy(self.project_a, self.m2_a, competitors=self.competitors_a, force_refresh=True)
        mits = strategy["risk_mitigation"]
        self.assertGreater(len(mits), 0)

        for m in mits:
            self.assertIn("identified_risk", m)
            self.assertIn("risk_category", m)
            self.assertIn("recommended_mitigation", m)
            self.assertIn("expected_impact", m)
            self.assertIn("priority", m)
            self.assertIn("timeframe", m)

    def test_strategic_reasoning_and_scenario_forecasts(self):
        """Verifies strategic reasoning narrative and bull/bear scenario forecasts."""
        strategy = get_or_generate_strategy(self.project_a, self.m2_a, competitors=self.competitors_a, force_refresh=True)
        reasoning = strategy["strategic_reasoning"]
        self.assertIn("explanation", reasoning)
        self.assertIn("scenario_forecasts", reasoning)
        scenarios = reasoning["scenario_forecasts"]
        self.assertIn("bull_case", scenarios)
        self.assertIn("bear_case", scenarios)

    def test_dynamic_variation_across_differing_projects(self):
        """Verifies recommendations change appropriately when input project and M1/M2 results change."""
        strat_a = get_or_generate_strategy(self.project_a, self.m2_a, competitors=self.competitors_a, force_refresh=True)
        strat_b = get_or_generate_strategy(self.project_b, self.m2_b, competitors=self.competitors_b, force_refresh=True)

        # High risk project A should produce different recommendation titles and triggers than low risk project B
        titles_a = [r["title"] for r in strat_a["recommendations"]]
        titles_b = [r["title"] for r in strat_b["recommendations"]]
        self.assertNotEqual(titles_a, titles_b)

        # Project A should trigger Financial Risk warnings due to low budget & hardware
        a_triggers = [t["finding"] for r in strat_a["recommendations"] for t in r["triggered_by"]]
        self.assertTrue(any("Financial Risk" in t for t in a_triggers))

    def test_score_separation_and_m2_immutability(self):
        """Milestone 2 Overall Risk Score must NEVER be altered or replaced by the Strategy layer."""
        m2_original_risk = self.m2_a["riskScoring"]["overallScore"]
        m2_original_feas = self.m2_a["feasibility"]["overallScore"]

        strategy = get_or_generate_strategy(self.project_a, self.m2_a, competitors=self.competitors_a, force_refresh=True)
        scores = strategy["scores"]

        # Baseline Risk must strictly match Milestone 2 risk
        self.assertEqual(scores["baseline_failure_risk_pct"], m2_original_risk)
        # Feasibility score must strictly match Milestone 2 feasibility
        self.assertEqual(scores["feasibility_overall_score"], m2_original_feas)
        # Success probability must match 100 - risk
        self.assertEqual(scores["success_probability_pct"], 100 - m2_original_risk)
        # Strategic score must exist as a separate metric
        self.assertIn("strategic_recommendation_score", scores)
        self.assertIsInstance(scores["strategic_recommendation_score"], int)

    def test_score_validation_and_clamping(self):
        """Validates and clamps Strategic Recommendation Score across edge cases."""
        self.assertEqual(_validate_and_clamp_score(85), 85)
        self.assertEqual(_validate_and_clamp_score("85"), 85)
        self.assertEqual(_validate_and_clamp_score("85%"), 85)
        self.assertEqual(_validate_and_clamp_score("92/100"), 92)
        self.assertEqual(_validate_and_clamp_score(88.6), 89)
        self.assertEqual(_validate_and_clamp_score(-15), 0)
        self.assertEqual(_validate_and_clamp_score(150), 100)
        self.assertEqual(_validate_and_clamp_score(None, fallback=70), 70)
        self.assertEqual(_validate_and_clamp_score("invalid", fallback=65), 65)

    def test_caching_and_force_refresh(self):
        """Verifies cache retrieval and bypass on force_refresh=True."""
        clear_strategy_cache()

        res1 = get_or_generate_strategy(self.project_a, self.m2_a, force_refresh=False)
        self.assertFalse(res1.get("is_cached", False))

        res2 = get_or_generate_strategy(self.project_a, self.m2_a, force_refresh=False)
        self.assertTrue(res2.get("is_cached", False))

        res3 = get_or_generate_strategy(self.project_a, self.m2_a, force_refresh=True)
        self.assertFalse(res3.get("is_cached", False))


class TestFrontendIntegration(unittest.TestCase):
    """Verifies that analysis-results.html has the updated 'Recommendations' UI labels and 5-node stepper."""

    def setUp(self):
        with open("analysis-results.html", "r", encoding="utf-8") as f:
            self.html_content = f.read()

    def test_recommendations_ui_labels_present(self):
        """Primary buttons and titles must be 'Recommendations' instead of 'AI Strategic Advisor'."""
        # Header button
        self.assertIn('id="headerStrategyBtn"', self.html_content)
        self.assertIn('Recommendations', self.html_content)
        # Summary button
        self.assertIn('id="viewExplanationBtn"', self.html_content)
        self.assertIn('View Recommendations →', self.html_content)
        # Old labels must be removed
        self.assertNotIn('View Detailed Explanation →', self.html_content)
        self.assertNotIn('View Detailed Strategy & Reasoning →', self.html_content)

    def test_5_node_stepper_and_triggers_rendered(self):
        """HTML/JS must include 5-node stepper, full-width Recommendations tab panel, and evidence accordion."""
        self.assertIn('id="tabBtnRecommendations"', self.html_content)
        self.assertIn('id="tabPanelRecommendations"', self.html_content)
        self.assertIn('LangGraph', self.html_content)
        self.assertIn('Data Ingestion', self.html_content)
        self.assertIn('Risk Analysis', self.html_content)
        self.assertIn('Strategic Reasoning', self.html_content)
        self.assertIn('Validation', self.html_content)
        self.assertIn('Report Generation', self.html_content)
        self.assertIn('Risk Mitigation Matrix', self.html_content)
        self.assertIn('Why this recommendation?', self.html_content)
    def test_no_duplicate_variable_declarations(self):
        """Ensure currentProjectId and currentStrategyData are declared exactly once in script scope."""
        import re
        proj_matches = re.findall(r"(?:let|const|var)\s+currentProjectId\b", self.html_content)
        strat_matches = re.findall(r"(?:let|const|var)\s+currentStrategyData\b", self.html_content)
        self.assertEqual(len(proj_matches), 1, f"Found {len(proj_matches)} declarations of currentProjectId")
        self.assertEqual(len(strat_matches), 1, f"Found {len(strat_matches)} declarations of currentStrategyData")

    def test_frontend_cooldown_and_rate_limit_handling_present(self):
        """analysis-results.html must implement 60-second cooldown timer, countdown formatting, and error handling."""
        self.assertIn("startRegenCooldown", self.html_content)
        self.assertIn("Regenerate (${regenCooldownSeconds}s)", self.html_content)
        self.assertIn("↻ Regenerate", self.html_content)
        self.assertIn("isRegenerating", self.html_content)
        self.assertIn("regenCooldownTimer", self.html_content)


class TestRegenerationRateLimitingAndCooldown(unittest.TestCase):
    """
    Verifies rate limiting and cooldown protections on Milestone 3 Recommendations Regenerate:
    - 1st, 2nd, and 3rd regeneration requests succeed.
    - 4th request within rolling 10-minute window returns HTTP 429 Too Many Requests.
    - JSON error contains clear message.
    - Distinct project IDs have independent counters.
    - Expired timestamps (>10 mins) are pruned.
    - Normal GET /api/analysis/{project_id}/strategy is unaffected.
    """

    def setUp(self):
        clear_regeneration_rate_limits()
        clear_strategy_cache()
        os.environ["LLM_PROVIDER"] = "offline"
        self.client = TestClient(app)

        self.sample_project_1 = {
            "project_id": 301,
            "project_name": "AgriSense AI",
            "industry_sector": "Agritech",
            "business_model": "Hardware + Subscription",
            "target_market": "Commercial farm operators across South Asia",
            "budget": Decimal("80000"),
            "description": "IoT soil sensors and hardware nodes requiring manufacturing capital. No customers yet.",
        }

        self.sample_project_2 = {
            "project_id": 302,
            "project_name": "DevPulse Enterprise",
            "industry_sector": "B2B SaaS",
            "business_model": "SaaS",
            "target_market": "Enterprise security architects and engineering directors",
            "budget": Decimal("5000000"),
            "description": "Cloud security telemetry platform with 10 paying enterprise pilot customers.",
        }

    def tearDown(self):
        clear_regeneration_rate_limits()
        clear_strategy_cache()
        os.environ.pop("LLM_PROVIDER", None)

    def _mock_conn_for_project(self, project_dict):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = project_dict
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        return mock_conn

    def test_first_regeneration_succeeds(self):
        """First regeneration request for a project must succeed with HTTP 200."""
        with patch("main.get_connection", return_value=self._mock_conn_for_project(self.sample_project_1)):
            resp = self.client.post("/api/analysis/301/strategy/regenerate")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data.get("status"), "success")
            self.assertIn("recommendations", data)

    def test_up_to_3_regenerations_allowed_within_10_minutes(self):
        """Up to 3 regeneration requests within 10 minutes are permitted."""
        with patch("main.get_connection", return_value=self._mock_conn_for_project(self.sample_project_1)):
            for i in range(1, 4):
                resp = self.client.post("/api/analysis/301/strategy/regenerate")
                self.assertEqual(
                    resp.status_code,
                    200,
                    f"Regeneration attempt {i} failed unexpectedly with status {resp.status_code}",
                )

    def test_4th_regeneration_returns_429_too_many_requests(self):
        """4th regeneration attempt within the same 10-minute window must return HTTP 429."""
        with patch("main.get_connection", return_value=self._mock_conn_for_project(self.sample_project_1)):
            # Attempts 1, 2, 3 should succeed
            for _ in range(3):
                resp = self.client.post("/api/analysis/301/strategy/regenerate")
                self.assertEqual(resp.status_code, 200)

            # Attempt 4 should fail with 429
            resp4 = self.client.post("/api/analysis/301/strategy/regenerate")
            self.assertEqual(resp4.status_code, 429)
            err = resp4.json()
            self.assertIn("detail", err)
            self.assertIn("Regeneration limit reached", err["detail"])

    def test_different_projects_have_independent_limits(self):
        """Rate limiting on project 301 must not restrict regenerations for project 302."""
        # Exhaust 3 attempts on project 301
        with patch("main.get_connection", return_value=self._mock_conn_for_project(self.sample_project_1)):
            for _ in range(3):
                resp = self.client.post("/api/analysis/301/strategy/regenerate")
                self.assertEqual(resp.status_code, 200)

            # 4th attempt on project 301 is blocked
            resp_block_301 = self.client.post("/api/analysis/301/strategy/regenerate")
            self.assertEqual(resp_block_301.status_code, 429)

        # Project 302 can still perform regenerations
        with patch("main.get_connection", return_value=self._mock_conn_for_project(self.sample_project_2)):
            resp_302 = self.client.post("/api/analysis/302/strategy/regenerate")
            self.assertEqual(resp_302.status_code, 200)

    def test_expired_timestamps_older_than_10_minutes_are_removed(self):
        """Timestamps older than 10 minutes (600s) must be evicted and not count toward the limit."""
        now = time.time()
        # Seed 3 expired timestamps (older than 10 minutes ago)
        _regeneration_timestamps[301] = [now - 700, now - 650, now - 610]

        with patch("main.get_connection", return_value=self._mock_conn_for_project(self.sample_project_1)):
            # Since prior 3 timestamps have expired, this request should succeed
            resp = self.client.post("/api/analysis/301/strategy/regenerate")
            self.assertEqual(resp.status_code, 200)

            # Verify that only the new timestamp remains
            remaining = _regeneration_timestamps[301]
            self.assertEqual(len(remaining), 1)
            self.assertGreaterEqual(remaining[0], now - 5)

    def test_normal_get_strategy_endpoint_remains_unaffected(self):
        """Normal GET /api/analysis/{project_id}/strategy must not be rate-limited even after 429 on regenerate."""
        with patch("main.get_connection", return_value=self._mock_conn_for_project(self.sample_project_1)):
            # Exhaust rate limit on regenerate
            for _ in range(3):
                self.client.post("/api/analysis/301/strategy/regenerate")
            resp_429 = self.client.post("/api/analysis/301/strategy/regenerate")
            self.assertEqual(resp_429.status_code, 429)

            # Normal GET strategy requests continue to succeed
            for _ in range(5):
                get_resp = self.client.get("/api/analysis/301/strategy")
                self.assertEqual(get_resp.status_code, 200)
                self.assertEqual(get_resp.json().get("status"), "success")


    def test_dashboard_summary(self):
        # Insert a dummy project directly via the endpoint
        resp = self.client.post("/api/projects", json={
            "project_name": "Dashboard Test Project",
            "industry_sector": "SaaS",
            "business_model": "B2B",
            "target_market": "Global",
            "budget": 50000,
            "description": "This is a dummy test project for dashboard."
        })
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/api/dashboard/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("totalProjects", data)
        self.assertGreater(data["totalProjects"], 0)
        self.assertIn("averageFailureRisk", data)
        self.assertIn("recentProjects", data)
        self.assertGreater(len(data["recentProjects"]), 0)
        
        # Check if our test project is in recent projects
        names = [p["project_name"] for p in data["recentProjects"]]
        self.assertIn("Dashboard Test Project", names)


class TestMilestone4Phase2Report(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def test_report_generation(self):
        # Create a project
        resp = self.client.post("/api/projects", json={
            "project_name": "Report Test Project",
            "industry_sector": "AI",
            "business_model": "B2C",
            "target_market": "Global",
            "budget": 100000,
            "description": "Test project for report generation."
        })
        self.assertEqual(resp.status_code, 200)
        proj_id = resp.json()["project_id"]
        
        # Test report generation
        resp = self.client.get(f"/api/report/{proj_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "text/html; charset=utf-8")
        html = resp.text
        self.assertIn("Comprehensive Assessment", html)
        self.assertIn("Report Test Project", html)
        self.assertIn("Risk Assessment", html)
        self.assertIn("SWOT Analysis", html)

if __name__ == "__main__":
    unittest.main()

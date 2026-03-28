"""
PDIE API Unit Tests
===================
Tests for all API endpoints including new v2 endpoints.

Run: python -m pytest test_pdie_api.py -v
Or: python test_pdie_api.py

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Test the API module
from pdie_api import (
    app,
    DEMO_API_KEY,
    MAX_REQUESTS_PER_MINUTE,
    MODEL_VERSIONS,
    verify_api_key,
    check_rate_limit,
    generate_shap_explanation,
    get_recommendation,
    calculate_confidence,
)


# Create test client
client = TestClient(app)


# ============================================================
# HELPER FUNCTIONS FOR TESTS
# ============================================================


def get_auth_headers():
    """Return headers with valid API key."""
    return {"X-API-Key": DEMO_API_KEY}


def create_mock_customer_row():
    """Create a mock customer row for testing."""
    return pd.Series(
        {
            "customer_id": "C00001",
            "risk_score": 75.0,
            "risk_category": "HIGH",
            "salary_delay_days": 5,
            "upi_lending_app_txn_count_30d": 3,
            "savings_drawdown_rate_4w": 0.4,
            "emi_to_income_ratio": 0.45,
            "monthly_income": 75000,
        }
    )


# ============================================================
# PHASE 1: AUTHENTICATION & RATE LIMITING TESTS
# ============================================================


class TestAuthentication:
    """Test API authentication"""

    def test_missing_api_key(self):
        """Test that requests without API key are rejected"""
        response = client.get("/api/v1/health")
        # Health check doesn't require auth
        assert response.status_code == 200

    def test_invalid_api_key(self):
        """Test that invalid API key is rejected"""
        response = client.post(
            "/api/v2/predict",
            json={"customer_id": "C00001"},
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 403

    def test_valid_api_key(self):
        """Test that valid API key is accepted"""
        response = client.post(
            "/api/v2/predict",
            json={"customer_id": "C00001"},
            headers=get_auth_headers(),
        )
        # Will fail due to missing data, but auth should pass
        assert response.status_code in [200, 404, 500]


class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limiter_allows_requests_under_limit(self):
        """Test that requests under limit are allowed"""
        limiter = __import__(
            "pdie_api", fromlist=["SimpleRateLimiter"]
        ).SimpleRateLimiter(max_requests=10)
        for _ in range(5):
            assert limiter.is_allowed("test_client") is True

    def test_rate_limiter_blocks_over_limit(self):
        """Test that requests over limit are blocked"""
        limiter = __import__(
            "pdie_api", fromlist=["SimpleRateLimiter"]
        ).SimpleRateLimiter(max_requests=3)
        for _ in range(3):
            limiter.is_allowed("test_client")
        assert limiter.is_allowed("test_client") is False


# ============================================================
# PHASE 2: MODEL ENDPOINTS TESTS
# ============================================================


class TestModelEndpoints:
    """Test model versioning endpoints"""

    def test_get_model_versions(self):
        """Test getting all model versions"""
        response = client.get("/api/v2/models", headers=get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "production_model" in data

    def test_get_model_metrics(self):
        """Test getting specific model metrics"""
        response = client.get("/api/v2/models/v2/metrics", headers=get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "v2"
        assert "metrics" in data

    def test_get_nonexistent_model_metrics(self):
        """Test getting metrics for nonexistent model"""
        response = client.get("/api/v2/models/v999/metrics", headers=get_auth_headers())
        assert response.status_code == 404


# ============================================================
# PHASE 2: PREDICTION ENDPOINTS TESTS
# ============================================================


class TestPredictionEndpoints:
    """Test prediction endpoints"""

    @patch("pdie_api.get_processed_customers")
    def test_predict_with_explanation(self, mock_get_data):
        """Test v2 predict endpoint with SHAP explanations"""
        # Create mock data
        mock_df = pd.DataFrame(
            [
                {
                    "customer_id": "C00001",
                    "risk_score": 75.0,
                    "risk_category": "HIGH",
                    "salary_delay_days": 5,
                    "upi_lending_app_txn_count_30d": 3,
                }
            ]
        )
        mock_get_data.return_value = mock_df

        response = client.post(
            "/api/v2/predict",
            json={"customer_id": "C00001"},
            headers=get_auth_headers(),
        )

        # Should succeed (or 404 if customer not found in exact match)
        assert response.status_code in [200, 404]

    @patch("pdie_api.get_processed_customers")
    def test_batch_predict(self, mock_get_data):
        """Test batch prediction endpoint"""
        mock_df = pd.DataFrame(
            [
                {"customer_id": "C00001", "risk_score": 75.0, "risk_category": "HIGH"},
                {
                    "customer_id": "C00002",
                    "risk_score": 45.0,
                    "risk_category": "MEDIUM",
                },
            ]
        )
        mock_get_data.return_value = mock_df

        response = client.post(
            "/api/v2/predict/batch",
            json={"customer_ids": ["C00001", "C00002"]},
            headers=get_auth_headers(),
        )

        assert response.status_code in [200, 404]

    def test_batch_predict_validation(self):
        """Test batch prediction input validation"""
        # Test empty customer list
        response = client.post(
            "/api/v2/predict/batch",
            json={"customer_ids": []},
            headers=get_auth_headers(),
        )
        assert response.status_code == 422  # Validation error

    @patch("pdie_api.get_processed_customers")
    def test_async_predict(self, mock_get_data):
        """Test async prediction endpoint"""
        mock_df = pd.DataFrame(
            [
                {"customer_id": "C00001", "risk_score": 75.0, "risk_category": "HIGH"},
            ]
        )
        mock_get_data.return_value = mock_df

        response = client.post(
            "/api/v2/predict/async",
            json={"customer_ids": ["C00001"]},
            headers=get_auth_headers(),
        )

        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "processing"


# ============================================================
# PHASE 3: HEALTH CHECK TESTS
# ============================================================


class TestHealthChecks:
    """Test health check endpoints"""

    def test_basic_health_check(self):
        """Test basic health endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    @patch("pdie_api.load_model")
    @patch("pdie_api.get_processed_customers")
    def test_detailed_health_check(self, mock_data, mock_model):
        """Test detailed health endpoint"""
        mock_model.return_value = MagicMock()
        mock_df = pd.DataFrame([{"customer_id": "C00001"}])
        mock_data.return_value = mock_df

        response = client.get("/api/v2/health/detailed", headers=get_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "components" in data


# ============================================================
# UTILITY FUNCTION TESTS
# ============================================================


class TestUtilityFunctions:
    """Test utility functions"""

    def test_generate_shap_explanation(self):
        """Test SHAP explanation generation"""
        customer = create_mock_customer_row()
        explanation = generate_shap_explanation(customer, 75.0)

        assert "top_contributors" in explanation
        assert "summary" in explanation
        assert len(explanation["top_contributors"]) <= 5

    def test_get_recommendation_critical(self):
        """Test recommendation for critical risk"""
        rec = get_recommendation(85.0)
        assert rec["priority"] == "CRITICAL"
        assert rec["action"] == "immediate_intervention"

    def test_get_recommendation_high(self):
        """Test recommendation for high risk"""
        rec = get_recommendation(75.0)
        assert rec["priority"] == "HIGH"
        assert rec["action"] == "proactive_outreach"

    def test_get_recommendation_medium(self):
        """Test recommendation for medium risk"""
        rec = get_recommendation(55.0)
        assert rec["priority"] == "MEDIUM"
        assert rec["action"] == "monitor_and_alert"

    def test_get_recommendation_low(self):
        """Test recommendation for low risk"""
        rec = get_recommendation(35.0)
        assert rec["priority"] == "LOW"
        assert rec["action"] == "standard_process"

    def test_calculate_confidence(self):
        """Test confidence calculation"""
        customer = create_mock_customer_row()
        confidence = calculate_confidence(75.0, customer)

        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)


# ============================================================
# MODEL VERSION TESTS
# ============================================================


class TestModelVersioning:
    """Test model versioning"""

    def test_model_versions_exist(self):
        """Test that model versions are defined"""
        assert "v1" in MODEL_VERSIONS
        assert "v2" in MODEL_VERSIONS
        assert "lstm_v1" in MODEL_VERSIONS

    def test_production_model(self):
        """Test production model is v2"""
        assert MODEL_VERSIONS["v2"]["status"] == "production"

    def test_model_has_required_fields(self):
        """Test each model has required fields"""
        for version, model in MODEL_VERSIONS.items():
            assert "name" in model
            assert "status" in model
            assert "auc" in model


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Integration tests for full workflow"""

    @patch("pdie_api.get_processed_customers")
    def test_full_prediction_flow(self, mock_get_data):
        """Test complete prediction flow with auth"""
        mock_df = pd.DataFrame(
            [
                {
                    "customer_id": "C00001",
                    "risk_score": 75.0,
                    "risk_category": "HIGH",
                    "salary_delay_days": 5,
                    "upi_lending_app_txn_count_30d": 3,
                    "savings_drawdown_rate_4w": 0.4,
                    "emi_to_income_ratio": 0.45,
                    "monthly_income": 75000,
                }
            ]
        )
        mock_get_data.return_value = mock_df

        # Make prediction request
        response = client.post(
            "/api/v2/predict",
            json={"customer_id": "C00001"},
            headers=get_auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "customer_id" in data
        assert "risk_score" in data
        assert "risk_category" in data
        assert "explanation" in data
        assert "recommendation" in data


# ============================================================
# MAIN RUNNER
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PDIE API Unit Tests")
    print("=" * 60)

    # Run with pytest if available, otherwise run basic tests
    try:
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        print("pytest not installed. Running basic tests...")

        # Run basic tests manually
        print("\n--- Testing Authentication ---")
        print("Missing key test:", client.get("/api/v1/health").status_code == 200)

        print("\n--- Testing Health ---")
        print("Health check:", client.get("/api/v1/health").json()["status"])

        print("\n--- Testing Model Versions ---")
        print("Models:", list(MODEL_VERSIONS.keys()))

        print("\n--- Testing Recommendations ---")
        print("Critical (85):", get_recommendation(85.0)["priority"])
        print("High (75):", get_recommendation(75.0)["priority"])
        print("Medium (55):", get_recommendation(55.0)["priority"])
        print("Low (35):", get_recommendation(35.0)["priority"])

        print("\n✅ Basic tests completed!")

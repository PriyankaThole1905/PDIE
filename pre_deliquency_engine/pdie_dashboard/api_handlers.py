"""
API Handlers — FastAPI REST Endpoints for Recovery Path Engine
Recovery Path Engine v2.0

Endpoints:
  POST /api/v1/simulate   — Run full pathway simulation
  POST /api/v1/offer      — Create offer from simulation result
  GET  /api/v1/audit/{id} — Retrieve audit log

Can run standalone: python api_handlers.py
Or import into existing app.

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from pathway_simulator import (
    CustomerProfile, simulate_all_pathways, load_engine_config,
)
from api_schemas import CustomerInput, SimulateRequest, SimulateResponse


# ─── In-memory audit store (production: use database) ───
_audit_store: Dict[str, Dict] = {}


def _customer_input_to_profile(data: Dict) -> CustomerProfile:
    """Convert API customer input to internal CustomerProfile."""
    loan = data.get("loan", {})
    assets = data.get("assets", {})

    return CustomerProfile(
        customer_id=data.get("customer_id", "C00000"),
        name=data.get("name", ""),
        monthly_income=data.get("monthly_income", 85000),
        essential_expenses=data.get("essential_expenses", 50000),
        principal=loan.get("principal", 500000),
        annual_rate=loan.get("annual_rate", 0.14),
        remaining_months=loan.get("remaining_months", 24),
        emi=loan.get("emi", 0),
        total_liquid_assets=sum(assets.values()) if isinstance(assets, dict) else 0,
        other_debts=data.get("other_debts", []),
        risk_band=data.get("risk_band", "B2"),
        cibil_score=data.get("cibil_score", 680),
    )


def handle_simulate(request_data: Dict) -> Dict:
    """
    Core simulation handler (usable with or without FastAPI).

    Args:
        request_data: Dict matching SimulateRequest schema

    Returns:
        Dict matching SimulateResponse schema
    """
    customer_id = request_data.get("customer_id", "C00000")
    pathways = request_data.get("pathways", None)
    user_config = request_data.get("config", {})

    # Load base config and override with user values
    config = load_engine_config()
    config.update(user_config)

    # Build customer profile from stored data or request
    customer_data = request_data.get("customer", {})
    if not customer_data:
        # Use defaults for demo
        customer_data = {
            "customer_id": customer_id,
            "monthly_income": 85000,
            "essential_expenses": 50000,
            "loan": {"principal": 500000, "annual_rate": 0.14, "remaining_months": 24, "emi": 0},
            "assets": {"FD": 250000, "MF": 120000, "LIC_surrender": 80000},
            "other_debts": [{"type": "cc", "principal": 120000, "rate": 0.42}],
        }

    customer = _customer_input_to_profile(customer_data)
    sim = simulate_all_pathways(customer, config, pathways)

    # Build response
    results_list = []
    for r in sim.results:
        result_dict = {
            "pathway": r.pathway_name,
            "display_name": r.display_name,
            "npv": round(r.npv, 0),
            "recovery_rate": round(r.recovery_rate, 4),
            "acceptance_prob": round(r.acceptance_prob, 4),
            "churn_reduction": round(r.churn_reduction, 4),
            "composite": round(r.composite_score, 4),
            "new_emi": round(r.new_emi, 0),
            "new_tenure_months": r.new_tenure_months,
            "description": r.description,
            "action": r.action,
            "immediate_relief": r.immediate_relief,
            "monthly_savings": round(r.monthly_savings, 0),
            "explainability": r.explainability,
            "short_explanation": r.short_explanation,
            "audit": r.audit or {},
        }
        if r.mc_result:
            result_dict["mc_result"] = r.mc_result
        if r.policy_result:
            result_dict["policy_checks"] = r.policy_result.get("checks", {})

        # Store audit
        if r.audit:
            sim_id = r.audit.get("simulation_id", str(uuid.uuid4())[:8])
            _audit_store[sim_id] = r.audit

        results_list.append(result_dict)

    response = {
        "customer_id": customer_id,
        "results": results_list,
        "recommended": sim.recommended,
        "policy_checks": sim.policy_checks,
        "timestamp": sim.timestamp,
    }

    # Validate response size < 1MB
    response_json = json.dumps(response, default=str)
    if len(response_json) > 1_000_000:
        # Strip audit details to reduce size
        for r in response["results"]:
            r.pop("audit", None)

    return response


def handle_offer(request_data: Dict) -> Dict:
    """Create an offer from a simulation result."""
    customer_id = request_data.get("customer_id", "")
    pathway = request_data.get("pathway", "")
    simulation_id = request_data.get("simulation_id", "")

    return {
        "offer_id": str(uuid.uuid4())[:12],
        "customer_id": customer_id,
        "pathway": pathway,
        "status": "DRAFT",
        "created_at": datetime.now().isoformat(),
        "simulation_reference": simulation_id,
    }


def handle_audit(simulation_id: str) -> Dict:
    """Retrieve audit log for a simulation."""
    if simulation_id in _audit_store:
        return _audit_store[simulation_id]
    return {"error": f"Audit record {simulation_id} not found"}


# ─── FastAPI App (optional) ───

def create_app() -> "FastAPI":
    """Create FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="PDIE Recovery Path Engine API",
        version="2.0.0",
        description="Bank-grade recovery pathway simulation with 5 pathways, Monte Carlo, and audit trails.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/v1/simulate")
    async def simulate(request: Dict):
        return handle_simulate(request)

    @app.post("/api/v1/offer")
    async def offer(request: Dict):
        return handle_offer(request)

    @app.get("/api/v1/audit/{simulation_id}")
    async def audit(simulation_id: str):
        result = handle_audit(simulation_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    return app


if __name__ == "__main__":
    # Quick test without FastAPI
    print("Testing API Handlers (standalone)...\n")

    test_request = {
        "customer_id": "C12345",
        "pathways": ["emi_holiday", "graduated_emi", "icr", "asset_backed", "consolidation"],
        "config": {"discount_rate": 0.08, "mc_runs": 1000},
        "customer": {
            "customer_id": "C12345",
            "monthly_income": 85000,
            "essential_expenses": 50000,
            "loan": {"principal": 500000, "annual_rate": 0.14, "remaining_months": 24, "emi": 18500},
            "assets": {"FD": 250000, "MF": 120000, "LIC_surrender": 80000},
            "other_debts": [{"type": "cc", "principal": 120000, "rate": 0.42}],
            "risk_band": "B2",
            "cibil_score": 680,
        },
    }

    response = handle_simulate(test_request)

    print(f"Customer: {response['customer_id']}")
    print(f"Recommended: {response['recommended']}")
    print(f"Timestamp: {response['timestamp']}")
    print(f"\nResults ({len(response['results'])} pathways):")
    for r in response["results"]:
        print(f"  {r['pathway']:20s} | Composite: {r['composite']:.4f} | NPV: ₹{r['npv']:,.0f} | Recovery: {r['recovery_rate']:.1%}")

    # Check response size
    size = len(json.dumps(response, default=str))
    print(f"\nResponse size: {size:,} bytes {'✅' if size < 1_000_000 else '❌ OVER 1MB'}")
    print("\n✅ API Handler test complete!")

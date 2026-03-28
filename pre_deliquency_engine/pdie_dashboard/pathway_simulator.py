"""
Pathway Simulator — 5 Bank-Grade Recovery Pathways
Recovery Path Engine v2.0

Pathways:
1. EMI Holiday — skip N EMIs, capitalize interest
2. Graduated EMI — phased EMI reduction
3. Income-Contingent Repayment (ICR) — EMI linked to income, Monte Carlo
4. Asset-Backed Liquidity Injection — lien on liquid assets
5. Debt Consolidation — merge debts, waterfall, lower rate

Each pathway returns: NPV, recovery_rate, acceptance_prob, composite_score,
new_emi, audit trail, and explainability text.

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import json
import uuid
import math
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

from npv_library import (
    compute_dicr, compute_acr,
    annual_to_monthly_rate, monthly_discount_factor,
    compute_emi, compute_amortization,
    compute_pv, compute_npv, compute_recovery_rate,
    compute_capitalized_interest, compute_composite_score,
    weighted_average_rate, waterfall_repayment_order,
    solve_tenure_for_emi,
)
from monte_carlo import mc_icr_npv, MCResult
from scoring_service import (
    compute_default_prob, default_prob_series,
    estimate_acceptance, estimate_churn_reduction,
    compute_stress_score,
)
from policy_engine import enforce_all_policies, PolicyCheckResult
from audit_engine import (
    create_audit_record, AuditRecord,
    generate_explainability_text, generate_short_explanation,
)


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class CustomerProfile:
    """Complete customer profile for pathway simulation."""
    customer_id: str
    name: str = "Customer"
    monthly_income: float = 85000
    essential_expenses: float = 50000
    income_sigma: float = 0.15          # income volatility
    income_history: List[Dict] = field(default_factory=list)
    # Assets
    total_liquid_assets: float = 0.0    # FD + MF + LIC etc.
    # Primary loan
    principal: float = 500000
    annual_rate: float = 0.14           # decimal
    remaining_months: int = 24
    emi: float = 0.0                    # computed if 0
    # Other debts
    other_debts: List[Dict] = field(default_factory=list)
    # Risk
    risk_band: str = "B2"
    cibil_score: int = 680
    payment_history_score: float = 0.85  # 0-1

    def __post_init__(self):
        # Ensure all numeric fields are NaN-safe
        self.monthly_income = float(self.monthly_income) if not math.isnan(self.monthly_income) else 85000.0
        self.essential_expenses = float(self.essential_expenses) if not math.isnan(self.essential_expenses) else self.monthly_income * 0.55
        self.principal = float(self.principal) if not math.isnan(self.principal) else 500000.0
        self.annual_rate = float(self.annual_rate) if not math.isnan(self.annual_rate) else 0.14
        self.remaining_months = int(self.remaining_months) if not math.isnan(self.remaining_months) else 24
        self.total_liquid_assets = float(self.total_liquid_assets) if not math.isnan(self.total_liquid_assets) else 450000.0
        
        if self.emi <= 0 or math.isnan(self.emi):
            self.emi = compute_emi(self.principal, self.annual_rate, self.remaining_months)


@dataclass
class PathwayResult:
    """Result for a single pathway simulation."""
    pathway_name: str
    display_name: str
    description: str
    action: str
    # Key metrics
    npv: float = 0.0
    recovery_rate: float = 0.0
    acceptance_prob: float = 0.0
    churn_reduction: float = 0.0
    composite_score: float = 0.0
    # Financial details
    new_emi: float = 0.0
    new_tenure_months: int = 0
    total_interest: float = 0.0
    immediate_relief: str = ""
    total_interest_change: float = 0.0   # positive = more interest
    monthly_savings: float = 0.0
    # Monte Carlo (ICR only)
    mc_result: Optional[Dict] = None
    # Policy
    policy_result: Optional[Dict] = None
    # Audit
    audit: Optional[Dict] = None
    explainability: str = ""
    short_explanation: str = ""
    # Extra details per pathway
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Full simulation result across all pathways."""
    customer_id: str
    timestamp: str
    results: List[PathwayResult]
    recommended: Optional[str] = None
    policy_checks: Dict[str, bool] = field(default_factory=dict)
    config_used: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# ENGINE CONFIG LOADER
# ─────────────────────────────────────────────

def load_engine_config(config_path: Optional[str] = None) -> Dict:
    """Load engine configuration from JSON file."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "engine_config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Sensible defaults
        return {
            "discount_rate": 0.08,
            "phi_icr": 0.22,
            "emi_min": 10000,
            "weights": {"accept": 0.40, "npv": 0.40, "churn": 0.20},
            "mc_runs": 10000,
            "acr_min": 0.75,
            "holiday_reduction_factor": 0.60,
            "default_coefficients": {"a0": -2.0, "a1": -1.5, "a2": -0.8, "a3": 0.5, "a4": 0.3},
            "min_recovery_rate": 0.50,
            "graduated_phases": [
                {"reduction": 0.50, "months": 3},
                {"reduction": 0.25, "months": 3},
                {"reduction": 0.00, "months": -1},
            ],
            "asset_backed": {"lien_multiplier": 1.5, "relief_months": 6, "release_ontime_payments": 6},
            "consolidation": {"rate_discount": 0.02, "min_rate": 0.08, "max_tenure_months": 60},
            "model_version": "2.0.0",
        }


# ─────────────────────────────────────────────
# HELPER: BASELINE NPV
# ─────────────────────────────────────────────

def _baseline_npv(customer: CustomerProfile, config: Dict) -> float:
    """Compute baseline (no-intervention) NPV for comparison."""
    discount_rate = config.get("discount_rate", 0.08)
    dicr = compute_dicr(customer.monthly_income, customer.essential_expenses, customer.emi)
    acr = compute_acr(customer.total_liquid_assets, customer.principal)
    base_p = compute_default_prob(dicr, acr, customer.income_sigma, 0.0,
                                  config.get("default_coefficients"))
    probs = default_prob_series(base_p, customer.remaining_months, 0.98, 0.0)
    cfs = [customer.emi] * customer.remaining_months
    return compute_npv(cfs, probs, discount_rate)


# ─────────────────────────────────────────────
# PATHWAY 1: EMI HOLIDAY
# ─────────────────────────────────────────────

def pathway_emi_holiday(customer: CustomerProfile,
                        config: Dict,
                        holiday_months: int = 2) -> PathwayResult:
    """
    EMI Holiday: skip M_h EMIs, capitalize interest, extend tenure.

    CapitalizedInterest = EMI × [(1+i_m)^M_h - 1] / i_m
    NewPrincipal = Principal + CapitalizedInterest
    NewTenure = OriginalTenure + M_h (or recompute to keep EMI ~constant)
    """
    discount_rate = config.get("discount_rate", 0.08)
    weights = config.get("weights", {"accept": 0.4, "npv": 0.4, "churn": 0.2})
    holiday_factor = config.get("holiday_reduction_factor", 0.60)

    i_m = annual_to_monthly_rate(customer.annual_rate)

    # Capitalized interest
    cap_interest = compute_capitalized_interest(customer.emi, i_m, holiday_months)
    new_principal = customer.principal + cap_interest
    new_tenure = customer.remaining_months + holiday_months

    # New EMI on extended principal and tenure
    new_emi = compute_emi(new_principal, customer.annual_rate, new_tenure)

    # Cashflows: 0 during holiday, then new_emi
    cashflows = [0.0] * holiday_months + [new_emi] * customer.remaining_months

    # Default probabilities: reduced during holiday, gradual return
    dicr = compute_dicr(customer.monthly_income, customer.essential_expenses, customer.emi)
    acr = compute_acr(customer.total_liquid_assets, customer.principal)
    base_p = compute_default_prob(dicr, acr, customer.income_sigma, 0.0,
                                  config.get("default_coefficients"))

    probs = []
    for t in range(new_tenure):
        if t < holiday_months:
            reduction = holiday_factor
        else:
            months_after = t - holiday_months
            reduction = max(0.15, holiday_factor - 0.04 * months_after)
        p_t = base_p * (1.0 - reduction) * (0.98 ** t)
        probs.append(min(max(p_t, 0.001), 0.95))

    # NPV
    npv = compute_npv(cashflows, probs, discount_rate)
    recovery = compute_recovery_rate(npv, customer.principal)

    # Acceptance & churn
    stress = compute_stress_score(customer.emi, customer.monthly_income,
                                   customer.total_liquid_assets)
    acceptance = estimate_acceptance(1.0, stress, "emi_holiday")  # 100% relief during holiday
    churn_red = estimate_churn_reduction("emi_holiday", holiday_factor)

    composite = compute_composite_score(acceptance, recovery, 1.0 - churn_red, weights)

    # Interest analysis
    total_paid = new_emi * customer.remaining_months  # after holiday
    original_total = customer.emi * customer.remaining_months
    interest_change = (total_paid - customer.principal) - (original_total - customer.principal)
    monthly_savings = customer.emi - new_emi

    # Build result
    results_dict = {
        "npv": npv, "recovery_rate": recovery, "acceptance_prob": acceptance,
        "composite": composite, "new_emi": new_emi, "holiday_months": holiday_months,
        "capitalized_interest": cap_interest, "monthly_savings": monthly_savings,
    }
    explainability = generate_explainability_text("emi_holiday", asdict(customer), results_dict, config)
    short = generate_short_explanation("emi_holiday", results_dict)

    # Policy check
    policy = enforce_all_policies(
        "emi_holiday", recovery, new_emi, customer.monthly_income,
        new_tenure, config
    )

    # Audit
    audit = create_audit_record(
        simulation_id=str(uuid.uuid4())[:8],
        customer_input={"customer_id": customer.customer_id, "principal": customer.principal,
                        "emi": customer.emi, "rate": customer.annual_rate},
        config=config, model_version=config.get("model_version", "2.0.0"),
        pathway_name="emi_holiday", results=results_dict,
        policy_checks=policy.checks, explainability=explainability,
    )

    return PathwayResult(
        pathway_name="emi_holiday",
        display_name="EMI Holiday",
        description=f"Skip next {holiday_months} EMIs, interest capitalized",
        action=f"No payment for {holiday_months} months, then ₹{new_emi:,.0f}/month",
        npv=npv, recovery_rate=recovery, acceptance_prob=acceptance,
        churn_reduction=churn_red, composite_score=composite,
        new_emi=new_emi, new_tenure_months=new_tenure,
        total_interest=total_paid - new_principal,
        immediate_relief=f"₹{customer.emi * holiday_months:,.0f} (no payment for {holiday_months} months)",
        total_interest_change=interest_change,
        monthly_savings=monthly_savings,
        policy_result=asdict(policy),
        audit=audit.to_dict(),
        explainability=explainability,
        short_explanation=short,
        details={"capitalized_interest": cap_interest, "new_principal": new_principal},
    )


# ─────────────────────────────────────────────
# PATHWAY 2: GRADUATED EMI REDUCTION
# ─────────────────────────────────────────────

def pathway_graduated_emi(customer: CustomerProfile,
                          config: Dict) -> PathwayResult:
    """
    Graduated EMI: phased reduction then return to normal.

    Phases defined in config: [{reduction, months}, ...]
    Last phase with months=-1 means 'remainder'.
    """
    discount_rate = config.get("discount_rate", 0.08)
    weights = config.get("weights")
    phases = config.get("graduated_phases", [
        {"reduction": 0.50, "months": 3},
        {"reduction": 0.25, "months": 3},
        {"reduction": 0.00, "months": -1},
    ])

    # Build monthly EMI schedule
    emi_schedule = []
    remaining = customer.remaining_months
    for phase in phases:
        r = phase["reduction"]
        m = phase["months"]
        phase_emi = customer.emi * (1.0 - r)
        if m == -1:
            # Fill remainder
            count = remaining - len(emi_schedule)
            emi_schedule.extend([phase_emi] * max(count, 0))
        else:
            emi_schedule.extend([phase_emi] * min(m, remaining - len(emi_schedule)))

    # Pad to full tenure if needed
    while len(emi_schedule) < customer.remaining_months:
        emi_schedule.append(customer.emi)

    total_months = len(emi_schedule)
    effective_emi = sum(emi_schedule) / total_months if total_months > 0 else customer.emi

    # Default probabilities
    dicr = compute_dicr(customer.monthly_income, customer.essential_expenses, customer.emi)
    acr = compute_acr(customer.total_liquid_assets, customer.principal)
    base_p = compute_default_prob(dicr, acr, customer.income_sigma, 0.0,
                                  config.get("default_coefficients"))

    probs = []
    for t in range(total_months):
        # Stress reduction proportional to EMI reduction at that month
        reduction_pct = 1.0 - (emi_schedule[t] / customer.emi)
        stress_red = min(0.70, reduction_pct * 2.0)
        p_t = base_p * (1.0 - stress_red) * (0.98 ** t)
        probs.append(min(max(p_t, 0.001), 0.95))

    # NPV
    npv = compute_npv(emi_schedule, probs, discount_rate)
    recovery = compute_recovery_rate(npv, customer.principal)

    # Acceptance & churn
    avg_relief = 1.0 - (effective_emi / customer.emi) if customer.emi > 0 else 0
    stress = compute_stress_score(customer.emi, customer.monthly_income,
                                   customer.total_liquid_assets)
    acceptance = estimate_acceptance(avg_relief, stress, "graduated_emi")
    churn_red = estimate_churn_reduction("graduated_emi", avg_relief)

    composite = compute_composite_score(acceptance, recovery, 1.0 - churn_red, weights)

    # Interest
    total_paid = sum(emi_schedule)
    original_total = customer.emi * customer.remaining_months
    interest_change = total_paid - original_total
    monthly_savings = customer.emi - effective_emi

    results_dict = {
        "npv": npv, "recovery_rate": recovery, "acceptance_prob": acceptance,
        "composite": composite, "new_emi": effective_emi, "monthly_savings": monthly_savings,
    }
    explainability = generate_explainability_text("graduated_emi", asdict(customer), results_dict, config)
    short = generate_short_explanation("graduated_emi", results_dict)

    policy = enforce_all_policies(
        "graduated_emi", recovery, min(emi_schedule), customer.monthly_income,
        total_months, config
    )

    audit = create_audit_record(
        simulation_id=str(uuid.uuid4())[:8],
        customer_input={"customer_id": customer.customer_id},
        config=config, model_version=config.get("model_version", "2.0.0"),
        pathway_name="graduated_emi", results=results_dict,
        policy_checks=policy.checks, explainability=explainability,
    )

    return PathwayResult(
        pathway_name="graduated_emi",
        display_name="Graduated EMI",
        description=f"Phased EMI reduction: avg ₹{effective_emi:,.0f}/mo",
        action=f"Phase 1: ₹{emi_schedule[0]:,.0f} → Phase 2: ₹{emi_schedule[min(3,len(emi_schedule)-1)]:,.0f} → Full: ₹{customer.emi:,.0f}",
        npv=npv, recovery_rate=recovery, acceptance_prob=acceptance,
        churn_reduction=churn_red, composite_score=composite,
        new_emi=effective_emi, new_tenure_months=total_months,
        total_interest=total_paid - customer.principal,
        immediate_relief=f"₹{customer.emi - emi_schedule[0]:,.0f}/mo saved initially",
        total_interest_change=interest_change,
        monthly_savings=monthly_savings,
        policy_result=asdict(policy),
        audit=audit.to_dict(),
        explainability=explainability,
        short_explanation=short,
        details={"phases": phases, "emi_schedule_preview": emi_schedule[:12]},
    )


# ─────────────────────────────────────────────
# PATHWAY 3: INCOME-CONTINGENT REPAYMENT (ICR)
# ─────────────────────────────────────────────

def pathway_icr(customer: CustomerProfile,
                config: Dict) -> PathwayResult:
    """
    ICR: EMI_t = max(EMI_min, floor(ϕ × income_t))
    Uses Monte Carlo simulation for NPV estimation.
    """
    discount_rate = config.get("discount_rate", 0.08)
    weights = config.get("weights")
    phi = config.get("phi_icr", 0.22)
    emi_min = config.get("emi_min", 10000)
    mc_runs = config.get("mc_runs", 10000)

    # Run Monte Carlo
    mc = mc_icr_npv(
        income_mean=customer.monthly_income,
        income_sigma=customer.income_sigma,
        essential_expenses=customer.essential_expenses,
        outstanding_principal=customer.principal,
        annual_discount_rate=discount_rate,
        phi=phi,
        emi_min=emi_min,
        rho=0.7,
        base_default_prob=0.03,
        months=customer.remaining_months,
        n_runs=mc_runs,
        seed=42,
    )

    npv = mc.mean_npv
    recovery = compute_recovery_rate(npv, customer.principal)

    # Expected EMI = phi * income (approx)
    expected_emi = max(emi_min, int(phi * customer.monthly_income))

    # Acceptance & churn
    relief = 1.0 - (expected_emi / customer.emi) if customer.emi > 0 else 0
    relief = max(0, relief)
    stress = compute_stress_score(customer.emi, customer.monthly_income,
                                   customer.total_liquid_assets)
    acceptance = estimate_acceptance(relief, stress, "icr")
    churn_red = estimate_churn_reduction("icr", max(0, relief))

    composite = compute_composite_score(acceptance, recovery, 1.0 - churn_red, weights)

    monthly_savings = customer.emi - expected_emi

    mc_dict = {
        "mean_npv": mc.mean_npv, "std_npv": mc.std_npv,
        "p5": mc.p5, "p10": mc.p10, "p50": mc.p50,
        "p90": mc.p90, "p95": mc.p95,
        "prob_above_threshold": mc.prob_above_threshold,
        "n_runs": mc.n_runs, "seed": mc.seed,
    }

    results_dict = {
        "npv": npv, "recovery_rate": recovery, "acceptance_prob": acceptance,
        "composite": composite, "new_emi": expected_emi,
        "monthly_savings": monthly_savings,
        "mc_runs": mc_runs, "p5": mc.p5, "p95": mc.p95,
    }
    explainability = generate_explainability_text("icr", asdict(customer), results_dict, config)
    short = generate_short_explanation("icr", results_dict)

    policy = enforce_all_policies(
        "icr", recovery, emi_min, customer.monthly_income,
        customer.remaining_months, config
    )

    audit = create_audit_record(
        simulation_id=str(uuid.uuid4())[:8],
        customer_input={"customer_id": customer.customer_id},
        config=config, model_version=config.get("model_version", "2.0.0"),
        pathway_name="icr", results=results_dict,
        policy_checks=policy.checks, explainability=explainability,
        seed=42,
    )

    return PathwayResult(
        pathway_name="icr",
        display_name="Income-Contingent Repayment",
        description=f"EMI linked to income at {phi*100:.0f}%, floor ₹{emi_min:,}",
        action=f"Expected EMI ~₹{expected_emi:,.0f} (adjusts monthly with income)",
        npv=npv, recovery_rate=recovery, acceptance_prob=acceptance,
        churn_reduction=churn_red, composite_score=composite,
        new_emi=expected_emi, new_tenure_months=customer.remaining_months,
        total_interest=expected_emi * customer.remaining_months - customer.principal,
        immediate_relief=f"EMI adjusts to income (floor ₹{emi_min:,}/mo)",
        total_interest_change=0,
        monthly_savings=monthly_savings,
        mc_result=mc_dict,
        policy_result=asdict(policy),
        audit=audit.to_dict(),
        explainability=explainability,
        short_explanation=short,
        details={"phi": phi, "emi_min": emi_min},
    )


# ─────────────────────────────────────────────
# PATHWAY 4: ASSET-BACKED LIQUIDITY INJECTION
# ─────────────────────────────────────────────

def pathway_asset_backed(customer: CustomerProfile,
                         config: Dict) -> PathwayResult:
    """
    Asset-Backed: place lien on liquid assets, provide relief months.

    LienAmount = min(1.5 × M_req × EMI, Available_Liquid_Assets)
    Relief for L months, then resume normal EMI.
    ACR_min threshold to qualify.
    """
    discount_rate = config.get("discount_rate", 0.08)
    weights = config.get("weights")
    ab_config = config.get("asset_backed", {})
    lien_mult = ab_config.get("lien_multiplier", 1.5)
    relief_months = ab_config.get("relief_months", 6)
    release_payments = ab_config.get("release_ontime_payments", 6)
    acr_min = config.get("acr_min", 0.75)

    # Compute ACR
    acr = compute_acr(customer.total_liquid_assets, customer.principal)

    # Lien amount
    lien_amount = min(lien_mult * relief_months * customer.emi, customer.total_liquid_assets)

    # During relief: reduced EMI (interest-only or partial)
    i_m = annual_to_monthly_rate(customer.annual_rate)
    interest_only_emi = customer.principal * i_m
    relief_emi = max(interest_only_emi, customer.emi * 0.3)  # at least interest-only

    new_tenure = customer.remaining_months + relief_months

    # Cashflows
    cashflows = [relief_emi] * relief_months + [customer.emi] * customer.remaining_months

    # Default probs
    dicr = compute_dicr(customer.monthly_income, customer.essential_expenses, customer.emi)
    base_p = compute_default_prob(dicr, acr, customer.income_sigma, 0.0,
                                  config.get("default_coefficients"))
    probs = []
    for t in range(new_tenure):
        if t < relief_months:
            reduction = 0.50  # lien provides safety net
        else:
            months_after = t - relief_months
            reduction = max(0.10, 0.40 - 0.03 * months_after)
        p_t = base_p * (1.0 - reduction) * (0.98 ** t)
        probs.append(min(max(p_t, 0.001), 0.95))

    npv = compute_npv(cashflows, probs, discount_rate)
    recovery = compute_recovery_rate(npv, customer.principal)

    # Acceptance & churn
    emi_relief_pct = 1.0 - (relief_emi / customer.emi) if customer.emi > 0 else 0
    stress = compute_stress_score(customer.emi, customer.monthly_income,
                                   customer.total_liquid_assets)
    acceptance = estimate_acceptance(emi_relief_pct, stress, "asset_backed")
    churn_red = estimate_churn_reduction("asset_backed", 0.40)

    composite = compute_composite_score(acceptance, recovery, 1.0 - churn_red, weights)

    total_paid = relief_emi * relief_months + customer.emi * customer.remaining_months
    original_total = customer.emi * customer.remaining_months
    interest_change = total_paid - original_total
    monthly_savings = customer.emi - relief_emi

    results_dict = {
        "npv": npv, "recovery_rate": recovery, "acceptance_prob": acceptance,
        "composite": composite, "new_emi": relief_emi,
        "lien_amount": lien_amount, "relief_months": relief_months,
        "acr": acr, "monthly_savings": monthly_savings,
    }
    explainability = generate_explainability_text("asset_backed", asdict(customer), results_dict, config)
    short = generate_short_explanation("asset_backed", results_dict)

    policy = enforce_all_policies(
        "asset_backed", recovery, relief_emi, customer.monthly_income,
        new_tenure, config, acr=acr
    )

    audit = create_audit_record(
        simulation_id=str(uuid.uuid4())[:8],
        customer_input={"customer_id": customer.customer_id},
        config=config, model_version=config.get("model_version", "2.0.0"),
        pathway_name="asset_backed", results=results_dict,
        policy_checks=policy.checks, explainability=explainability,
    )

    return PathwayResult(
        pathway_name="asset_backed",
        display_name="Asset-Backed Liquidity",
        description=f"Lien ₹{lien_amount:,.0f} on assets, {relief_months}-month relief",
        action=f"₹{relief_emi:,.0f}/mo for {relief_months} months, then ₹{customer.emi:,.0f}",
        npv=npv, recovery_rate=recovery, acceptance_prob=acceptance,
        churn_reduction=churn_red, composite_score=composite,
        new_emi=relief_emi, new_tenure_months=new_tenure,
        total_interest=total_paid - customer.principal,
        immediate_relief=f"₹{monthly_savings:,.0f}/mo saved for {relief_months} months",
        total_interest_change=interest_change,
        monthly_savings=monthly_savings,
        policy_result=asdict(policy),
        audit=audit.to_dict(),
        explainability=explainability,
        short_explanation=short,
        details={
            "lien_amount": lien_amount, "acr": acr, "acr_min": acr_min,
            "qualifies": acr >= acr_min, "release_condition": f"{release_payments} on-time payments",
        },
    )


# ─────────────────────────────────────────────
# PATHWAY 5: DEBT CONSOLIDATION
# ─────────────────────────────────────────────

def pathway_consolidation(customer: CustomerProfile,
                          config: Dict) -> PathwayResult:
    """
    Debt Consolidation: merge all debts into single loan at lower rate.

    NewPrincipal = primary + sum(other_debts)
    R_new < weighted_average_rate
    Waterfall: repay high-cost debts first.
    """
    discount_rate = config.get("discount_rate", 0.08)
    weights = config.get("weights")
    cons_config = config.get("consolidation", {})
    rate_discount = cons_config.get("rate_discount", 0.02)
    min_rate = cons_config.get("min_rate", 0.08)
    max_tenure = cons_config.get("max_tenure_months", 60)

    # Aggregate debts
    all_debts = [{"principal": customer.principal, "rate": customer.annual_rate, "type": "primary"}]
    for d in customer.other_debts:
        all_debts.append({
            "principal": d.get("principal", 0),
            "rate": d.get("rate", 0.20),
            "type": d.get("type", "other"),
        })

    num_debts = len(all_debts)
    total_principal = sum(d["principal"] for d in all_debts)
    old_war = weighted_average_rate(all_debts)

    # New rate
    new_rate = max(old_war - rate_discount, min_rate)

    # New tenure (keep same or extend slightly)
    new_tenure = min(max(customer.remaining_months, 36), max_tenure)

    # New EMI
    new_emi = compute_emi(total_principal, new_rate, new_tenure)

    # Old combined EMI (approximate)
    old_total_emi = customer.emi
    for d in customer.other_debts:
        # Estimate EMI for each other debt (assume 12 months remaining if unknown)
        d_months = d.get("remaining_months", 12)
        d_emi = compute_emi(d["principal"], d.get("rate", 0.20), d_months)
        old_total_emi += d_emi

    # Waterfall order
    waterfall = waterfall_repayment_order(all_debts)

    # Cashflows
    cashflows = [new_emi] * new_tenure

    # Default probs (consolidated = lower stress)
    post_dicr = compute_dicr(customer.monthly_income, customer.essential_expenses, new_emi)
    acr = compute_acr(customer.total_liquid_assets, total_principal)
    base_p = compute_default_prob(post_dicr, acr, customer.income_sigma, 0.0,
                                  config.get("default_coefficients"))
    probs = default_prob_series(base_p, new_tenure, 0.97, 0.30)

    npv = compute_npv(cashflows, probs, discount_rate)
    recovery = compute_recovery_rate(npv, total_principal)

    # Acceptance & churn
    relief = max(0, (old_total_emi - new_emi) / old_total_emi) if old_total_emi > 0 else 0
    stress = compute_stress_score(old_total_emi, customer.monthly_income,
                                   customer.total_liquid_assets)
    acceptance = estimate_acceptance(relief, stress, "consolidation")
    churn_red = estimate_churn_reduction("consolidation", relief)

    composite = compute_composite_score(acceptance, recovery, 1.0 - churn_red, weights)

    total_paid = new_emi * new_tenure
    monthly_savings = old_total_emi - new_emi
    total_interest = total_paid - total_principal
    total_interest_saved = (old_total_emi * customer.remaining_months - customer.principal) - total_interest

    results_dict = {
        "npv": npv, "recovery_rate": recovery, "acceptance_prob": acceptance,
        "composite": composite, "new_emi": new_emi,
        "monthly_savings": monthly_savings, "total_interest": total_interest,
        "old_weighted_rate": old_war, "new_rate": new_rate,
        "num_debts": num_debts, "post_dicr": post_dicr,
    }
    explainability = generate_explainability_text("consolidation", asdict(customer), results_dict, config)
    short = generate_short_explanation("consolidation", results_dict)

    policy = enforce_all_policies(
        "consolidation", recovery, new_emi, customer.monthly_income,
        new_tenure, config, num_debts=num_debts
    )

    audit = create_audit_record(
        simulation_id=str(uuid.uuid4())[:8],
        customer_input={"customer_id": customer.customer_id},
        config=config, model_version=config.get("model_version", "2.0.0"),
        pathway_name="consolidation", results=results_dict,
        policy_checks=policy.checks, explainability=explainability,
    )

    return PathwayResult(
        pathway_name="consolidation",
        display_name="Debt Consolidation",
        description=f"Merge {num_debts} debts at {new_rate*100:.1f}% (was {old_war*100:.1f}%)",
        action=f"New EMI: ₹{new_emi:,.0f}/mo (saves ₹{monthly_savings:,.0f}/mo)",
        npv=npv, recovery_rate=recovery, acceptance_prob=acceptance,
        churn_reduction=churn_red, composite_score=composite,
        new_emi=new_emi, new_tenure_months=new_tenure,
        total_interest=total_interest,
        immediate_relief=f"₹{monthly_savings:,.0f}/mo ongoing savings",
        total_interest_change=-total_interest_saved,
        monthly_savings=monthly_savings,
        policy_result=asdict(policy),
        audit=audit.to_dict(),
        explainability=explainability,
        short_explanation=short,
        details={
            "total_principal": total_principal, "old_war": old_war,
            "new_rate": new_rate, "waterfall": [d["type"] for d in waterfall],
            "total_interest_saved": total_interest_saved, "num_debts": num_debts,
            "post_dicr": post_dicr,
        },
    )


# ─────────────────────────────────────────────
# OPTIMIZER: SIMULATE ALL & RANK
# ─────────────────────────────────────────────

def simulate_all_pathways(customer: CustomerProfile,
                          config: Optional[Dict] = None,
                          pathways_to_run: Optional[List[str]] = None) -> SimulationResult:
    """
    Run all (or selected) pathways, rank by composite score,
    and return the recommended pathway.

    Args:
        customer:         CustomerProfile data
        config:           Engine config dict (loaded from file if None)
        pathways_to_run:  List of pathway names to run (default: all 5)

    Returns:
        SimulationResult with ranked results and recommendation
    """
    if config is None:
        config = load_engine_config()

    all_names = ["emi_holiday", "graduated_emi", "icr", "asset_backed", "consolidation"]
    if pathways_to_run is None:
        pathways_to_run = all_names

    results: List[PathwayResult] = []

    pathway_funcs = {
        "emi_holiday": lambda: pathway_emi_holiday(customer, config),
        "graduated_emi": lambda: pathway_graduated_emi(customer, config),
        "icr": lambda: pathway_icr(customer, config),
        "asset_backed": lambda: pathway_asset_backed(customer, config),
        "consolidation": lambda: pathway_consolidation(customer, config),
    }

    for name in pathways_to_run:
        if name in pathway_funcs:
            try:
                result = pathway_funcs[name]()
                results.append(result)
            except Exception as e:
                # Log but don't fail the whole simulation
                results.append(PathwayResult(
                    pathway_name=name,
                    display_name=name.replace("_", " ").title(),
                    description=f"Error: {str(e)}",
                    action="Unavailable",
                    explainability=f"Simulation failed: {str(e)}",
                ))

    # Sort by composite score (descending)
    results.sort(key=lambda r: r.composite_score, reverse=True)

    # Recommend: highest composite that passes policy
    recommended = None
    policy_summary = {}
    for r in results:
        if r.policy_result and r.policy_result.get("passed", False):
            if recommended is None:
                recommended = r.pathway_name
        policy_summary[r.pathway_name] = r.policy_result.get("passed", False) if r.policy_result else True

    # Fallback: if none pass policy, still recommend highest composite
    if recommended is None and results:
        recommended = results[0].pathway_name

    return SimulationResult(
        customer_id=customer.customer_id,
        timestamp=datetime.now().isoformat(),
        results=results,
        recommended=recommended,
        policy_checks=policy_summary,
        config_used=config,
    )


# ─────────────────────────────────────────────
# BACKWARD COMPATIBILITY
# ─────────────────────────────────────────────

# Keep LoanDetails and create_loan_from_customer for dashboard integration
from dataclasses import dataclass as _dc

@_dc
class LoanDetails:
    """Legacy compatibility — maps to CustomerProfile fields."""
    outstanding_principal: float
    emi_amount: float
    interest_rate: float   # Annual % (e.g., 14.5)
    remaining_months: int
    monthly_income: float
    current_savings: float
    payment_history_score: float


def create_loan_from_customer(customer_data: Dict) -> LoanDetails:
    """Legacy helper: convert dict to LoanDetails."""
    return LoanDetails(
        outstanding_principal=customer_data.get('outstanding_principal', 500000),
        emi_amount=customer_data.get('emi_amount', 18500),
        interest_rate=customer_data.get('interest_rate', 14.5),
        remaining_months=customer_data.get('remaining_months', 24),
        monthly_income=customer_data.get('monthly_income', 85000),
        current_savings=customer_data.get('current_savings', 42000),
        payment_history_score=customer_data.get('payment_history_score', 0.90),
    )


def create_customer_from_loan(loan: LoanDetails,
                              customer_id: str = "C00000",
                              essential_expenses: Optional[float] = None) -> CustomerProfile:
    """Convert legacy LoanDetails to CustomerProfile."""
    if essential_expenses is None:
        essential_expenses = loan.monthly_income * 0.55  # estimate

    # Normalize rate: if > 1, treat as percentage
    rate = loan.interest_rate
    if rate > 1:
        rate = rate / 100.0

    return CustomerProfile(
        customer_id=customer_id,
        monthly_income=loan.monthly_income,
        essential_expenses=essential_expenses,
        principal=loan.outstanding_principal,
        annual_rate=rate,
        remaining_months=loan.remaining_months,
        emi=loan.emi_amount,
        total_liquid_assets=loan.current_savings,
        payment_history_score=loan.payment_history_score,
    )


class RecoveryPathwayEngine:
    """
    Legacy-compatible wrapper around the new pathway simulator.
    Used by dashboard.py — provides the same API as the old engine.
    """

    def __init__(self, discount_rate: float = 0.08, cost_of_capital: float = 0.08):
        self.config = load_engine_config()
        self.config["discount_rate"] = discount_rate

    def generate_all_pathways(self, loan: LoanDetails):
        """
        Generate all 5 pathways and return in legacy format:
        List of (pathway_details_dict, PathwayMetrics-like object)
        """
        customer = create_customer_from_loan(loan)
        sim = simulate_all_pathways(customer, self.config)

        # Convert to legacy format
        results = []
        for r in sim.results:
            details = {
                'name': r.display_name,
                'description': r.description,
                'action': r.action,
                'new_emi': r.new_emi,
                'new_tenure_months': r.new_tenure_months,
                'immediate_relief': r.immediate_relief,
                'total_interest_increase': r.total_interest_change,
                # New fields for enhanced dashboard
                'pathway_name': r.pathway_name,
                'explainability': r.explainability,
                'short_explanation': r.short_explanation,
                'mc_result': r.mc_result,
                'policy_result': r.policy_result,
                'audit': r.audit,
                'details': r.details,
            }

            # Create a metrics-like object using a simple namespace
            class _Metrics:
                pass
            m = _Metrics()
            m.acceptance_probability = r.acceptance_prob
            m.npv_recovery_rate = r.recovery_rate
            m.churn_reduction = r.churn_reduction
            m.composite_score = r.composite_score
            m.monthly_payment = r.new_emi
            m.total_interest = r.total_interest
            m.new_tenure_months = r.new_tenure_months

            results.append((details, m))

        return results

    def estimate_default_probability(self, loan, months_ahead, pathway_stress_reduction=0.0):
        """Legacy compatibility for cash flow forecast."""
        rate = loan.interest_rate
        if rate > 1:
            rate = rate / 100.0
        emi_ratio = loan.emi_amount / loan.monthly_income if loan.monthly_income > 0 else 1
        savings_months = loan.current_savings / loan.emi_amount if loan.emi_amount > 0 else 6
        base_hazard = 0.01 + 0.15 * emi_ratio + 0.05 * max(0, (1 - savings_months / 6))
        base_hazard *= (1 - loan.payment_history_score * 0.5)
        adjusted = base_hazard * (1 - pathway_stress_reduction)
        survival = (1 - adjusted) ** months_ahead
        return min(1 - survival, 0.95)


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("RECOVERY PATH ENGINE v2.0 — PATHWAY SIMULATOR TEST")
    print("=" * 80)

    # Prompt example customer
    customer = CustomerProfile(
        customer_id="C12345",
        name="Test Customer",
        monthly_income=85000,
        essential_expenses=50000,
        principal=500000,
        annual_rate=0.14,
        remaining_months=24,
        total_liquid_assets=450000,
        other_debts=[
            {"type": "cc", "principal": 120000, "rate": 0.42},
            {"type": "personal", "principal": 80000, "rate": 0.18},
        ],
        cibil_score=680,
    )

    print(f"\nCustomer: {customer.customer_id}")
    print(f"  Income: ₹{customer.monthly_income:,}, Expenses: ₹{customer.essential_expenses:,}")
    print(f"  Principal: ₹{customer.principal:,}, Rate: {customer.annual_rate*100:.1f}%")
    print(f"  EMI: ₹{customer.emi:,.0f}, Remaining: {customer.remaining_months} months")
    dicr = compute_dicr(customer.monthly_income, customer.essential_expenses, customer.emi)
    print(f"  DICR: {dicr:.2f}")

    sim = simulate_all_pathways(customer)

    print(f"\n{'='*80}")
    print(f"RANKED PATHWAYS (Recommended: {sim.recommended})")
    print(f"{'='*80}")

    for i, r in enumerate(sim.results, 1):
        badge = "⭐" if r.pathway_name == sim.recommended else "  "
        print(f"\n{badge} #{i}: {r.display_name} (Composite: {r.composite_score:.3f})")
        print(f"   {r.description}")
        print(f"   NPV: ₹{r.npv:,.0f} | Recovery: {r.recovery_rate:.1%}")
        print(f"   Acceptance: {r.acceptance_prob:.1%} | Churn Reduction: {r.churn_reduction:.1%}")
        print(f"   New EMI: ₹{r.new_emi:,.0f} | Tenure: {r.new_tenure_months} mo")
        policy_ok = r.policy_result.get("passed", "N/A") if r.policy_result else "N/A"
        print(f"   Policy: {'✅ PASS' if policy_ok else '❌ FAIL'}")

    print(f"\n{'='*80}")
    print("✅ Pathway Simulator test complete!")

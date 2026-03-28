"""
Scoring Service — Default Risk & Acceptance Models
Recovery Path Engine v2.0

Implements:
- Logistic default probability model (sigmoid)
- Time-series default probability with recovery decay
- Acceptance probability estimation
- Churn risk estimation

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import math
from typing import Dict, List, Optional


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def compute_default_prob(dicr: float,
                         acr: float,
                         income_volatility: float = 0.0,
                         macro_shock: float = 0.0,
                         coefficients: Optional[Dict[str, float]] = None) -> float:
    """
    Logistic default probability model.
    """
    if coefficients is None:
        coefficients = {"a0": -2.0, "a1": -1.5, "a2": -0.8, "a3": 0.5, "a4": 0.3}

    try:
        # Check for NaN in primary inputs
        if any(math.isnan(x) for x in [dicr, acr, income_volatility, macro_shock]):
            return 0.5 # Neutral fallback

        z = (coefficients.get("a0", -2.0)
             + coefficients.get("a1", -1.5) * dicr
             + coefficients.get("a2", -0.8) * acr
             + coefficients.get("a3", 0.5) * income_volatility
             + coefficients.get("a4", 0.3) * macro_shock)

        if math.isnan(z): return 0.5
        return sigmoid(z)
    except:
        return 0.5


def default_prob_series(base_prob: float,
                        months: int,
                        decay_factor: float = 0.95,
                        pathway_reduction: float = 0.0) -> List[float]:
    """
    Generate a time-series of default probabilities after pathway intervention.

    p_t = p_base × (1 - pathway_reduction) × decay_factor^t

    Args:
        base_prob:           Baseline default probability
        months:              Number of months
        decay_factor:        Monthly decay after recovery (< 1.0 means risk reduces)
        pathway_reduction:   Immediate reduction from pathway (0 to 1)

    Returns:
        List of monthly default probabilities
    """
    if math.isnan(base_prob): base_prob = 0.5
    adjusted_base = base_prob * (1.0 - pathway_reduction)
    probs = []
    for t in range(months):
        p_t = adjusted_base * (decay_factor ** t)
        if math.isnan(p_t): p_t = 0.5
        probs.append(min(max(p_t, 0.001), 0.95))
    return probs


def estimate_acceptance(emi_relief_pct: float,
                        stress_score: float,
                        pathway_type: str) -> float:
    """
    Logistic acceptance probability model.
    P(accept) = sigmoid(β0 + β1×stress + β2×relief + β3×pathway)

    Args:
        emi_relief_pct:  Fraction of EMI reduction (0 to 1)
        stress_score:    Financial stress indicator (0 to 1)
        pathway_type:    One of: emi_holiday, graduated_emi, icr,
                         asset_backed, consolidation

    Returns:
        Acceptance probability [0.10, 0.95]
    """
    beta_0 = -2.0
    beta_stress = 3.0
    beta_relief = 4.0

    pathway_betas = {
        "emi_holiday": 0.5,        # Immediate relief → popular
        "graduated_emi": 0.3,      # Phased approach
        "icr": -0.2,               # Complex / income-linked
        "asset_backed": -0.3,      # Requires collateral
        "consolidation": -0.1,     # Paperwork but big benefit
    }
    beta_pathway = pathway_betas.get(pathway_type, 0.0)

    if math.isnan(emi_relief_pct) or math.isnan(stress_score):
        return 0.5

    z = beta_0 + beta_stress * stress_score + beta_relief * emi_relief_pct + beta_pathway
    if math.isnan(z): return 0.5
    prob = sigmoid(z)
    return min(max(prob, 0.10), 0.95)


def estimate_churn_reduction(pathway_type: str,
                             stress_reduction: float) -> float:
    """
    Estimate churn risk reduction from pathway intervention.

    Higher values (closer to 1) mean the pathway does more to retain the customer.

    Args:
        pathway_type:     Pathway identifier
        stress_reduction: How much financial stress is reduced (0 to 1)

    Returns:
        Churn reduction score [0, 0.60]
    """
    base_churn_risk = 0.40  # 40% churn if no intervention

    pathway_factors = {
        "emi_holiday": 0.35,
        "graduated_emi": 0.45,
        "icr": 0.40,
        "asset_backed": 0.30,
        "consolidation": 0.50,
    }

    base_reduction = pathway_factors.get(pathway_type, 0.30)
    total_reduction = base_reduction * (0.7 + 0.3 * stress_reduction)
    churn_reduction = base_churn_risk * total_reduction

    return min(churn_reduction, 0.60)


def compute_stress_score(emi: float, income: float, savings: float) -> float:
    """
    Compute a 0-1 financial stress score from basic indicators.

    Args:
        emi:      Monthly EMI obligation
        income:   Monthly income
        savings:  Current savings balance

    Returns:
        Stress score [0, 1]  (higher = more stressed)
    """
    try:
        if math.isnan(emi) or math.isnan(income) or math.isnan(savings):
            return 0.5
        emi_ratio = emi / income if income > 0 else 1.0
        savings_months = savings / emi if emi > 0 else 6.0
        stress = min(1.0, emi_ratio + max(0, (1.0 - savings_months / 3.0)) * 0.5)
        return stress if not math.isnan(stress) else 0.5
    except:
        return 0.5

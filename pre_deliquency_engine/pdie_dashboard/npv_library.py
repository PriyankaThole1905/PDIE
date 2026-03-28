"""
NPV Library — Core Financial Formulae
Recovery Path Engine v2.0

Pure mathematical functions for:
- DICR (Disposable Income Coverage Ratio)
- ACR (Asset Coverage Ratio)
- Amortization schedules
- Present Value / Net Present Value with default risk
- Capitalized interest
- Composite scoring

All monetary values in INR. Rates as decimals (0.14 = 14% p.a.).

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import math
from typing import List, Dict, Tuple, Optional


# ───────────────────────────────────────────
# COVERAGE RATIOS
# ───────────────────────────────────────────

def compute_dicr(monthly_income: float, essential_expenses: float, emi: float) -> float:
    """
    Disposable Income Coverage Ratio.
    DICR = (Monthly_Income - Essential_Monthly_Expenses) / EMI

    Returns:
        DICR ratio (>1 means disposable income covers EMI)
    """
    try:
        if emi <= 0 or math.isnan(emi) or math.isinf(emi):
            return 10.0 # High number = safe
        income = monthly_income if not math.isnan(monthly_income) else 0.0
        expenses = essential_expenses if not math.isnan(essential_expenses) else 0.0
        return (income - expenses) / emi
    except:
        return 0.0


def compute_acr(total_liquid_assets: float, outstanding_principal: float) -> float:
    """
    Asset Coverage Ratio.
    ACR = Total_Liquid_Assets / Outstanding_Principal

    Returns:
        ACR ratio (>1 means assets cover the full principal)
    """
    if outstanding_principal <= 0:
        return float('inf')
    return total_liquid_assets / outstanding_principal


# ───────────────────────────────────────────
# RATE CONVERSIONS
# ───────────────────────────────────────────

def annual_to_monthly_rate(annual_rate: float) -> float:
    """Convert annual interest rate to monthly rate.  i_month = i / 12"""
    return annual_rate / 12.0


def monthly_discount_factor(annual_discount_rate: float) -> float:
    """Monthly discount factor:  d = (1 + r)^(1/12)"""
    return (1.0 + annual_discount_rate) ** (1.0 / 12.0)


# ───────────────────────────────────────────
# AMORTIZATION
# ───────────────────────────────────────────

def compute_emi(principal: float, annual_rate: float, months: int) -> float:
    """
    Standard amortization EMI.
    EMI = P × i_m × (1+i_m)^N / ((1+i_m)^N - 1)
    """
    if months <= 0 or principal <= 0 or math.isnan(principal) or math.isnan(annual_rate):
        return 0.0
    i_m = annual_rate / 12.0
    if i_m <= 0:
        return principal / months
    try:
        factor = (1.0 + i_m) ** months
        emi = principal * i_m * factor / (factor - 1.0)
        return emi if not math.isnan(emi) and not math.isinf(emi) else principal/months
    except:
        return principal / months


def compute_amortization(principal: float, annual_rate: float, months: int) -> Tuple[float, List[Dict]]:
    """
    Full amortization schedule.

    Returns:
        (emi, schedule) where schedule is list of dicts with keys:
        month, emi, interest, principal_paid, balance
    """
    emi = compute_emi(principal, annual_rate, months)
    i_m = annual_rate / 12.0
    schedule = []
    balance = principal

    for t in range(1, months + 1):
        interest = balance * i_m
        if math.isnan(interest): interest = 0.0
        principal_paid = emi - interest
        if math.isnan(principal_paid): principal_paid = 0.0
        balance -= principal_paid
        # Clamp rounding errors on last payment
        if t == months:
            principal_paid += balance
            balance = 0.0
        schedule.append({
            "month": t,
            "emi": round(float(emi), 2) if not math.isnan(emi) else 0.0,
            "interest": round(float(interest), 2),
            "principal_paid": round(float(principal_paid), 2),
            "balance": round(max(float(balance), 0.0), 2),
        })

    return round(emi, 2), schedule


# ───────────────────────────────────────────
# PRESENT VALUE / NPV
# ───────────────────────────────────────────

def compute_pv(cashflows: List[float], annual_discount_rate: float) -> float:
    """
    Present Value of monthly cashflows.
    PV = Σ CF_t / d^t   where d = (1 + r)^(1/12)
    """
    d = monthly_discount_factor(annual_discount_rate)
    pv = 0.0
    for t, cf in enumerate(cashflows, start=1):
        pv += cf / (d ** t)
    return pv


def compute_npv(cashflows: List[float],
                default_probs: List[float],
                annual_discount_rate: float) -> float:
    """
    Risk-adjusted Net Present Value.
    NPV = Σ (CF_t × (1 - p_t)) / d^t

    Args:
        cashflows:       Monthly expected cashflows
        default_probs:   Monthly default probability (0-1)
        annual_discount_rate: Bank cost of capital (e.g. 0.08)

    Returns:
        Risk-adjusted NPV
    """
    d = monthly_discount_factor(annual_discount_rate)
    npv = 0.0
    n = min(len(cashflows), len(default_probs))
    for t in range(n):
        npv += cashflows[t] * (1.0 - default_probs[t]) / (d ** (t + 1))
    return npv


def compute_recovery_rate(npv: float, outstanding_principal: float) -> float:
    """Recovery_Rate = Expected_NPV / Outstanding_Principal"""
    if outstanding_principal <= 0:
        return 0.0
    return npv / outstanding_principal


# ───────────────────────────────────────────
# CAPITALIZED INTEREST (EMI HOLIDAY)
# ───────────────────────────────────────────

def compute_capitalized_interest(emi: float, monthly_rate: float, holiday_months: int) -> float:
    """
    Exact compound capitalized interest during EMI holiday.
    CapitalizedInterest = EMI × [(1 + i_m)^M_h - 1] / i_m

    This represents the interest that accrues on the missed EMI payments.
    """
    if monthly_rate <= 0 or holiday_months <= 0:
        return 0.0
    return emi * ((1.0 + monthly_rate) ** holiday_months - 1.0) / monthly_rate


# ───────────────────────────────────────────
# COMPOSITE SCORING
# ───────────────────────────────────────────

def compute_composite_score(acceptance_prob: float,
                            recovery_rate: float,
                            churn_rate: float,
                            weights: Optional[Dict[str, float]] = None) -> float:
    """
    Weighted Composite Score for pathway ranking.
    Composite = w_accept × Acceptance_Prob + w_npv × Recovery_Rate + w_churn × (1 - Churn_Rate)

    Args:
        acceptance_prob:  P(customer accepts)  [0,1]
        recovery_rate:    NPV / Principal       [0,~1.x]
        churn_rate:       P(customer churns)    [0,1]
        weights:          {"accept": 0.4, "npv": 0.4, "churn": 0.2}

    Returns:
        Composite score (higher is better)
    """
    if weights is None:
        weights = {"accept": 0.40, "npv": 0.40, "churn": 0.20}

    w_a = weights.get("accept", 0.40)
    w_n = weights.get("npv", 0.40)
    w_c = weights.get("churn", 0.20)

    return w_a * acceptance_prob + w_n * recovery_rate + w_c * (1.0 - churn_rate)


# ───────────────────────────────────────────
# DEBT CONSOLIDATION HELPERS
# ───────────────────────────────────────────

def weighted_average_rate(debts: List[Dict]) -> float:
    """
    Compute weighted average interest rate across multiple debts.
    Each debt dict has keys: principal, rate
    """
    total_principal = sum(d["principal"] for d in debts)
    if total_principal <= 0:
        return 0.0
    return sum(d["principal"] * d["rate"] for d in debts) / total_principal


def waterfall_repayment_order(debts: List[Dict]) -> List[Dict]:
    """
    Sort debts by rate descending (highest-cost first) for waterfall repayment.
    """
    return sorted(debts, key=lambda d: d.get("rate", 0), reverse=True)


# ───────────────────────────────────────────
# TENURE SOLVER
# ───────────────────────────────────────────

def solve_tenure_for_emi(principal: float, annual_rate: float, target_emi: float) -> int:
    """
    Solve for number of months N given target EMI.
    N = -ln(1 - P×i_m/EMI) / ln(1+i_m)

    Returns integer months (ceiling).
    """
    i_m = annual_rate / 12.0
    if i_m <= 0:
        return int(math.ceil(principal / target_emi)) if target_emi > 0 else 1

    ratio = principal * i_m / target_emi
    if ratio >= 1.0:
        # EMI too low to ever repay — return a very long tenure
        return 360  # 30 years cap
    n = -math.log(1.0 - ratio) / math.log(1.0 + i_m)
    return int(math.ceil(n))

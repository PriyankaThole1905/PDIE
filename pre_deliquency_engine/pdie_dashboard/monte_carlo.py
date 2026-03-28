"""
Monte Carlo Engine — Income-Contingent Repayment Simulation
Recovery Path Engine v2.0

Implements:
- AR(1) income path simulation with auto-correlation
- Monte Carlo NPV estimation for ICR pathway
- Distribution statistics (mean, std, quantiles)
- Reproducible results via configurable random seed

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from npv_library import (
    monthly_discount_factor,
    annual_to_monthly_rate,
    compute_npv,
)


@dataclass
class MCResult:
    """Monte Carlo simulation result."""
    mean_npv: float
    std_npv: float
    p5: float       # 5th percentile
    p10: float      # 10th percentile
    p50: float      # median
    p90: float      # 90th percentile
    p95: float      # 95th percentile
    prob_above_threshold: float   # P(recovery_rate > threshold)
    n_runs: int
    seed: int
    all_npvs: Optional[np.ndarray] = None   # raw array for plotting


def simulate_income_path(mean_income: float,
                         sigma: float,
                         months: int,
                         rho: float = 0.7,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Simulate a monthly income path using AR(1) process with log-normal shocks.

    income_t = mean × exp(z_t)
    z_t = rho × z_{t-1} + epsilon_t
    epsilon_t ~ N(0, sigma^2 × (1 - rho^2))

    Args:
        mean_income: Average monthly income
        sigma:       Standard deviation of log-income (e.g., 0.15 = 15% volatility)
        months:      Number of months to simulate
        rho:         Auto-correlation coefficient (0 = iid, 1 = random walk)
        rng:         Numpy random generator (for reproducibility)

    Returns:
        Array of simulated monthly incomes (length = months)
    """
    if rng is None:
        rng = np.random.default_rng()

    innovation_sigma = sigma * np.sqrt(max(1.0 - rho ** 2, 0.01))
    z = np.zeros(months)
    z[0] = rng.normal(0, sigma)

    for t in range(1, months):
        z[t] = rho * z[t - 1] + rng.normal(0, innovation_sigma)

    # Convert to income levels (log-normal ensures positivity)
    incomes = mean_income * np.exp(z - 0.5 * sigma ** 2)  # bias correction
    return np.maximum(incomes, 0)


def compute_icr_cashflows(income_path: np.ndarray,
                          phi: float = 0.22,
                          emi_min: float = 10000.0) -> np.ndarray:
    """
    Compute ICR EMI path: EMI_t = max(EMI_min, floor(phi × income_t))

    Args:
        income_path: Simulated monthly incomes
        phi:         Fraction of income for repayment (default 22%)
        emi_min:     Minimum EMI floor

    Returns:
        Array of monthly EMI payments
    """
    emi_path = np.maximum(emi_min, np.floor(phi * income_path))
    return emi_path


def compute_default_probs_from_dicr(income_path: np.ndarray,
                                     emi_path: np.ndarray,
                                     essential_expenses: float,
                                     base_default_prob: float = 0.03,
                                     decay_factor: float = 0.95) -> np.ndarray:
    """
    Compute month-by-month default probability based on DICR.

    When DICR < 1, default risk rises sharply.
    After recovery pathway starts, apply a decay factor.

    Args:
        income_path:         Monthly income array
        emi_path:            Monthly EMI array
        essential_expenses:  Fixed monthly expenses
        base_default_prob:   Baseline default probability
        decay_factor:        Monthly decay after recovery starts

    Returns:
        Array of default probabilities per month
    """
    months = len(income_path)
    p_t = np.zeros(months)

    for t in range(months):
        disposable = income_path[t] - essential_expenses
        if emi_path[t] > 0:
            dicr = disposable / emi_path[t]
        else:
            dicr = 10.0  # no EMI = no default risk

        # Logistic mapping: low DICR → high default prob
        if dicr >= 2.0:
            p_month = base_default_prob * 0.5
        elif dicr >= 1.5:
            p_month = base_default_prob
        elif dicr >= 1.0:
            p_month = base_default_prob * 2.0
        elif dicr >= 0.5:
            p_month = base_default_prob * 5.0
        else:
            p_month = min(0.30, base_default_prob * 10.0)

        # Apply recovery decay
        p_t[t] = p_month * (decay_factor ** t)

    return np.clip(p_t, 0.0, 0.95)


def mc_icr_npv(income_mean: float,
               income_sigma: float,
               essential_expenses: float,
               outstanding_principal: float,
               annual_discount_rate: float = 0.08,
               phi: float = 0.22,
               emi_min: float = 10000.0,
               rho: float = 0.7,
               base_default_prob: float = 0.03,
               months: int = 60,
               n_runs: int = 10000,
               recovery_threshold: float = 0.70,
               seed: int = 42,
               return_all: bool = False) -> MCResult:
    """
    Monte Carlo simulation for Income-Contingent Repayment NPV.

    For each run:
    1. Simulate income path (AR(1) with log-normal)
    2. Compute ICR EMI path: EMI_t = max(emi_min, floor(phi × income_t))
    3. Compute default probability series from DICR
    4. Compute risk-adjusted NPV

    Args:
        income_mean:           Average monthly income
        income_sigma:          Income volatility (std of log-income)
        essential_expenses:    Fixed monthly expenses
        outstanding_principal: Current loan principal
        annual_discount_rate:  Bank cost of capital
        phi:                   Income fraction for EMI
        emi_min:              Minimum EMI floor
        rho:                   Income auto-correlation
        base_default_prob:     Baseline default probability
        months:                Simulation horizon
        n_runs:                Number of MC simulations
        recovery_threshold:    Threshold for P(recovery > X) calculation
        seed:                  Random seed for reproducibility
        return_all:            If True, include all NPV values in result

    Returns:
        MCResult with statistics
    """
    rng = np.random.default_rng(seed)
    npvs = np.zeros(n_runs)

    for run in range(n_runs):
        # 1. Simulate income
        income_path = simulate_income_path(income_mean, income_sigma, months, rho, rng)

        # 2. Compute ICR EMIs
        emi_path = compute_icr_cashflows(income_path, phi, emi_min)

        # 3. Compute default probabilities
        p_t = compute_default_probs_from_dicr(
            income_path, emi_path, essential_expenses, base_default_prob
        )

        # 4. Compute NPV
        npvs[run] = compute_npv(
            emi_path.tolist(),
            p_t.tolist(),
            annual_discount_rate
        )

    # Compute recovery rates
    recovery_rates = npvs / outstanding_principal if outstanding_principal > 0 else npvs

    result = MCResult(
        mean_npv=float(np.mean(npvs)),
        std_npv=float(np.std(npvs)),
        p5=float(np.percentile(npvs, 5)),
        p10=float(np.percentile(npvs, 10)),
        p50=float(np.percentile(npvs, 50)),
        p90=float(np.percentile(npvs, 90)),
        p95=float(np.percentile(npvs, 95)),
        prob_above_threshold=float(np.mean(recovery_rates > recovery_threshold)),
        n_runs=n_runs,
        seed=seed,
        all_npvs=npvs if return_all else None,
    )

    return result


if __name__ == "__main__":
    print("Testing Monte Carlo Engine...\n")

    result = mc_icr_npv(
        income_mean=85000,
        income_sigma=0.15,
        essential_expenses=50000,
        outstanding_principal=500000,
        n_runs=5000,
        months=24,
        seed=42,
    )

    print(f"MC ICR NPV Results ({result.n_runs} runs, seed={result.seed}):")
    print(f"  Mean NPV:       ₹{result.mean_npv:,.0f}")
    print(f"  Std  NPV:       ₹{result.std_npv:,.0f}")
    print(f"  5th percentile: ₹{result.p5:,.0f}")
    print(f"  Median:         ₹{result.p50:,.0f}")
    print(f"  95th percentile:₹{result.p95:,.0f}")
    print(f"  P(recovery>70%): {result.prob_above_threshold*100:.1f}%")
    print("\n✅ Monte Carlo Engine test complete!")

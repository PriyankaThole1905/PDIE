"""
Policy Engine — Regulatory & Bank Policy Enforcement
Recovery Path Engine v2.0

Checks every pathway result against:
- Minimum recovery rate threshold
- EMI / income covenant
- Regulatory limits (max tenure, min EMI)
- ACR qualification for asset-backed pathway
- Consolidation eligibility

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PolicyCheckResult:
    """Result of all policy checks for a single pathway."""
    pathway_name: str
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)


def check_min_recovery(recovery_rate: float, threshold: float = 0.50) -> bool:
    """Recovery rate must exceed minimum threshold."""
    return recovery_rate >= threshold


def check_emi_income_covenant(emi: float, income: float,
                               max_ratio: float = 0.60) -> bool:
    """Post-pathway EMI must not exceed max_ratio of income."""
    if income <= 0:
        return False
    return (emi / income) <= max_ratio


def check_max_tenure(tenure_months: int, max_months: int = 360) -> bool:
    """Tenure cannot exceed regulatory maximum."""
    return tenure_months <= max_months


def check_min_emi(emi: float, min_emi: float = 1000.0) -> bool:
    """EMI cannot fall below a minimum floor."""
    return emi >= min_emi


def check_acr_qualification(acr: float, acr_min: float = 0.75) -> bool:
    """Asset-backed pathway requires minimum ACR."""
    return acr >= acr_min


def check_consolidation_eligibility(num_debts: int, min_debts: int = 2) -> bool:
    """Debt consolidation requires at least min_debts obligations."""
    return num_debts >= min_debts


def enforce_all_policies(pathway_name: str,
                         recovery_rate: float,
                         emi: float,
                         income: float,
                         tenure_months: int,
                         config: Optional[Dict] = None,
                         acr: Optional[float] = None,
                         num_debts: Optional[int] = None) -> PolicyCheckResult:
    """
    Run all applicable policy checks for a pathway.

    Args:
        pathway_name:   Pathway identifier
        recovery_rate:  Computed recovery rate
        emi:           Post-pathway EMI
        income:        Customer monthly income
        tenure_months: Post-pathway tenure
        config:        Engine configuration dict
        acr:           Asset Coverage Ratio (for asset_backed)
        num_debts:     Number of debts (for consolidation)

    Returns:
        PolicyCheckResult with pass/fail and violation details
    """
    if config is None:
        config = {}

    checks = {}
    violations = []

    # 1. Minimum recovery rate
    min_rr = config.get("min_recovery_rate", 0.50)
    ok = check_min_recovery(recovery_rate, min_rr)
    checks["min_recovery_rate"] = ok
    if not ok:
        violations.append(
            f"Recovery rate {recovery_rate:.2%} below minimum {min_rr:.0%}"
        )

    # 2. EMI / income covenant
    max_ratio = config.get("emi_income_max_ratio", 0.60)
    ok = check_emi_income_covenant(emi, income, max_ratio)
    checks["emi_income_covenant"] = ok
    if not ok:
        ratio = emi / income if income > 0 else float('inf')
        violations.append(
            f"EMI/Income ratio {ratio:.1%} exceeds maximum {max_ratio:.0%}"
        )

    # 3. Max tenure
    max_t = config.get("max_tenure_months", 360)
    ok = check_max_tenure(tenure_months, max_t)
    checks["max_tenure"] = ok
    if not ok:
        violations.append(
            f"Tenure {tenure_months} months exceeds maximum {max_t}"
        )

    # 4. Min EMI
    min_e = config.get("emi_min", 1000)
    ok = check_min_emi(emi, min_e)
    checks["min_emi"] = ok
    if not ok:
        violations.append(
            f"EMI ₹{emi:,.0f} below minimum ₹{min_e:,.0f}"
        )

    # 5. Pathway-specific checks
    if pathway_name == "asset_backed" and acr is not None:
        acr_min = config.get("acr_min", 0.75)
        ok = check_acr_qualification(acr, acr_min)
        checks["acr_qualification"] = ok
        if not ok:
            violations.append(
                f"ACR {acr:.2f} below minimum {acr_min:.2f} for asset-backed pathway"
            )

    if pathway_name == "consolidation" and num_debts is not None:
        ok = check_consolidation_eligibility(num_debts, 2)
        checks["consolidation_eligibility"] = ok
        if not ok:
            violations.append(
                f"Only {num_debts} debt(s); consolidation requires at least 2"
            )

    return PolicyCheckResult(
        pathway_name=pathway_name,
        passed=len(violations) == 0,
        checks=checks,
        violations=violations,
    )

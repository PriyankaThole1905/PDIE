"""
API Schemas — Pydantic Models for Recovery Path Engine API
Recovery Path Engine v2.0

Defines request/response models matching the JSON contracts.

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class LoanInput:
    principal: float = 500000
    annual_rate: float = 0.14
    remaining_months: int = 24
    emi: float = 0


@dataclass
class DebtInput:
    type: str = "cc"
    principal: float = 0
    rate: float = 0.20


@dataclass
class AssetInput:
    FD: float = 0
    MF: float = 0
    LIC_surrender: float = 0

    @property
    def total(self) -> float:
        return self.FD + self.MF + self.LIC_surrender


@dataclass
class IncomeHistoryEntry:
    date: str = ""
    amount: float = 0


@dataclass
class CustomerInput:
    """POST /api/v1/customer — Customer input schema."""
    customer_id: str = "C12345"
    name: str = ""
    monthly_income: float = 85000
    essential_expenses: float = 50000
    income_history: List[Dict] = field(default_factory=list)
    assets: Dict[str, float] = field(default_factory=dict)
    loan: Dict[str, Any] = field(default_factory=dict)
    other_debts: List[Dict] = field(default_factory=list)
    risk_band: str = "B2"
    cibil_score: int = 680

    def total_liquid_assets(self) -> float:
        return sum(self.assets.values())


@dataclass
class ScenarioWeight:
    name: str = "base"
    weight: float = 1.0


@dataclass
class SimulateRequest:
    """POST /api/v1/simulate — Simulation request."""
    customer_id: str = "C12345"
    pathways: List[str] = field(default_factory=lambda: [
        "emi_holiday", "graduated_emi", "icr", "asset_backed", "consolidation"
    ])
    scenario_weights: List[Dict] = field(default_factory=lambda: [
        {"name": "base", "weight": 0.6},
        {"name": "salary_delay", "weight": 0.3},
        {"name": "job_loss", "weight": 0.1},
    ])
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditSchema:
    """Audit object in API response."""
    simulation_id: str = ""
    timestamp: str = ""
    model_version: str = ""
    input_snapshot: Dict = field(default_factory=dict)
    parameters: Dict = field(default_factory=dict)
    random_seed: Optional[int] = None
    explainability: str = ""


@dataclass
class PathwayResultSchema:
    """Single pathway result in API response."""
    pathway: str = ""
    display_name: str = ""
    npv: float = 0
    recovery_rate: float = 0
    acceptance_prob: float = 0
    churn_reduction: float = 0
    composite: float = 0
    new_emi: float = 0
    new_tenure_months: int = 0
    total_interest: float = 0
    immediate_relief: str = ""
    monthly_savings: float = 0
    description: str = ""
    action: str = ""
    explainability: str = ""
    short_explanation: str = ""
    mc_result: Optional[Dict] = None
    policy_checks: Dict[str, bool] = field(default_factory=dict)
    audit: Dict = field(default_factory=dict)


@dataclass
class PolicyCheckSchema:
    min_recovery_ok: bool = True
    regulatory_limit_ok: bool = True
    emi_covenant_ok: bool = True


@dataclass
class SimulateResponse:
    """POST /api/v1/simulate — Simulation response."""
    customer_id: str = ""
    results: List[Dict] = field(default_factory=list)
    recommended: str = ""
    policy_checks: Dict[str, bool] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

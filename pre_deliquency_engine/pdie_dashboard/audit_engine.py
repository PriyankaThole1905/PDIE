"""
Audit Engine — Provenance, Timestamps & Explainability
Recovery Path Engine v2.0

Produces:
- Audit records with full input snapshots and parameter provenance
- Human-readable explainability text
- Short 1-line recommendations
- Relationship Manager email templates

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class AuditRecord:
    """Immutable audit record for a simulation run."""
    simulation_id: str
    timestamp: str
    model_version: str
    input_snapshot: Dict[str, Any]
    parameters: Dict[str, Any]
    random_seed: Optional[int]
    pathway_name: str
    results_summary: Dict[str, float]
    policy_checks: Dict[str, bool]
    explainability: str

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def create_audit_record(simulation_id: str,
                        customer_input: Dict,
                        config: Dict,
                        model_version: str,
                        pathway_name: str,
                        results: Dict[str, float],
                        policy_checks: Dict[str, bool],
                        explainability: str,
                        seed: Optional[int] = None) -> AuditRecord:
    """
    Create a complete audit record for a simulation.

    Args:
        simulation_id:  Unique ID for this simulation run
        customer_input: Sanitized snapshot of customer data
        config:         Engine configuration used
        model_version:  Version string of the model
        pathway_name:   Name of the pathway simulated
        results:        Key numeric results (NPV, recovery_rate, etc.)
        policy_checks:  Pass/fail for each policy check
        explainability: Human-readable explanation text
        seed:           Random seed (for Monte Carlo)

    Returns:
        AuditRecord
    """
    return AuditRecord(
        simulation_id=simulation_id,
        timestamp=datetime.now().isoformat(),
        model_version=model_version,
        input_snapshot=customer_input,
        parameters=config,
        random_seed=seed,
        pathway_name=pathway_name,
        results_summary=results,
        policy_checks=policy_checks,
        explainability=explainability,
    )


def _graduated_emi_text(config, new_emi, emi, dicr, recovery, acceptance):
    """Helper to generate graduated EMI explainability text (avoids f-string backslash issues)."""
    import math
    def _s(v, d=0):
        try:
            val = float(v)
            return d if math.isnan(val) or math.isinf(val) else val
        except: return d

    new_emi = _s(new_emi, 18500)
    emi = _s(emi, 18500)
    dicr = _s(dicr, 1.0)
    recovery = _s(recovery, 0.5)
    acceptance = _s(acceptance, 0.5)

    phases = config.get("graduated_phases", [])
    phase_strs = []
    for p in phases:
        r = _s(p.get("reduction", 0), 0)
        pct = int((1 - r) * 100)
        phase_strs.append(f"{pct}%")
    phases_text = " -> ".join(phase_strs) if phase_strs else "N/A"
    return (
        f"Graduated EMI applies phased reductions: "
        f"{phases_text} of original EMI. "
        f"Effective average EMI is INR {new_emi:,.0f}, vs original INR {emi:,.0f}. "
        f"This improves DICR from {dicr:.2f}x and reduces near-term default pressure. "
        f"Recovery rate: {recovery:.1%}, acceptance probability: {acceptance:.1%}."
    )


def generate_explainability_text(pathway_name: str,
                                  customer: Dict,
                                  results: Dict[str, float],
                                  config: Dict) -> str:
    """
    Generate human-readable explainability text for audit and compliance.

    Covers: what the pathway does, why it was chosen, key numbers, conditions.
    """
    import math
    def _f(v, d=0):
        try:
            val = float(v)
            return d if math.isnan(val) or math.isinf(val) else val
        except: return d

    income = _f(customer.get("monthly_income"), 85000)
    expenses = _f(customer.get("essential_expenses"), 50000)
    emi = _f(customer.get("emi"), 18500)
    principal = _f(customer.get("principal"), 500000)

    dicr = (income - expenses) / emi if emi > 0 else 0
    if math.isnan(dicr): dicr = 0.0
    
    npv = _f(results.get("npv"), 0)
    recovery = _f(results.get("recovery_rate"), 0)
    acceptance = _f(results.get("acceptance_prob"), 0)
    composite = _f(results.get("composite"), 0)
    new_emi = _f(results.get("new_emi"), emi)

    templates = {
        "emi_holiday": (
            f"EMI Holiday grants {results.get('holiday_months', 2)} months of payment relief. "
            f"During this period, interest of ₹{results.get('capitalized_interest', 0):,.0f} is capitalized onto the principal. "
            f"Post-holiday EMI becomes ₹{new_emi:,.0f}. "
            f"DICR at current income is {dicr:.2f}x. "
            f"Because DICR {'dropped below 2.0x' if dicr < 2 else 'remains healthy'}, "
            f"the holiday reduces immediate default probability, improving recovery rate to {recovery:.1%}. "
            f"Discount rate: {config.get('discount_rate', 0.08):.0%} p.a."
        ),
        "graduated_emi": _graduated_emi_text(config, new_emi, emi, dicr, recovery, acceptance),
        "icr": (
            f"Income-Contingent Repayment links EMI to verified income at ϕ={config.get('phi_icr', 0.22):.0%}. "
            f"EMI floor: ₹{config.get('emi_min', 10000):,.0f}. "
            f"Monte Carlo simulation ({results.get('mc_runs', 10000):,} runs) yields mean NPV ₹{npv:,.0f} "
            f"(5th-95th: ₹{results.get('p5', 0):,.0f}–₹{results.get('p95', 0):,.0f}). "
            f"Recovery rate: {recovery:.1%}. Income volatility captured via AR(1) model."
        ),
        "asset_backed": (
            f"Asset-Backed Liquidity Injection: lien of ₹{results.get('lien_amount', 0):,.0f} placed on liquid assets. "
            f"Provides {results.get('relief_months', 6)} months of payment relief. "
            f"ACR is {results.get('acr', 0):.2f} (minimum required: {config.get('acr_min', 0.75):.2f}). "
            f"Lien released after {config.get('asset_backed', {}).get('release_ontime_payments', 6)} on-time payments. "
            f"Recovery rate: {recovery:.1%}."
        ),
        "consolidation": (
            f"Debt Consolidation merges {results.get('num_debts', 0)} obligations into a single loan. "
            f"Weighted average rate drops from {results.get('old_weighted_rate', 0):.1%} to {results.get('new_rate', 0):.1%}. "
            f"New EMI: ₹{new_emi:,.0f} (saves ₹{max(0, emi - new_emi):,.0f}/month). "
            f"Waterfall: high-cost facilities repaid first. "
            f"Post-consolidation DICR: {results.get('post_dicr', dicr):.2f}x. Recovery rate: {recovery:.1%}."
        ),
    }

    base = templates.get(pathway_name, f"Pathway {pathway_name}: Recovery rate {recovery:.1%}, NPV ₹{npv:,.0f}.")
    return base


def generate_short_explanation(pathway_name: str,
                                results: Dict[str, float]) -> str:
    """Generate 1-line recommendation text."""
    new_emi = results.get("new_emi", 0)
    recovery = results.get("recovery_rate", 0)
    savings = results.get("monthly_savings", 0)

    import math
    def _f_emi(v): return f"₹{float(v):,.0f}" if v is not None and not math.isnan(float(v)) else "₹0"
    def _f_pct(v): return f"{float(v):.0%}" if v is not None and not math.isnan(float(v)) else "0%"
    def _f_sav(v): return f"₹{float(v):,.0f}" if v is not None and not math.isnan(float(v)) else "₹0"

    display_names = {
        "emi_holiday": "EMI Holiday",
        "graduated_emi": "Graduated EMI",
        "icr": "Income-Contingent Repayment",
        "asset_backed": "Asset-Backed Liquidity",
        "consolidation": "Debt Consolidation",
    }
    name = display_names.get(pathway_name, pathway_name)

    if savings > 0:
        return f"Recommended: {name} — lowers monthly outflow to {_f_emi(new_emi)} (saves {_f_sav(savings)}/mo), recovery {_f_pct(recovery)}."
    else:
        return f"Recommended: {name} — new EMI {_f_emi(new_emi)}, recovery rate {_f_pct(recovery)}."


def generate_rm_email_text(pathway_name: str,
                            customer: Dict,
                            results: Dict[str, float],
                            config: Dict) -> str:
    """Generate Relationship Manager email template."""
    customer_name = customer.get("name", customer.get("customer_id", "Customer"))
    short = generate_short_explanation(pathway_name, results)
    detail = generate_explainability_text(pathway_name, customer, results, config)

    email = f"""Subject: Recovery Pathway Recommendation — {customer_name}

Dear Relationship Manager,

{short}

--- Details ---
{detail}

--- Customer Impact ---
• New monthly EMI: ₹{results.get('new_emi', 0):,.0f}
• Monthly savings: ₹{results.get('monthly_savings', 0):,.0f}
• Total interest over tenure: ₹{results.get('total_interest', 0):,.0f}
• Recovery rate: {results.get('recovery_rate', 0):.1%}
• Composite score: {results.get('composite', 0):.3f}

--- Conditions ---
• Quarterly income verification required
• No new unsecured credit for 12 months
• Subject to policy approval

Generated by PDIE Recovery Path Engine v{config.get('model_version', '2.0.0')}
Timestamp: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}
"""
    return email

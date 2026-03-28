"""
AI Hub — Unified Barclays PDIE AI Command Center
Merges: AI Communication Agent + Agentic AI + Recovery Message Flow + Financial Health Calculator

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
import theme

sys.path.insert(0, str(Path(__file__).parent))

try:
    from ai_agent import (
        AICommunicationAgent,
        CustomerContext,
        MessageChannel,
        RiskTier,
        create_context_from_customer,
    )

    HAS_AI_AGENT = True
except Exception:
    HAS_AI_AGENT = False

try:
    from pathway_simulator import (
        CustomerProfile,
        simulate_all_pathways,
        load_engine_config,
    )

    HAS_SIMULATOR = True
except Exception:
    HAS_SIMULATOR = False

try:
    from agentic_engine import AgenticPDIE, PDIEAgenticEngine, TOOL_REGISTRY, ToolStatus

    HAS_AGENTIC = True
except Exception:
    HAS_AGENTIC = False


# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
HUB_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { box-sizing: border-box; }
html, body, [class*="st-"] { font-family: 'Inter', sans-serif !important; }

/* NAV TABS */
.hub-tab-bar { display:flex; gap:0.5rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.hub-tab {
    padding:0.55rem 1.2rem; border-radius:10px; border:1px solid rgba(255,255,255,0.1);
    background:rgba(255,255,255,0.04); color:#94a3b8; font-size:0.85rem; font-weight:600;
    cursor:pointer; transition:all 0.2s ease;
}
.hub-tab:hover { border-color:rgba(0,163,224,0.4); color:#38bdf8; }
.hub-tab.active { background:linear-gradient(135deg,#00539B,#00A3E0); color:#fff; border-color:transparent; }

/* SECTION HEADER */
.hub-header {
    background: linear-gradient(120deg, #0a0e1a 0%, #0d2144 60%, #003366 100%);
    border:1px solid rgba(0,163,224,0.2); border-radius:16px;
    padding:1.4rem 2rem; margin-bottom:1.5rem;
    display:flex; align-items:center; justify-content:space-between;
}
.hub-header-left h2 { color:#e0f2fe; margin:0 0 0.25rem; font-size:1.35rem; font-weight:800; }
.hub-header-left p  { color:#94a3b8; margin:0; font-size:0.82rem; }
.hub-badge {
    background:linear-gradient(135deg,#00539B,#00A3E0); color:#fff;
    padding:0.3rem 0.9rem; border-radius:20px; font-size:0.78rem; font-weight:700;
    letter-spacing:0.5px;
}

/* CARD */
.hub-card {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:1.3rem 1.5rem; margin-bottom:1rem;
    transition: border-color 0.2s;
}
.hub-card:hover { border-color:rgba(0,163,224,0.3); }
.hub-card h4 { color:#7dd3fc; margin:0 0 0.8rem; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; }

/* CHAT BUBBLES */
.chat-window {
    background:#0a0e1a; border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:1.2rem; min-height:200px; max-height:520px;
    overflow-y:auto; margin:1rem 0;
}
.bubble-bank {
    background:linear-gradient(135deg,#00539B,#0066bb); color:#fff;
    border-radius:18px 18px 18px 4px; padding:0.9rem 1.2rem;
    margin:0.6rem 0; max-width:82%; font-size:0.88rem; line-height:1.6;
    box-shadow:0 2px 12px rgba(0,83,155,0.3);
}
.bubble-bank .bub-header {
    font-size:0.7rem; font-weight:700; color:rgba(255,255,255,0.6);
    text-transform:uppercase; letter-spacing:0.6px; margin-bottom:0.4rem;
}
.bubble-bank .bub-time { font-size:0.68rem; color:rgba(255,255,255,0.45); margin-top:0.4rem; text-align:right; }
.bubble-cta {
    background:rgba(0,163,224,0.12); border:1px solid rgba(0,163,224,0.3);
    border-radius:10px; padding:0.7rem 1rem; margin:0.4rem 0;
    font-size:0.82rem; color:#7dd3fc; font-weight:600;
}
.bubble-recovery {
    background:linear-gradient(135deg,#064e3b,#065f46);
    border:1px solid rgba(74,222,128,0.3);
    border-radius:18px; padding:1rem 1.2rem; margin:0.5rem 0;
    max-width:90%; font-size:0.85rem; line-height:1.6; color:#d1fae5;
}
.bubble-recovery .rp-title { font-size:0.95rem; font-weight:800; color:#4ade80; margin-bottom:0.5rem; }

/* STATUS STEPS */
.step-row { display:flex; align-items:center; gap:0.8rem; padding:0.6rem 0; }
.step-icon { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.85rem; flex-shrink:0; }
.step-done { background:rgba(74,222,128,0.2); border:1px solid #4ade80; }
.step-active { background:rgba(56,189,248,0.2); border:1px solid #38bdf8; }
.step-pending { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15); }
.step-text { font-size:0.82rem; color:#cbd5e1; }
.step-text strong { color:#e2e8f0; }

/* METRIC PILL */
.metric-pill {
    display:inline-flex; flex-direction:column; align-items:center;
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
    border-radius:12px; padding:0.7rem 1.1rem; text-align:center; min-width:100px;
}
.mp-val { font-size:1.3rem; font-weight:800; }
.mp-lbl { font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:0.6px; margin-top:0.15rem; }

/* HEALTH SCORE RING */
.health-ring-wrap { text-align:center; padding:1rem; }
.health-score-big { font-size:3rem; font-weight:900; }

/* INPUT GROUP */
.input-label { font-size:0.78rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.2rem; font-weight:600; }
.output-block { background:rgba(0,83,155,0.1); border:1px solid rgba(0,163,224,0.2); border-radius:12px; padding:1rem 1.4rem; margin-top:0.8rem; }
.output-block .ob-title { font-size:0.78rem; color:#38bdf8; text-transform:uppercase; font-weight:700; margin-bottom:0.6rem; }
.output-row { display:flex; justify-content:space-between; align-items:center; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05); }
.output-row:last-child { border-bottom:none; }
.or-label { font-size:0.82rem; color:#94a3b8; }
.or-value { font-size:0.9rem; font-weight:700; }
.green { color:#4ade80; } .yellow { color:#fbbf24; } .red { color:#f87171; } .blue { color:#38bdf8; }

/* AGENT LOG */
.agent-log {
    background:#0a0e1a; border:1px solid rgba(255,255,255,0.06);
    border-radius:10px; padding:0.8rem 1rem; font-family:monospace;
    font-size:0.78rem; color:#64748b; max-height:280px; overflow-y:auto;
}
.log-line { padding:0.15rem 0; border-bottom:1px solid rgba(255,255,255,0.03); }
.log-ok   { color:#4ade80; }
.log-warn { color:#fbbf24; }
.log-info { color:#38bdf8; }
.log-err  { color:#f87171; }

/* WORKFLOW LANE */
.wf-lane { display:flex; align-items:center; gap:0; overflow-x:auto; padding:0.5rem 0 1rem; }
.wf-node {
    min-width:130px; text-align:center; padding:0.8rem 0.5rem;
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
    border-radius:12px; font-size:0.78rem; font-weight:600; color:#cbd5e1; flex-shrink:0;
    position:relative;
}
.wf-node.done { border-color:#4ade80; background:rgba(74,222,128,0.08); color:#4ade80; }
.wf-node.active { border-color:#38bdf8; background:rgba(56,189,248,0.1); color:#38bdf8; }
.wf-arrow { font-size:1.2rem; color:#334155; padding:0 0.3rem; flex-shrink:0; }
.wf-icon { font-size:1.4rem; display:block; margin-bottom:0.3rem; }
</style>
"""


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────


def fmt(v):
    import math

    try:
        if v is None or math.isnan(float(v)):
            return "₹0"
        return f"₹{float(v):,.0f}"
    except:
        return "₹0"


def _safe_int(v, default=0):
    import pandas as pd
    import math

    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return int(default)
        return int(f)
    except:
        return int(default)


def _risk_color(score):
    if score >= 80:
        return "#f87171", "CRITICAL"
    if score >= 70:
        return "#fb923c", "HIGH"
    if score >= 50:
        return "#fbbf24", "ELEVATED"
    return "#4ade80", "LOW"


def _health_color(score):
    if score >= 75:
        return "#4ade80", "Healthy"
    if score >= 55:
        return "#fbbf24", "Fair"
    if score >= 35:
        return "#fb923c", "At Risk"
    return "#f87171", "Critical"


def _generate_proactive_sms(customer: dict, risk_tier: str = "HIGH") -> dict:
    """
    Generate proactive early-stage SMS messages based on customer risk level.
    These are friendly reminders sent BEFORE customer defaults to prevent delinquency.
    """
    name = customer.get("name", "Customer")
    if not name or name == "N/A":
        name = customer.get("customer_id", "Valued Customer")

    emi = fmt(customer.get("emi", 18500))
    next_payment_day = (datetime.now() + timedelta(days=7)).strftime("%d %b %Y")

    # Early Warning Stage - Risk detected but not yet defaulted
    if risk_tier == "HIGH":
        msg_early = f"""Hi {name}, paying your loan on time can improve your credit score significantly. Your next payment of {emi} is due on {next_payment_day}. Stay ahead and keep your credit score healthy! -Barclays Intl"""

        msg_reminder = f"""Hi {name}, this is a friendly reminder from Barclays. Your EMI of {emi} is due soon. Need help with payment? Reply HELP or call us. Let's keep your credit score strong! -Barclays"""

        msg_alert = f"""Hi {name}, we noticed your account shows early signs of financial pressure. Don't worry - Barclays is here to help! Reply YES for free financial consultation. -Barclays Intl"""

        pathway_suggestion = "EMI Holiday / Reduced EMI for 3 months"

    elif risk_tier == "CRITICAL":
        msg_early = f"""Urgent: {name}, your loan payment of {emi} is critical. Contact Barclays NOW on 1800-XXX-XXXX to avoid default. We can help! -Barclays Intl"""

        msg_reminder = f"""Hi {name}, your payment is overdue. This affects your CIBIL score. Pay {emi} today or reply PLAN to see repayment options. -Barclays"""

        msg_alert = f"""Hi {name}, your account needs immediate attention. Barclays offers debt restructuring with up to 30% EMI reduction. Reply RESHUFFLE now. -Barclays Intl"""

        pathway_suggestion = "Emergency Loan Restructuring / Debt Consolidation"

    else:  # MEDIUM or default
        msg_early = f"""Hi {name}, thank you for being a valued Barclays customer! Your next EMI of {emi} is due on {next_payment_day}. Set up auto-pay for convenience! -Barclays"""

        msg_reminder = f"""Hi {name}, just a friendly reminder from Barclays: Your EMI of {emi} is due soon. Stay on track with easy payments! -Barclays"""

        msg_alert = f"""Hi {name}, want to save on interest? Barclays offers top-up loans at 11%. Reply TOPUP for details. -Barclays Intl"""

        pathway_suggestion = "Top-up Loan / Balance Transfer"

    return {
        "early_warning": msg_early,
        "reminder": msg_reminder,
        "alert": msg_alert,
        "pathway_suggestion": pathway_suggestion,
        "risk_tier": risk_tier,
        "next_payment_date": next_payment_day,
    }


def _generate_recovery_pathway(customer: dict, risk_tier: str = "HIGH") -> dict:
    """
    Generate recovery pathway suggestions based on customer risk level and financial profile.
    """
    income = customer.get("income", 85000)
    emi = customer.get("emi", 18500)
    emi_ratio = emi / income if income > 0 else 0

    pathways = []

    if risk_tier == "CRITICAL":
        pathways = [
            {
                "name": "Emergency EMI Holiday",
                "description": "Pause EMI payments for 3-6 months",
                "impact": "Immediate cash flow relief",
                "eligibility": "Available for customers facing temporary hardship",
                "success_rate": "78%",
            },
            {
                "name": "Debt Consolidation",
                "description": "Combine all debts into single lower EMI - reduce multiple EMIs to one manageable payment",
                "impact": "Up to 40% reduction in monthly burden",
                "eligibility": "Multiple active loans with good payment history",
                "success_rate": "85%",
            },
            {
                "name": "Graduated EMI",
                "description": "Start with lower EMI, gradually increase",
                "impact": "30% lower initial payments",
                "eligibility": "New to credit or income expected to increase",
                "success_rate": "72%",
            },
        ]
    elif risk_tier == "HIGH":
        pathways = [
            {
                "name": "EMI Reduction",
                "description": "Reduce EMI by 15-25% with extended tenure",
                "impact": "Lower monthly burden",
                "eligibility": "Stable income with good payment history",
                "success_rate": "82%",
            },
            {
                "name": "Top-up Loan",
                "description": "Additional loan at lower rate to prepay expensive debt",
                "impact": "Consolidate high-interest debt",
                "eligibility": "Good CIBIL score (700+)",
                "success_rate": "88%",
            },
            {
                "name": "Balance Transfer",
                "description": "Transfer balance to lower interest rate loan",
                "description": "Save on interest with 0% transfer option",
                "impact": "Save up to 5% on interest",
                "eligibility": "Good credit history",
                "success_rate": "90%",
            },
        ]
    else:
        pathways = [
            {
                "name": "Auto-Pay Setup",
                "description": "Automate payments to never miss due date",
                "impact": "Never miss a payment, build good credit",
                "eligibility": "All customers",
                "success_rate": "95%",
            },
            {
                "name": "Digital Savings",
                "description": "Set aside automatic savings for loan repayment",
                "impact": "Build buffer while repaying",
                "eligibility": "All customers",
                "success_rate": "92%",
            },
        ]

    return {
        "risk_tier": risk_tier,
        "current_emi_ratio": f"{emi_ratio * 100:.1f}%",
        "pathways": pathways,
    }


def _generate_professional_message(
    customer: dict, pathway_name: str = None, pathway_details: dict = None
) -> dict:
    """Generate the 3-part professional message sequence."""
    name = customer.get("name", customer.get("customer_id", "Valued Customer"))
    emi = fmt(customer.get("emi", 18500))
    income = fmt(customer.get("income", 85000))
    risk = customer.get("risk_score", 72)
    now = datetime.now().strftime("%d %b %Y")

    # MSG 1 — Problem awareness
    msg1 = f"""Dear {name},

We hope this message finds you well.

At Barclays, your financial wellbeing is our top priority. Our system has flagged some early signals on your account that we want to bring to your attention — not to cause alarm, but because we believe in proactive support.

📌 What we noticed:
• Your recent payment patterns show signs of financial pressure
• Your EMI commitment of {emi}/month is being monitored
• We want to reach out before any difficulty arises

This is NOT a demand notice. We are reaching out as a proactive measure to see if there is anything Barclays can do to support you.

— Barclays Customer Support Team
  {now}"""

    # MSG 2 — Offer of help
    msg2 = f"""Hi {name},

We're following up on our earlier message.

We understand that financial pressures can happen to anyone — and at Barclays, we have dedicated solutions designed specifically for customers like you.

Our team has already analysed your profile and identified personalised recovery options that could significantly reduce your monthly burden.

💬 If you are currently facing any of the following, please reply YES:
  1. Difficulty meeting your monthly EMI
  2. Recent salary delay or income reduction
  3. Unexpected medical or family expenses
  4. Feeling financially stretched this month

We are here to help — not to judge. A YES from you triggers our Recovery Assistance Programme at zero cost to you.

— Barclays Financial Wellness Team"""

    # MSG 3 — Recovery pathway (conditional)
    msg3 = None
    if pathway_name and pathway_details:
        new_emi = fmt(pathway_details.get("new_emi", 0))
        savings = fmt(pathway_details.get("monthly_savings", 0))
        tenure = pathway_details.get("new_tenure_months", 24)
        pw_display = pathway_name.replace("_", " ").title()
        recovery_pct = min(pathway_details.get("recovery_rate", 0.85) * 100, 99)

        msg3 = f"""Dear {name},

Thank you for reaching out. We truly appreciate your trust in Barclays.

Based on a detailed analysis of your financial profile, our Recovery Path Engine has identified the most suitable solution for your situation:

━━━━━━━━━━━━━━━━━━━━━━━
🏦 YOUR PERSONALISED RECOVERY PLAN
Pathway: {pw_display}
━━━━━━━━━━━━━━━━━━━━━━━

✅ New Monthly EMI: {new_emi}
💰 Monthly Savings: {savings}
📅 New Duration: {tenure} months
📊 Recovery Confidence: {recovery_pct:.0f}%

What this means for you:
• Your monthly payment burden is reduced immediately
• No penalty for opting into this plan
• Your credit profile is protected during this period
• Quarterly reviews ensure the plan stays right for you

To accept this offer, simply reply ACCEPT to this message or call 1800-XXX-XXXX.

This offer is valid for 7 days from {now}.

We are committed to your financial recovery.

Warm regards,
Barclays Financial Wellness Team
Ref: PDIE-{datetime.now().strftime("%Y%m%d%H%M")}"""

    return {"msg1": msg1, "msg2": msg2, "msg3": msg3}


def _compute_financial_health(income, expenses, emi, savings, debt_total, cibil):
    """Compute financial health score and breakdown from inputs."""
    if income <= 0:
        return {
            "score": 0,
            "grade": "N/A",
            "risk_factors": [],
            "recommendations": [],
            "benchmarks": {},
        }

    emi_ratio = emi / income
    expense_ratio = expenses / income
    savings_ratio = savings / income if savings > 0 else 0
    debt_income = debt_total / (income * 12) if income > 0 else 0
    disposable = income - expenses - emi

    total_obligations = expenses + emi
    obligation_ratio = total_obligations / income if income > 0 else 1
    net_disposable = income - total_obligations
    emergency_fund_days = (
        (savings / (expenses + emi)) * 30 if (expenses + emi) > 0 else 0
    )

    default_prob = 0
    if emi_ratio > 0.50:
        default_prob += 35
    elif emi_ratio > 0.40:
        default_prob += 20
    elif emi_ratio > 0.30:
        default_prob += 5
    if cibil < 650:
        default_prob += 25
    elif cibil < 700:
        default_prob += 10
    if emergency_fund_days < 30:
        default_prob += 15
    elif emergency_fund_days < 90:
        default_prob += 5
    if net_disposable < 0:
        default_prob += 20
    elif net_disposable < 5000:
        default_prob += 10
    default_prob = min(95, default_prob)

    emi_score = max(0, 100 - emi_ratio * 300)
    exp_score = max(0, 100 - expense_ratio * 200)
    sav_score = min(100, savings_ratio * 500)
    dti_score = max(0, 100 - debt_income * 150)
    cibil_score = min(100, (cibil - 300) / 6)
    runway_score = min(100, emergency_fund_days / 3)
    disposable_score = min(100, max(0, (net_disposable / (income * 0.2)) * 100))

    weights = [0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10]
    scores = [
        emi_score,
        exp_score,
        sav_score,
        dti_score,
        cibil_score,
        runway_score,
        disposable_score,
    ]
    total = sum(w * s for w, s in zip(weights, scores))
    color, grade = _health_color(total)

    stress_level = (
        "Low" if emi_ratio < 0.25 else "Medium" if emi_ratio < 0.40 else "High"
    )
    months_runway = int(savings / emi) if emi > 0 else 99

    risk_factors = []
    if emi_ratio > 0.40:
        risk_factors.append(
            {
                "factor": "High EMI Burden",
                "severity": "Critical",
                "detail": f"EMI is {emi_ratio * 100:.0f}% of income (safe: <30%)",
            }
        )
    if cibil < 650:
        risk_factors.append(
            {
                "factor": "Low CIBIL Score",
                "severity": "High",
                "detail": f"Score {cibil} indicates poor credit history",
            }
        )
    if emergency_fund_days < 30:
        risk_factors.append(
            {
                "factor": "Insufficient Emergency Fund",
                "severity": "High",
                "detail": f"Only {emergency_fund_days:.0f} days coverage (recommended: 90+ days)",
            }
        )
    if net_disposable < 5000:
        risk_factors.append(
            {
                "factor": "Low Disposable Income",
                "severity": "Medium",
                "detail": f"Only ₹{net_disposable:,.0f} remaining after obligations",
            }
        )
    if debt_total > income * 24:
        risk_factors.append(
            {
                "factor": "High Total Debt",
                "severity": "Medium",
                "detail": f"Total debt equals {debt_total / income:.1f}x annual income",
            }
        )

    benchmarks = {
        "emi_ratio_industry": 0.28,
        "savings_rate_industry": 0.15,
        "cibil_median": 720,
        "emergency_fund_industry": 90,
    }

    return {
        "score": round(total, 1),
        "color": color,
        "grade": grade,
        "emi_ratio": emi_ratio,
        "expense_ratio": expense_ratio,
        "savings_ratio": savings_ratio,
        "disposable": disposable,
        "net_disposable": net_disposable,
        "dti": debt_income,
        "stress": stress_level,
        "runway": min(months_runway, 36),
        "emergency_fund_days": emergency_fund_days,
        "obligation_ratio": obligation_ratio,
        "default_probability": default_prob,
        "risk_factors": risk_factors,
        "benchmarks": benchmarks,
        "breakdowns": {
            "EMI Burden": round(emi_score, 1),
            "Expense Control": round(exp_score, 1),
            "Savings Rate": round(sav_score, 1),
            "Debt Profile": round(dti_score, 1),
            "Credit Health": round(cibil_score, 1),
            "Emergency Fund": round(runway_score, 1),
            "Cash Flow": round(disposable_score, 1),
        },
    }

    emi_ratio = emi / income
    expense_ratio = expenses / income
    savings_ratio = savings / income if savings > 0 else 0
    debt_income = debt_total / (income * 12) if income > 0 else 0
    disposable = income - expenses - emi

    # Additional metrics
    total_obligations = expenses + emi
    obligation_ratio = total_obligations / income if income > 0 else 1
    net_disposable = income - total_obligations
    emergency_fund_days = (
        (savings / (expenses + emi)) * 30 if (expenses + emi) > 0 else 0
    )
    loan_affordability = (income * 0.4 - emi) / income if income > 0 else 0  # 40% rule
    financial_freedom = (
        (savings / (income * 12)) if income > 0 else 0
    )  # Savings as % of annual income

    # Default probability estimation (simplified model)
    default_prob = 0
    if emi_ratio > 0.50:
        default_prob += 35
    elif emi_ratio > 0.40:
        default_prob += 20
    elif emi_ratio > 0.30:
        default_prob += 5

    if cibil < 650:
        default_prob += 25
    elif cibil < 700:
        default_prob += 10

    if emergency_fund_days < 30:
        default_prob += 15
    elif emergency_fund_days < 90:
        default_prob += 5

    if net_disposable < 0:
        default_prob += 20
    elif net_disposable < 5000:
        default_prob += 10

    default_prob = min(95, default_prob)

    # Score components (0-100 each)
    emi_score = max(0, 100 - emi_ratio * 300)  # 30% EMI/income → 0
    exp_score = max(0, 100 - expense_ratio * 200)  # 50% expenses/income → 0
    sav_score = min(100, savings_ratio * 500)  # 20% savings → 100
    dti_score = max(0, 100 - debt_income * 150)  # DTI ratio
    cibil_score = min(100, (cibil - 300) / 6)  # 300-900 → 0-100

    # New scoring components
    runway_score = min(100, emergency_fund_days / 3)  # 3 months = 100
    disposable_score = min(
        100, max(0, (net_disposable / (income * 0.2)) * 100)
    )  # 20% disposable = 100

    weights = [0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10]
    scores = [
        emi_score,
        exp_score,
        sav_score,
        dti_score,
        cibil_score,
        runway_score,
        disposable_score,
    ]
    total = sum(w * s for w, s in zip(weights, scores))
    color, grade = _health_color(total)

    stress_level = (
        "Low" if emi_ratio < 0.25 else "Medium" if emi_ratio < 0.40 else "High"
    )
    months_runway = int(savings / emi) if emi > 0 else 99

    # Risk factors identification
    risk_factors = []
    if emi_ratio > 0.40:
        risk_factors.append(
            {
                "factor": "High EMI Burden",
                "severity": "Critical",
                "detail": f"EMI is {emi_ratio * 100:.0f}% of income (safe: <30%)",
            }
        )
    if cibil < 650:
        risk_factors.append(
            {
                "factor": "Low CIBIL Score",
                "severity": "High",
                "detail": f"Score {cibil} indicates poor credit history",
            }
        )
    if emergency_fund_days < 30:
        risk_factors.append(
            {
                "factor": "Insufficient Emergency Fund",
                "severity": "High",
                "detail": f"Only {emergency_fund_days:.0f} days coverage (recommended: 90+ days)",
            }
        )
    if net_disposable < 5000:
        risk_factors.append(
            {
                "factor": "Low Disposable Income",
                "severity": "Medium",
                "detail": f"Only ₹{net_disposable:,.0f} remaining after obligations",
            }
        )
    if debt_total > income * 24:
        risk_factors.append(
            {
                "factor": "High Total Debt",
                "severity": "Medium",
                "detail": f"Total debt equals {debt_total / income:.1f}x annual income",
            }
        )
    if savings < expenses * 3:
        risk_factors.append(
            {
                "factor": "Insufficient Savings Buffer",
                "severity": "Low",
                "detail": f"Savings cover only {savings / expenses:.1f} months of expenses",
            }
        )

    # Industry benchmarks
    benchmarks = {
        "emi_ratio_industry": 0.28,
        "savings_rate_industry": 0.15,
        "cibil_median": 720,
        "emergency_fund_industry": 90,
    }

    return {
        "score": round(total, 1),
        "color": color,
        "grade": grade,
        "emi_ratio": emi_ratio,
        "expense_ratio": expense_ratio,
        "savings_ratio": savings_ratio,
        "disposable": disposable,
        "net_disposable": net_disposable,
        "dti": debt_income,
        "stress": stress_level,
        "runway": min(months_runway, 36),
        "emergency_fund_days": emergency_fund_days,
        "obligation_ratio": obligation_ratio,
        "loan_affordability": loan_affordability,
        "financial_freedom": financial_freedom,
        "default_probability": default_prob,
        "risk_factors": risk_factors,
        "benchmarks": benchmarks,
        "breakdowns": {
            "EMI Burden": round(emi_score, 1),
            "Expense Control": round(exp_score, 1),
            "Savings Rate": round(sav_score, 1),
            "Debt Profile": round(dti_score, 1),
            "Credit Health": round(cibil_score, 1),
            "Emergency Fund": round(runway_score, 1),
            "Cash Flow": round(disposable_score, 1),
        },
    }


# ─────────────────────────────────────────────────────────
# SECTION: AI COMMUNICATION — MESSAGE FLOW
# ─────────────────────────────────────────────────────────


def _section_message_flow(customer: dict, sk: str):
    st.markdown(
        """
    <div class="hub-header">
        <div class="hub-header-left">
            <h2>💬 AI Customer Communication</h2>
            <p>3-step professional message flow · Editable before send · WhatsApp & SMS ready</p>
        </div>
        <span class="hub-badge">Barclays AI Messaging</span>
    </div>""",
        unsafe_allow_html=True,
    )

    risk_score = customer.get("risk_score", 72)
    rc, rlabel = _risk_color(risk_score)

    # Customer summary strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-pill"><span class="mp-val" style="color:{rc}">{risk_score}</span><span class="mp-lbl">Risk Score</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-pill"><span class="mp-val blue">{fmt(customer.get("emi", 18500))}</span><span class="mp-lbl">Monthly EMI</span></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-pill"><span class="mp-val green">{fmt(customer.get("income", 85000))}</span><span class="mp-lbl">Monthly Income</span></div>',
            unsafe_allow_html=True,
        )
    with c4:
        channel = st.selectbox(
            "📱 Channel", ["WhatsApp", "SMS", "Email"], key=f"{sk}_chan"
        )

    st.markdown("---")

    # ── Workflow progress ──
    step = st.session_state.get(f"{sk}_msg_step", 0)
    steps_def = [
        ("🏦", "Awareness", "Notify customer of signals"),
        ("💬", "Engagement", "Ask if they need help"),
        ("📋", "Recovery Offer", "Send personalised plan"),
    ]
    wf_html = '<div class="wf-lane">'
    for i, (icon, title, desc) in enumerate(steps_def):
        cls = "done" if i < step else "active" if i == step else ""
        wf_html += f'<div class="wf-node {cls}"><span class="wf-icon">{icon}</span>{title}</div>'
        if i < 2:
            wf_html += '<span class="wf-arrow">→</span>'
    wf_html += "</div>"
    st.markdown(wf_html, unsafe_allow_html=True)

    # ── Generate messages ──
    msgs = _generate_professional_message(customer)

    # Fetch selected pathway for msg3
    pathway_details = st.session_state.get(f"{sk}_pathway_details")
    pathway_name = st.session_state.get(f"{sk}_pathway_name")
    if pathway_details:
        msgs = _generate_professional_message(customer, pathway_name, pathway_details)

    # ── CHAT WINDOW ──
    now_time = datetime.now().strftime("%H:%M")
    chat_html = '<div class="chat-window">'

    if step >= 1:
        chat_html += f"""
        <div class="bubble-bank">
            <div class="bub-header">🏦 Barclays · Awareness Message</div>
            {msgs["msg1"].replace(chr(10), "<br>")}
            <div class="bub-time">✓✓ Delivered · {now_time}</div>
        </div>"""

    if step >= 2:
        chat_html += f"""
        <div class="bubble-bank" style="margin-left:auto; background:linear-gradient(135deg,#1e3a5f,#1e40af);">
            <div class="bub-header">🏦 Barclays · Engagement Check</div>
            {msgs["msg2"].replace(chr(10), "<br>")}
            <div class="bub-time">✓✓ Delivered · {now_time}</div>
        </div>"""

    if step >= 3 and msgs.get("msg3"):
        chat_html += f"""
        <div class="bubble-recovery">
            <div class="rp-title">🌿 Recovery Pathway — Personalised Offer</div>
            {msgs["msg3"].replace(chr(10), "<br>")}
            <div style="font-size:0.7rem; color:#6ee7b7; margin-top:0.5rem;">✓✓ Delivered · {now_time}</div>
        </div>"""
    elif step == 0:
        chat_html += '<div style="text-align:center; color:#334155; padding:2rem;">Click "Step 1 — Send Awareness" to begin the conversation</div>'

    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ── Message editor ──
    if step == 0:
        preview_msg = msgs["msg1"]
        msg_label = "✏️ Edit Awareness Message (Step 1)"
    elif step == 1:
        preview_msg = msgs["msg2"]
        msg_label = "✏️ Edit Engagement Message (Step 2)"
    else:
        preview_msg = msgs.get(
            "msg3",
            "No recovery pathway selected yet. Please select a pathway from Recovery Engine first.",
        )
        msg_label = "✏️ Edit Recovery Offer Message (Step 3)"

    with st.expander(f"✏️ Edit Message Before Sending", expanded=(step < 3)):
        edited = st.text_area(
            msg_label, value=preview_msg, height=280, key=f"{sk}_edit_msg_{step}"
        )
        char_count = len(edited or "")
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            st.caption(f"📏 {char_count} characters · Channel: {channel}")
        with col_c2:
            if channel == "SMS" and char_count > 160:
                st.warning("⚠️ Exceeds SMS limit")

    # ── ACTION BUTTONS ──
    btn1, btn2, btn3, btn4 = st.columns(4)
    with btn1:
        if step == 0 and st.button(
            "📤 Step 1 — Send Awareness", type="primary", key=f"{sk}_snd1"
        ):
            st.session_state[f"{sk}_msg_step"] = 1
            st.success("✅ Awareness message sent!")
            st.rerun()
    with btn2:
        if step == 1 and st.button(
            "💬 Step 2 — Send Engagement", type="primary", key=f"{sk}_snd2"
        ):
            st.session_state[f"{sk}_msg_step"] = 2
            st.success("✅ Engagement message sent! Waiting for customer YES/NO...")
            st.rerun()
    with btn3:
        if step == 2:
            if st.button(
                "✅ Customer Said YES → Send Recovery Plan",
                type="primary",
                key=f"{sk}_snd3",
            ):
                if not pathway_details:
                    st.warning(
                        "⚠️ Please go to ⚡ Recovery Decision Engine first, select a pathway and click 'Offer Now', then return here."
                    )
                else:
                    st.session_state[f"{sk}_msg_step"] = 3
                    st.success("🎉 Recovery pathway offer sent to customer!")
                    st.rerun()
    with btn4:
        if step > 0 and st.button("🔄 Reset Conversation", key=f"{sk}_reset_chat"):
            st.session_state[f"{sk}_msg_step"] = 0
            st.rerun()

    # ── Recovery pathway linker ──
    if step >= 2 and not pathway_details:
        st.markdown(
            """
        <div style="background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.3);
                    border-radius:10px; padding:0.8rem 1.2rem; margin-top:0.5rem; font-size:0.82rem; color:#fcd34d;">
            ⚠️ <strong>Action Required:</strong> Go to the <strong>⚡ Recovery Decision Engine</strong> page,
            explore the pathways, and click <strong>"✅ Offer Now"</strong> on your chosen pathway.
            Then return here to send the personalised recovery message to the customer.
        </div>""",
            unsafe_allow_html=True,
        )

    # ── PROACTIVE SMS SECTION (Moved here from Financial Health) ──
    st.markdown("---")
    _section_proactive_sms(customer, sk)


# ─────────────────────────────────────────────────────────
# SECTION: AI AGENT COMMAND CENTER
# ─────────────────────────────────────────────────────────


def _section_ai_agent(customer: dict, sk: str):
    st.markdown(
        """
    <div class="hub-header">
        <div class="hub-header-left">
            <h2>🤖 AI Agent Command Center</h2>
            <p>Autonomous planning · Tool execution · Real-time monitoring</p>
        </div>
        <span class="hub-badge">Agentic AI</span>
    </div>""",
        unsafe_allow_html=True,
    )

    cid = customer.get("customer_id", "C12345")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### 🎯 What should the AI Agent do?")

        quick_tasks = {
            "🔍 Full Risk Assessment": f"Analyse customer {cid} — check risk score, SHAP factors, financial ratios, and produce a complete intervention recommendation.",
            "📩 Draft Recovery Message": f"For customer {cid}, generate a professional Barclays outreach message based on their risk profile and best recovery pathway.",
            "📊 Pathway Recommendation": f"Simulate all 5 recovery pathways for {cid} and recommend the highest-scoring option with justification.",
            "📞 Schedule Call & SMS": f"For customer {cid}, schedule an outreach call and send an advance SMS notification for the best contact window.",
            "🔁 Full Automation Plan": f"Run the complete PDIE automation for {cid}: risk check → message → pathway → offer → schedule follow-up.",
        }

        selected_task = st.selectbox(
            "Quick Task Templates", list(quick_tasks.keys()), key=f"{sk}_agent_task_sel"
        )
        custom_query = st.text_area(
            "Or describe a custom task:",
            value=quick_tasks[selected_task],
            height=100,
            key=f"{sk}_agent_query",
        )

        run_col, clear_col = st.columns([2, 1])
        with run_col:
            run_agent_btn = st.button(
                "🚀 Run Agentic Recovery Analysis", type="primary", key=f"{sk}_run_agent"
            )
        with clear_col:
            if st.button("🗑️ Clear Log", key=f"{sk}_clear_log"):
                st.session_state[f"{sk}_agent_log"] = []
                st.rerun()

        if run_agent_btn:
            log = st.session_state.get(f"{sk}_agent_log", [])
            log.append(
                {
                    "t": datetime.now().strftime("%H:%M:%S"),
                    "lvl": "info",
                    "msg": f"Agent started: {selected_task}",
                }
            )

            if HAS_AGENTIC:
                try:
                    # Initialize the REAL Agentic Engine
                    agent = PDIEAgenticEngine()
                    # Run the agent autonomously with PydanticAI
                    session_result = agent.run_agent(customer, custom_query)
                    
                    # Display the Reasoning Steps (Chain of Thought)
                    for step_item in session_result.steps:
                        log.append(
                            {
                                "t": datetime.now().strftime("%H:%M:%S"),
                                "lvl": "info" if "Reasoning" in step_item.action else "ok",
                                "msg": f"[{step_item.action}] {step_item.thought}",
                            }
                        )
                    
                    # Final Plan Result
                    log.append(
                        {
                            "t": datetime.now().strftime("%H:%M:%S"),
                            "lvl": "ok",
                            "msg": f"✅ Agentic Logic Complete: {session_result.final_answer}",
                        }
                    )
                except Exception as e:
                    log.append(
                        {
                            "t": datetime.now().strftime("%H:%M:%S"),
                            "lvl": "err",
                            "msg": f"Agentic Error: {str(e)}",
                        }
                    )
            else:
                # Simulated execution steps
                sim_steps = [
                    ("ok", f"✓ Loaded customer profile: {cid}"),
                    (
                        "ok",
                        f"✓ Risk score retrieved: {customer.get('risk_score', 72)}/100",
                    ),
                    ("info", f"→ Running pathway simulation engine..."),
                    ("ok", f"✓ Best pathway identified: Graduated EMI (Score: 0.731)"),
                    ("ok", f"✓ Message drafted using Barclays tone guidelines"),
                    ("ok", f"✓ Contact window identified: Thursday 3-6 PM"),
                    ("ok", f"✓ Offer letter generated and queued"),
                    (
                        "ok",
                        f"✓ Audit trail logged (ID: AUD-{cid}-{datetime.now().strftime('%Y%m%d')})",
                    ),
                    ("ok", "✅ Agent task completed successfully."),
                ]
                for lvl, msg in sim_steps:
                    log.append(
                        {
                            "t": datetime.now().strftime("%H:%M:%S"),
                            "lvl": lvl,
                            "msg": msg,
                        }
                    )

            st.session_state[f"{sk}_agent_log"] = log
            st.rerun()

        # Log display
        log_entries = st.session_state.get(f"{sk}_agent_log", [])
        if log_entries:
            log_html = '<div class="agent-log">'
            for e in log_entries:
                log_html += f'<div class="log-line log-{e["lvl"]}"><span style="color:#475569">[{e["t"]}]</span>  {e["msg"]}</div>'
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

    with right:
        st.markdown("#### ⚙️ Agent Tools Active")
        tools = [
            ("🔍", "Risk Analyser", "Active"),
            ("💬", "Message Composer", "Active"),
            ("📊", "Pathway Simulator", "Active"),
            ("📞", "Call Scheduler", "Simulated"),
            ("📧", "Email Engine", "Active"),
            ("🔒", "Audit Logger", "Active"),
            ("🤖", "Gemini AI", "Simulated"),
        ]
        for icon, name, status in tools:
            col_s = (
                "#4ade80"
                if status == "Active"
                else "#fbbf24"
                if status == "Simulated"
                else "#f87171"
            )
            st.markdown(
                f"""
            <div class="step-row">
                <div class="step-icon step-{"done" if status == "Active" else "active"}"><span>{icon}</span></div>
                <div class="step-text"><strong>{name}</strong><br><span style="color:{col_s};font-size:0.72rem">{status}</span></div>
            </div>""",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────
# SECTION: FINANCIAL HEALTH CALCULATOR
# ─────────────────────────────────────────────────────────


def _section_financial_health(customer: dict, sk: str):
    st.markdown(
        """
    <div class="hub-header">
        <div class="hub-header-left">
            <h2>💰 Financial Health Calculator</h2>
            <p>Enter customer financials → Get AI-powered health assessment, stress indicators & recommendations</p>
        </div>
        <span class="hub-badge">Real-time Analysis</span>
    </div>""",
        unsafe_allow_html=True,
    )

    # ── INPUT PANEL ──
    st.markdown("#### 📝 Enter Financial Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="input-label">Monthly Income (₹)</div>', unsafe_allow_html=True
        )
        income = st.number_input(
            "",
            min_value=0,
            max_value=5000000,
            value=_safe_int(customer.get("income"), 85000),
            step=1000,
            key=f"{sk}_fh_income",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="input-label" style="margin-top:0.8rem">Monthly EMI (₹)</div>',
            unsafe_allow_html=True,
        )
        emi = st.number_input(
            "",
            min_value=0,
            max_value=500000,
            value=_safe_int(customer.get("emi"), 18500),
            step=500,
            key=f"{sk}_fh_emi",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown(
            '<div class="input-label">Monthly Expenses (₹)</div>',
            unsafe_allow_html=True,
        )
        expenses = st.number_input(
            "",
            min_value=0,
            max_value=2000000,
            value=_safe_int(customer.get("expenses"), income * 0.55),
            step=1000,
            key=f"{sk}_fh_exp",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="input-label" style="margin-top:0.8rem">Current Savings (₹)</div>',
            unsafe_allow_html=True,
        )
        savings = st.number_input(
            "",
            min_value=0,
            max_value=10000000,
            value=_safe_int(customer.get("assets"), income * 2),
            step=5000,
            key=f"{sk}_fh_sav",
            label_visibility="collapsed",
        )

    with col3:
        st.markdown(
            '<div class="input-label">Total Other Debts (₹)</div>',
            unsafe_allow_html=True,
        )
        debts = st.number_input(
            "",
            min_value=0,
            max_value=10000000,
            value=_safe_int(customer.get("other_debts_total"), 200000),
            step=10000,
            key=f"{sk}_fh_debt",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="input-label" style="margin-top:0.8rem">CIBIL Score</div>',
            unsafe_allow_html=True,
        )
        cibil = st.number_input(
            "",
            min_value=300,
            max_value=900,
            value=_safe_int(customer.get("cibil_score"), 680),
            step=5,
            key=f"{sk}_fh_cibil",
            label_visibility="collapsed",
        )

    analyse = st.button(
        "🔬 Analyse Financial Health", type="primary", key=f"{sk}_fh_run"
    )
    if analyse or st.session_state.get(f"{sk}_fh_done"):
        st.session_state[f"{sk}_fh_done"] = True
        h = _compute_financial_health(income, expenses, emi, savings, debts, cibil)

        st.markdown("---")
        st.markdown("#### 📊 Health Assessment Results")

        # Health score + grade
        left_h, right_h = st.columns([1, 3])
        with left_h:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=h["score"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"color": "#64748b"}},
                        "bar": {"color": h["color"]},
                        "steps": [
                            {"range": [0, 35], "color": "rgba(248,113,113,0.15)"},
                            {"range": [35, 55], "color": "rgba(251,191,36,0.12)"},
                            {"range": [55, 75], "color": "rgba(74,222,128,0.08)"},
                            {"range": [75, 100], "color": "rgba(74,222,128,0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": h["color"], "width": 3},
                            "value": h["score"],
                        },
                    },
                    number={"font": {"color": h["color"], "size": 36}},
                    title={
                        "text": f"Health Score<br><b style='color:{h['color']}'>{h['grade']}</b>",
                        "font": {"color": "#94a3b8", "size": 13},
                    },
                )
            )
            fig_gauge.update_layout(
                height=200,
                margin=dict(t=30, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with right_h:
            # Component breakdown radar
            labels = list(h["breakdowns"].keys())
            values = list(h["breakdowns"].values())
            _FILL_MAP = {
                "#4ade80": "rgba(74,222,128,0.12)",
                "#fbbf24": "rgba(251,191,36,0.12)",
                "#fb923c": "rgba(251,146,60,0.12)",
                "#f87171": "rgba(248,113,113,0.12)",
            }
            fig_r = go.Figure()
            fig_r.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=labels + [labels[0]],
                    fill="toself",
                    name="Health",
                    line=dict(color=h["color"], width=2),
                    fillcolor=_FILL_MAP.get(h["color"], "rgba(100,116,139,0.12)"),
                )
            )
            fig_r.update_layout(
                height=220,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.06)"
                    ),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                ),
                showlegend=False,
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # Key output metrics - Enhanced with more columns
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        disposable = income - expenses - emi
        stress_col = (
            "#4ade80"
            if h["stress"] == "Low"
            else "#fbbf24"
            if h["stress"] == "Medium"
            else "#f87171"
        )
        runway_col = (
            "#4ade80"
            if h["runway"] >= 6
            else "#fbbf24"
            if h["runway"] >= 3
            else "#f87171"
        )
        default_col = (
            "#4ade80"
            if h["default_probability"] < 15
            else "#fbbf24"
            if h["default_probability"] < 35
            else "#f87171"
        )

        metrics_data = [
            (
                m1,
                "EMI/Income",
                f"{h['emi_ratio'] * 100:.1f}%",
                "#4ade80"
                if h["emi_ratio"] < 0.30
                else "#fbbf24"
                if h["emi_ratio"] < 0.45
                else "#f87171",
            ),
            (
                m2,
                "Savings Rate",
                f"{h['savings_ratio'] * 100:.1f}%",
                "#4ade80" if h["savings_ratio"] > 0.15 else "#fbbf24",
            ),
            (
                m3,
                "Disposable/mo",
                fmt(max(0, disposable)),
                "#4ade80" if disposable > 10000 else "#f87171",
            ),
            (m4, "Stress Level", h["stress"], stress_col),
            (m5, "Months Runway", f"{h['runway']} mo", runway_col),
            (m6, "Default Prob", f"{h['default_probability']:.0f}%", default_col),
        ]

        for col_w, label, val, color in metrics_data:
            with col_w:
                st.markdown(
                    f'<div class="metric-pill"><span class="mp-val" style="color:{color}">{val}</span><span class="mp-lbl">{label}</span></div>',
                    unsafe_allow_html=True,
                )

        # Additional metrics row
        st.markdown("##### 📊 Detailed Financial Ratios")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric(
                "Total Obligations",
                f"₹{fmt(expenses + emi)}",
                f"{(h['obligation_ratio']) * 100:.1f}% of income",
            )
        with r2:
            st.metric(
                "Net Disposable",
                f"₹{fmt(h['net_disposable'])}",
                "After all obligations"
                if h["net_disposable"] > 0
                else "Negative cashflow!",
            )
        with r3:
            st.metric(
                "Emergency Fund",
                f"{h['emergency_fund_days']:.0f} days",
                "90+ days recommended"
                if h["emergency_fund_days"] >= 90
                else "Insufficient!",
            )
        with r4:
            debt_service = h["obligation_ratio"] * 100
            st.metric(
                "Debt Service Ratio",
                f"{debt_service:.1f}%",
                "<36% is healthy" if debt_service < 36 else "Too high!",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk Factors Section
        if h.get("risk_factors"):
            st.markdown("#### ⚠️ Identified Risk Factors")
            for rf in h["risk_factors"]:
                severity_color = (
                    "#dc2626"
                    if rf["severity"] == "Critical"
                    else "#ea580c"
                    if rf["severity"] == "High"
                    else "#ca8a04"
                )
                severity_bg = (
                    "#fef2f2"
                    if rf["severity"] == "Critical"
                    else "#fff7ed"
                    if rf["severity"] == "High"
                    else "#fefce8"
                )
                st.markdown(
                    f"""
                <div style="background:{severity_bg}; border-left:4px solid {severity_color}; border-radius:0 8px 8px 0; padding:0.8rem 1rem; margin:0.3rem 0;">
                    <span style="color:{severity_color}; font-weight:700; font-size:0.8rem;">{rf["severity"].upper()}</span>
                    <div style="color:#1e293b; font-weight:600; margin-top:0.2rem;">{rf["factor"]}</div>
                    <div style="color:#64748b; font-size:0.8rem;">{rf["detail"]}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

        # Industry Benchmarks Comparison
        if h.get("benchmarks"):
            st.markdown("#### 📊 Industry Benchmarks Comparison")
            bm = h["benchmarks"]
            bench_cols = st.columns(4)
            with bench_cols[0]:
                emi_diff = (h["emi_ratio"] - bm["emi_ratio_industry"]) * 100
                st.metric(
                    "EMI Ratio",
                    f"{h['emi_ratio'] * 100:.1f}%",
                    f"{emi_diff:+.1f}% vs industry 28%",
                )
            with bench_cols[1]:
                sav_diff = (h["savings_ratio"] - bm["savings_rate_industry"]) * 100
                st.metric(
                    "Savings Rate",
                    f"{h['savings_ratio'] * 100:.1f}%",
                    f"{sav_diff:+.1f}% vs industry 15%",
                )
            with bench_cols[2]:
                st.metric(
                    "CIBIL Score",
                    f"{cibil}",
                    f"{cibil - bm['cibil_median']:+d} vs median 720",
                )
            with bench_cols[3]:
                days_diff = h["emergency_fund_days"] - bm["emergency_fund_industry"]
                st.metric(
                    "Emergency Fund",
                    f"{h['emergency_fund_days']:.0f} days",
                    f"{days_diff:+.0f} days vs 90 days",
                )

        # Default Probability Gauge
        st.markdown("#### 🎯 21-Day Default Probability")
        def_prob = h.get("default_probability", 0)
        def_color = (
            "#22c55e" if def_prob < 15 else "#eab308" if def_prob < 35 else "#ef4444"
        )
        fig_def = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=def_prob,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#64748b"}},
                    "bar": {"color": def_color},
                    "steps": [
                        {"range": [0, 15], "color": "rgba(34,197,94,0.15)"},
                        {"range": [15, 35], "color": "rgba(234,179,8,0.15)"},
                        {"range": [35, 100], "color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": def_color, "width": 3},
                        "value": def_prob,
                    },
                },
                number={"font": {"color": def_color, "size": 32}},
                title={
                    "text": f"Risk Level<br><b style='color:{def_color}'>{'LOW' if def_prob < 15 else 'MEDIUM' if def_prob < 35 else 'HIGH'}</b>",
                    "font": {"color": "#94a3b8", "size": 12},
                },
            )
        )
        fig_def.update_layout(
            height=180,
            margin=dict(t=30, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_def, use_container_width=True)

        # Recommendations
        recs = []
        if h["emi_ratio"] > 0.40:
            recs.append(
                (
                    "🔴",
                    "EMI Overload",
                    f"Your EMI is {h['emi_ratio'] * 100:.0f}% of income — well above 30%. Consider EMI Holiday or Graduated EMI pathway.",
                )
            )
        if h["savings_ratio"] < 0.10:
            recs.append(
                (
                    "🟡",
                    "Low Savings Buffer",
                    f"Emergency fund covers only {h['runway']} months. Target at least 3 months of EMI as buffer.",
                )
            )
        if cibil < 650:
            recs.append(
                (
                    "🟠",
                    "Credit Health Risk",
                    f"CIBIL {cibil} is below 700. Timely payments over 6 months can improve by 40-60 points.",
                )
            )
        if h["net_disposable"] < 5000:
            recs.append(
                (
                    "🔴",
                    "Thin Disposable Income",
                    f"Only ₹{h['net_disposable']:,.0f} free cash after obligations. High default risk in income shock.",
                )
            )
        if h["emergency_fund_days"] < 30:
            recs.append(
                (
                    "🔴",
                    "Insufficient Emergency Fund",
                    f"Only {h['emergency_fund_days']:.0f} days. Industry standard recommends 90+ days.",
                )
            )
        if not recs:
            recs.append(
                (
                    "🟢",
                    "Financially Stable",
                    "Your ratios are within healthy limits. Continue maintaining timely payments and growing savings buffer.",
                )
            )

        st.markdown("#### 💡 Personalized Recommendations")
        for icon, title, text in recs:
            border_color = (
                "#f87171"
                if "🔴" in icon
                else "#fbbf24"
                if "🟡" in icon
                else "#fb923c"
                if "🟠" in icon
                else "#4ade80"
            )
            st.markdown(
                f"""
            <div style="background:rgba(255,255,255,0.03); border-left:3px solid {border_color}; border-radius:0 10px 10px 0; padding:0.8rem 1.1rem; margin:0.4rem 0; font-size:0.85rem; color:#cbd5e1;">
                {icon} <strong style="color:#e2e8f0">{title}</strong><br>{text}
            </div>""",
                unsafe_allow_html=True,
            )

        # Trend chart — projected 6-month trajectory
        st.markdown("#### 📈 6-Month Financial Trajectory Forecast")
        months = ["Now", "M+1", "M+2", "M+3", "M+4", "M+5", "M+6"]
        score_now = h["score"]
        # Improvement trajectory if recommended actions taken
        if score_now < 50:
            trajectory = [score_now + i * 3.5 for i in range(7)]
        elif score_now < 70:
            trajectory = [score_now + i * 1.5 for i in range(7)]
        else:
            trajectory = [min(100, score_now + i * 0.8) for i in range(7)]
        current_path = [max(0, score_now - i * 2.0) for i in range(7)]  # No action path

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=months,
                y=[min(100, v) for v in trajectory],
                name="With Recovery Plan",
                line=dict(color="#4ade80", width=3),
                fill="tozeroy",
                fillcolor="rgba(74,222,128,0.06)",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=months,
                y=[max(0, v) for v in current_path],
                name="Without Action",
                line=dict(color="#f87171", width=2, dash="dash"),
            )
        )
        fig_trend.update_layout(
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=30, l=40, r=20),
            yaxis_title="Health Score",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 105]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)


# ─────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────


def show_ai_hub(customer_data: dict | None = None):
    st.markdown(HUB_CSS, unsafe_allow_html=True)

    if customer_data is None:
        customer_data = {
            "customer_id": "C12345",
            "name": "Demo Customer",
            "income": 85000,
            "expenses": 46750,
            "emi": 18500,
            "assets": 170000,
            "cibil_score": 680,
            "risk_score": 74,
        }

    cid = customer_data.get("customer_id", "C00000")
    sk = f"hub_{cid}"

    # Sync selected pathway from RDE session state
    rde_sk = f"rde_{cid}"
    selected_pw = st.session_state.get(f"{rde_sk}_selected")
    if selected_pw and not st.session_state.get(f"{sk}_pathway_name"):
        # Attempt to load pathway details from RDE results if cached
        st.session_state[f"{sk}_pathway_name"] = selected_pw

    # Page title
    risk_score = customer_data.get("risk_score", 72)
    rc, rlabel = _risk_color(risk_score)
    st.markdown(
        f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem;">
        <div>
            <h1 style="color:#e0f2fe; margin:0; font-size:1.6rem; font-weight:900;">🏦 Barclays Financial Health Calculator</h1>
            <p style="color:#64748b; margin:0.2rem 0 0; font-size:0.82rem;">
                Income <strong style="color:#4ade80">{fmt(customer_data.get("income"))}</strong> &nbsp;·&nbsp;
                EMI <strong style="color:#38bdf8">{fmt(customer_data.get("emi"))}</strong> &nbsp;·&nbsp;
                Total Loan <strong style="color:#fbbf24">{fmt(customer_data.get("total_loan"))}</strong> &nbsp;·&nbsp;
                City <strong style="color:#a78bfa">{customer_data.get("city")}</strong>
            </p>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    # ── HUB NAVIGATION ──
    tabs = {
        "📊 Financial Health": _section_financial_health,
        "💬 Message Flow": _section_message_flow,
        "🤖 AI Agent": _section_ai_agent,
        "📱 Proactive SMS": _section_proactive_sms
    }
    
    # Use streamlit tabs for clean navigation
    st_tabs = st.tabs(list(tabs.keys()))
    
    for i, (tab_name, section_fn) in enumerate(tabs.items()):
        with st_tabs[i]:
            section_fn(customer_data, sk)


# Standalone
if __name__ == "__main__":
    st.set_page_config(
        page_title="Barclays Financial Health", page_icon="🏦", layout="wide"
    )
    show_ai_hub()


def _section_proactive_sms(customer: dict, sk: str):
    """Proactive SMS & Recovery Pathway Section - Early Stage Interventions"""
    st.markdown(
        """
    <div class="hub-header">
        <div class="hub-header-left">
            <h2>📱 Proactive SMS & Recovery Pathways</h2>
            <p>Early-stage intervention messages to prevent default • Risk-based personalized recovery options</p>
        </div>
        <span class="hub-badge">Prevention First</span>
    </div>""",
        unsafe_allow_html=True,
    )

    risk_score = customer.get("risk_score", 72)

    # Determine risk tier
    if risk_score >= 80:
        risk_tier = "CRITICAL"
    elif risk_score >= 65:
        risk_tier = "HIGH"
    else:
        risk_tier = "MEDIUM"

    # Risk tier indicator
    tier_colors = {"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "MEDIUM": "#22c55e"}
    tier_color = tier_colors.get(risk_tier, "#6b7280")

    st.markdown(
        f"""
    <div style="background:linear-gradient(135deg, #1e293b, #0f172a); border-radius:12px; padding:1rem; margin:1rem 0; border-left:4px solid {tier_color};">
        <div style="display:flex; align-items:center; gap:1rem;">
            <div style="font-size:2rem;">{"🚨" if risk_tier == "CRITICAL" else "⚠️" if risk_tier == "HIGH" else "✅"}</div>
            <div>
                <div style="color:#94a3b8; font-size:0.8rem;">Customer Risk Level</div>
                <div style="color:{tier_color}; font-size:1.4rem; font-weight:700;">{risk_tier}</div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <div style="color:#94a3b8; font-size:0.8rem;">Risk Score</div>
                <div style="color:{tier_color}; font-size:1.4rem; font-weight:700;">{risk_score}/100</div>
            </div>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    # Generate proactive SMS based on risk tier
    proactive_msgs = _generate_proactive_sms(customer, risk_tier)
    recovery_paths = _generate_recovery_pathway(customer, risk_tier)

    # SMS Templates Section
    st.markdown("#### 📱 Early-Stage SMS Templates")

    sms_tab1, sms_tab2, sms_tab3 = st.tabs(
        ["🔔 Early Warning", "⏰ Payment Reminder", "⚡ Alert"]
    )

    with sms_tab1:
        st.markdown(
            f"""
        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:1rem; margin:0.5rem 0;">
            <div style="color:#166534; font-weight:600; margin-bottom:0.5rem;">📤 Ready to Send</div>
            <div style="background:white; padding:1rem; border-radius:6px; font-size:0.9rem; color:#1e293b; line-height:1.6;">
                {proactive_msgs["early_warning"]}
            </div>
            <div style="margin-top:0.8rem; display:flex; gap:0.5rem;">
                <button style="background:#16a34a; color:white; border:none; padding:0.5rem 1rem; border-radius:4px; cursor:pointer;">📋 Copy</button>
                <button style="background:#0ea5e9; color:white; border:none; padding:0.5rem 1rem; border-radius:4px; cursor:pointer;">📤 Send Now</button>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

    with sms_tab2:
        st.markdown(
            f"""
        <div style="background:#fef3c7; border:1px solid #fde68a; border-radius:8px; padding:1rem; margin:0.5rem 0;">
            <div style="color:#92400e; font-weight:600; margin-bottom:0.5rem;">📤 Ready to Send</div>
            <div style="background:white; padding:1rem; border-radius:6px; font-size:0.9rem; color:#1e293b; line-height:1.6;">
                {proactive_msgs["reminder"]}
            </div>
            <div style="margin-top:0.8rem; display:flex; gap:0.5rem;">
                <button style="background:#d97706; color:white; border:none; padding:0.5rem 1rem; border-radius:4px; cursor:pointer;">📋 Copy</button>
                <button style="background:#0ea5e9; color:white; border:none; padding:0.5rem 1rem; border-radius:4px; cursor:pointer;">📤 Send Now</button>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

    with sms_tab3:
        st.markdown(
            f"""
        <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:1rem; margin:0.5rem 0;">
            <div style="color:#dc2626; font-weight:600; margin-bottom:0.5rem;">📤 Ready to Send</div>
            <div style="background:white; padding:1rem; border-radius:6px; font-size:0.9rem; color:#1e293b; line-height:1.6;">
                {proactive_msgs["alert"]}
            </div>
            <div style="margin-top:0.8rem; display:flex; gap:0.5rem;">
                <button style="background:#dc2626; color:white; border:none; padding:0.5rem 1rem; border-radius:4px; cursor:pointer;">📋 Copy</button>
                <button style="background:#0ea5e9; color:white; border:none; padding:0.5rem 1rem; border-radius:4px; cursor:pointer;">📤 Send Now</button>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

    # Recovery Pathways Section
    st.markdown("---")
    st.markdown("#### 🔄 Recommended Recovery Pathways")

    st.markdown(
        f"""
    <div style="background:linear-gradient(135deg, #1e3a5f, #1e40af); border-radius:12px; padding:1.5rem; margin:1rem 0;">
        <div style="color:#93c5fd; font-size:0.9rem; margin-bottom:0.5rem;">Suggested Approach for <span style="color:white; font-weight:700;">{risk_tier}</span> Risk</div>
        <div style="color:white; font-size:1.2rem; font-weight:600;">{proactive_msgs["pathway_suggestion"]}</div>
        <div style="color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;">Current EMI-to-Income: {recovery_paths["current_emi_ratio"]}</div>
    </div>""",
        unsafe_allow_html=True,
    )

    # Pathway cards
    for i, pathway in enumerate(recovery_paths.get("pathways", [])):
        success_color = (
            "#22c55e"
            if float(pathway.get("success_rate", "0").replace("%", "")) > 80
            else "#f59e0b"
        )

        st.markdown(
            f"""
        <div style="background:white; border:1px solid #e2e8f0; border-radius:10px; padding:1.2rem; margin:0.8rem 0; box-shadow:0 2px 4px rgba(0,0,0,0.05);">
            <div style="display:flex; justify-content:space-between; align-items:start;">
                <div>
                    <div style="display:flex; align-items:center; gap:0.5rem;">
                        <span style="background:#e0f2fe; color:#0369a1; padding:0.2rem 0.6rem; border-radius:20px; font-size:0.75rem; font-weight:600;">Option {i + 1}</span>
                        <span style="font-size:1.1rem; font-weight:700; color:#1e293b;">{pathway.get("name", "N/A")}</span>
                    </div>
                    <div style="color:#64748b; font-size:0.9rem; margin-top:0.5rem;">{pathway.get("description", "")}</div>
                    <div style="margin-top:0.8rem; padding:0.6rem; background:#f8fafc; border-radius:6px;">
                        <div style="color:#0f172a; font-weight:600; font-size:0.85rem;">📊 Impact</div>
                        <div style="color:#16a34a; font-size:0.9rem;">{pathway.get("impact", "N/A")}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="color:#94a3b8; font-size:0.75rem;">Success Rate</div>
                    <div style="color:{success_color}; font-size:1.3rem; font-weight:700;">{pathway.get("success_rate", "N/A")}</div>
                </div>
            </div>
            <div style="margin-top:0.8rem; padding-top:0.8rem; border-top:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center;">
                <div style="color:#64748b; font-size:0.8rem;">
                    <span style="font-weight:600;">Eligibility:</span> {pathway.get("eligibility", "N/A")}
                </div>
                <button style="background:#0f172a; color:white; border:none; padding:0.5rem 1rem; border-radius:6px; cursor:pointer; font-size:0.85rem;">Apply This Pathway</button>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Next Payment Info
    st.markdown("#### 📅 Next Action Items")
    next_actions = [
        (
            "1",
            "Send Early Warning SMS",
            "Immediate",
            "Send proactive message to customer",
        ),
        (
            "2",
            "Schedule Recovery Call",
            "Within 24 hours",
            "Agent to call and discuss pathway options",
        ),
        (
            "3",
            "Prepare Documentation",
            "Within 48 hours",
            "Ready paperwork for chosen pathway",
        ),
        ("4", "Follow-up", "After 7 days", "Check if customer responded to SMS"),
    ]

    for num, action, timeline, desc in next_actions:
        st.markdown(
            f"""
        <div style="display:flex; align-items:center; padding:0.8rem; background:#f8fafc; border-radius:8px; margin:0.4rem 0;">
            <div style="background:#0f172a; color:white; width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.85rem; margin-right:1rem;">{num}</div>
            <div style="flex:1;">
                <div style="font-weight:600; color:#1e293b;">{action}</div>
                <div style="color:#64748b; font-size:0.8rem;">{desc}</div>
            </div>
            <div style="background:#dbeafe; color:#1e40af; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.75rem; font-weight:600;">{timeline}</div>
        </div>""",
            unsafe_allow_html=True,
        )

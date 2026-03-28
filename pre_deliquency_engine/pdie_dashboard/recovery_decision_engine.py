"""
Recovery Decision Engine — Interactive Streamlit Page
Bank-grade, real-time simulation for recovery pathway decisions.

Run standalone: streamlit run recovery_decision_engine.py
Or called from dashboard.py as: from recovery_decision_engine import show_recovery_engine

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from datetime import datetime
from dataclasses import asdict
import sys
from pathlib import Path

# Allow imports from same directory
sys.path.insert(0, str(Path(__file__).parent))

import theme

from npv_library import (
    compute_dicr,
    compute_acr,
    compute_emi,
    compute_amortization,
    compute_pv,
    compute_npv,
    compute_recovery_rate,
    annual_to_monthly_rate,
    monthly_discount_factor,
    compute_capitalized_interest,
    compute_composite_score,
    weighted_average_rate,
)
from scoring_service import (
    compute_default_prob,
    default_prob_series,
    estimate_acceptance,
    estimate_churn_reduction,
    compute_stress_score,
)
from audit_engine import (
    generate_explainability_text,
    generate_short_explanation,
    generate_rm_email_text,
)
from policy_engine import enforce_all_policies
from pathway_simulator import CustomerProfile, simulate_all_pathways, load_engine_config


# ──────────────────────────────────────────────────────
# CSS — Bloomberg × Fintech glassmorphism design
# ──────────────────────────────────────────────────────

RDE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="st-"] { font-family: 'Inter', sans-serif !important; }

/* ── HEADER ── */
.rde-header {
    background: linear-gradient(135deg, #001f5c 0%, #00539B 100%);
    border: 1px solid #00A3E0;
    border-radius: 12px; padding: 1.8rem 2.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 6px 24px rgba(0,83,155,0.4);
    position: relative; overflow: hidden;
}
.rde-header::before {
    content:''; position:absolute; top:-60px; right:-40px;
    width:220px; height:220px; border-radius:50%;
    background: radial-gradient(circle, rgba(0,163,224,0.2) 0%, transparent 70%);
}
.rde-header h1 { color:#ffffff; margin:0 0 0.4rem; font-size:1.8rem; font-weight:800; letter-spacing:-0.5px; }
.rde-header p  { color:#e0f2fe; margin:0; font-size:0.95rem; font-weight:500; }

/* ── CONTROL PANEL ── */
.ctrl-panel {
    background: #0d1b3e;
    border: 1px solid rgba(0,163,224,0.3);
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.ctrl-panel h3 { color:#7dd3fc; margin:0 0 0.8rem; font-size:1rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; }

/* ── METRIC CARDS ── */
.live-metric {
    background: #0a1128;
    border: 1px solid rgba(0,163,224,0.4);
    border-radius: 10px; padding: 1.4rem 1.2rem; text-align:center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    transition: all 0.2s ease;
}
.live-metric:hover { border-color:#00A3E0; transform:translateY(-2px); box-shadow: 0 6px 16px rgba(0,163,224,0.3); }
.live-metric .lm-label { font-size:0.75rem; color:#cbd5e1; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem; font-weight:600; }
.live-metric .lm-value { font-size:2.0rem; font-weight:900; letter-spacing:-0.5px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }
.live-metric .lm-delta { font-size:0.85rem; margin-top:0.4rem; font-weight:700; }
.lm-green  { color:#4ade80; }
.lm-red    { color:#f87171; }
.lm-blue   { color:#38bdf8; }
.lm-yellow { color:#fbbf24; }

/* ── PATHWAY CARDS ── */
.pathway-card {
    background: #0d1b3e;
    border: 1px solid rgba(0,163,224,0.3); border-radius:12px;
    padding:1.4rem 1.6rem; margin-bottom:1rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    transition: all 0.3s ease; cursor:pointer;
}
.pathway-card:hover { border-color:#00A3E0; background:#001f5c; transform:translateX(4px); }
.pathway-card.selected { border-color:#38bdf8; background:rgba(0,163,224,0.2); box-shadow: 0 0 20px rgba(0,163,224,0.4); }
.pathway-card.best { border-color:#4ade80; }
.pc-rank { font-size:0.8rem; font-weight:800; padding:0.3rem 0.8rem; border-radius:20px; display:inline-block; margin-bottom:0.6rem; }
.pc-rank-1 { background:linear-gradient(135deg,#f59e0b,#d97706); color:#000; }
.pc-rank-2 { background:linear-gradient(135deg,#94a3b8,#64748b); color:#fff; }
.pc-rank-3 { background:linear-gradient(135deg,#b45309,#92400e); color:#fff; }
.pc-rank-n { background:rgba(255,255,255,0.15); color:#e2e8f0; }
.pc-title { font-size:1.1rem; font-weight:800; color:#ffffff; }
.pc-benefit { font-size:1rem; font-weight:700; color:#4ade80; margin:0.4rem 0; }
.pc-tag { font-size:0.75rem; padding:0.2rem 0.6rem; border-radius:10px; display:inline-block; margin-right:0.4rem; font-weight:700; }
.tag-best { background:rgba(74,222,128,0.25); color:#4ade80; }
.tag-low  { background:rgba(56,189,248,0.25); color:#38bdf8; }
.tag-save { background:rgba(251,191,36,0.25); color:#fbbf24; }
.tag-warn { background:rgba(248,113,113,0.25); color:#f87171; }
.tag-icr  { background:rgba(167,139,250,0.25); color:#a78bfa; }

/* ── POLICY CHIP ── */
.policy-ok   { background:rgba(74,222,128,0.2); border:1px solid #4ade80; color:#4ade80; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.75rem; font-weight:800; }
.policy-fail { background:rgba(248,113,113,0.2); border:1px solid #f87171; color:#f87171; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.75rem; font-weight:800; }
.policy-warn { background:rgba(251,191,36,0.2); border:1px solid #fbbf24; color:#fbbf24; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.75rem; font-weight:800; }

/* ── SCENARIO CHIP ── */
.scenario-chip { display:inline-flex; align-items:center; gap:0.5rem; padding:0.5rem 1rem; border-radius:20px; font-size:0.85rem; font-weight:800; border:2px solid; cursor:pointer; transition:all 0.2s ease; }

/* ── EXPLAINABILITY ── */
.expl-block {
    background: rgba(255,255,255,0.06); border-left:4px solid #38bdf8;
    border-radius:0 10px 10px 0; padding:1rem 1.2rem; margin:0.6rem 0;
    font-size:0.9rem; line-height:1.6; color:#e2e8f0; font-weight:500;
}
.expl-block strong { color:#7dd3fc; font-weight:700; }
.kpi-highlight { color:#4ade80; font-weight:800; font-size:1.1em; }
.kpi-warn      { color:#fbbf24; font-weight:800; }
.kpi-red       { color:#f87171; font-weight:800; }

/* ── DECISION MODE ── */
.decision-banner {
    background: #1e3a8a;
    border: 1px solid #3b82f6; border-radius: 12px;
    padding: 1.2rem 1.8rem; text-align: center; margin: 1.5rem 0;
    color: #bfdbfe; font-weight: 800; font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(59,130,246,0.2);
}

/* ── OFFER LETTER ── */
.offer-letter {
    background: #0f172a; border: 1px solid rgba(0,163,224,0.5);
    border-radius: 14px; padding: 2rem; font-family:'Inter',monospace;
    color: #e2e8f0; font-size: 0.9rem; line-height: 1.8;
}
.offer-letter h2 { color:#38bdf8; border-bottom:2px solid rgba(56,189,248,0.5); padding-bottom:0.6rem; margin-bottom:1.2rem; font-weight:800; }
.offer-letter .field { color:#94a3b8; font-weight:600; }
.offer-letter .value { color:#ffffff; font-weight:800; }

/* ── DARK SECTION ── */
.dark-section {
    background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 1.4rem;
}

/* ── PROGRESS BAR OVERRIDE ── */
.stProgress > div > div { background: linear-gradient(90deg,#00539B,#00A3E0) !important; }

/* ── STREAMLIT METRIC OVERRIDE ── */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 900 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
}
</style>
"""


# ──────────────────────────────────────────────────────
# HELPER: Live metric compute (fast, no MC)
# ──────────────────────────────────────────────────────


def compute_live_metrics(base: dict, ctrl: dict) -> dict:
    """
    Fast real-time metric compute based on control panel values.
    Returns dict of live metrics for immediate display.
    """
    income = base["income"] * (1 + ctrl["income_shock"])
    expenses = base["expenses"]
    principal = base["principal"]
    rate = base["rate"]
    months = base["months"]
    assets = base["assets"] if ctrl["include_assets"] else 0.0

    # Adjusted EMI
    emi_adj = base["emi"] * (1 - ctrl["emi_reduction"])
    # For ICR mode override
    if ctrl["icr_mode"]:
        emi_adj = max(10000, income * 0.22)

    # Tenure
    new_months = max(6, months + ctrl["tenure_ext"])

    # Recompute EMI on new tenure if extension applied
    if ctrl["tenure_ext"] > 0 and not ctrl["icr_mode"]:
        emi_adj = compute_emi(principal, rate, new_months)
        emi_adj = emi_adj * (1 - ctrl["emi_reduction"])

    emi_adj = max(1000.0, emi_adj)

    # DICR
    dicr = compute_dicr(income, expenses, emi_adj)
    # ACR
    acr = compute_acr(assets, principal)

    # Default prob
    coef = {"a0": -2.0, "a1": -1.5, "a2": -0.8, "a3": 0.5, "a4": 0.3}
    macro = abs(ctrl["income_shock"]) * 2.0  # shock → macro stress
    base_p = compute_default_prob(dicr, acr, 0.15, macro, coef)

    # NPV
    cfs = [emi_adj] * new_months
    probs = default_prob_series(base_p, new_months, 0.97, 0.20)
    npv = compute_npv(cfs, probs, 0.08)
    rr = compute_recovery_rate(npv, principal)

    # Composite
    stress = compute_stress_score(emi_adj, income, assets)
    relief = max(0, 1.0 - emi_adj / max(base["emi"], 1))
    acc_prob = estimate_acceptance(relief, stress, "custom")
    churn_r = estimate_churn_reduction("graduated_emi", relief)
    composite = compute_composite_score(acc_prob, rr, 1 - churn_r)

    # Monthly savings
    savings = base["emi"] - emi_adj
    total_interest = max(0, emi_adj * new_months - principal)

    return {
        "emi": emi_adj,
        "months": new_months,
        "dicr": dicr,
        "acr": acr,
        "npv": npv,
        "recovery_rate": rr,
        "default_prob": base_p,
        "acceptance": acc_prob,
        "churn_reduction": churn_r,
        "composite": composite,
        "savings": savings,
        "total_interest": total_interest,
        "income_used": income,
        "stress": stress,
    }


def fmt(v: float) -> str:
    try:
        import math

        if v is None or math.isnan(float(v)):
            return "₹0"
        return f"₹{float(v):,.0f}"
    except:
        return "₹0"


PATHWAY_TAGS = {
    "emi_holiday": ["⚡ Instant Relief", "🕐 Short-term"],
    "graduated_emi": ["📉 Stepped Down", "🟢 Low Risk"],
    "icr": ["📊 Income-Linked", "🔮 Dynamic"],
    "asset_backed": ["🏦 Asset-Secured", "🟡 Requires FD/MF"],
    "consolidation": ["💰 Max Savings", "🔗 All Debts"],
}

RISK_COLOR = {
    "low": ("#4ade80", "🟢"),
    "medium": ("#fbbf24", "🟡"),
    "high": ("#f87171", "🔴"),
}


def risk_level(default_prob: float) -> str:
    if default_prob < 0.08:
        return "low"
    if default_prob < 0.20:
        return "medium"
    return "high"


# ──────────────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────────────

DARK_BG = "rgba(10,14,26,0.0)"
GRID_COL = "rgba(255,255,255,0.06)"


def dark_layout(fig, height=350, title=""):
    fig.update_layout(
        height=height,
        title=title,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color="#94a3b8", family="Inter"),
        margin=dict(t=40 if title else 20, b=40, l=50, r=20),
        xaxis=dict(
            gridcolor=GRID_COL, linecolor=GRID_COL, tickfont=dict(color="#64748b")
        ),
        yaxis=dict(
            gridcolor=GRID_COL, linecolor=GRID_COL, tickfont=dict(color="#64748b")
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    )
    return fig


def cashflow_chart(base_data, live, scenario_label="Base"):
    months = list(range(1, live["months"] + 1))
    emi_line = [live["emi"]] * len(months)
    income_line = [live["income_used"]] * len(months)
    expenses_line = [base_data["expenses"]] * len(months)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=months,
            y=income_line,
            name="Monthly Income",
            line=dict(color="#38bdf8", width=2, dash="dash"),
            fill=None,
            hovertemplate="Month %{x}: ₹%{y:,.0f}<extra>Income</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=expenses_line,
            name="Essential Expenses",
            line=dict(color="#f87171", width=1.5, dash="dot"),
            fill=None,
            hovertemplate="Month %{x}: ₹%{y:,.0f}<extra>Expenses</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=emi_line,
            name=f"EMI ({scenario_label})",
            line=dict(color="#4ade80", width=3),
            fill="tozeroy",
            fillcolor="rgba(74,222,128,0.08)",
            hovertemplate="Month %{x}: ₹%{y:,.0f}<extra>EMI</extra>",
        )
    )
    # Disposable income band
    disp = [live["income_used"] - base_data["expenses"] - live["emi"]] * len(months)
    fig.add_trace(
        go.Scatter(
            x=months,
            y=[max(0, d) for d in disp],
            name="Disposable",
            line=dict(color="#fbbf24", width=2),
            fill="tozeroy",
            fillcolor="rgba(251,191,36,0.06)",
            hovertemplate="Month %{x}: ₹%{y:,.0f}<extra>Disposable</extra>",
        )
    )
    dark_layout(fig, height=300, title="")
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="₹ Amount",
        yaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def npv_distribution_chart(pathway_results):
    names = [r.display_name for r in pathway_results]
    npvs = [r.npv for r in pathway_results]
    rrs = [min(r.recovery_rate * 100, 150) for r in pathway_results]
    colors = [
        "#4ade80"
        if r.composite_score == max(x.composite_score for x in pathway_results)
        else "#38bdf8"
        for r in pathway_results
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names,
            y=rrs,
            name="Recovery Rate %",
            marker_color=colors,
            text=[f"{v:.0f}%" for v in rrs],
            textposition="auto",
            textfont=dict(color="#000", size=13, weight="bold"),
            hovertemplate="<b>%{x}</b><br>Recovery: %{y:.1f}%<extra></extra>",
        )
    )
    dark_layout(fig, height=260)
    fig.update_layout(yaxis_title="Recovery Rate %", showlegend=False)
    return fig


def composite_radar(results):
    categories = ["Acceptance", "NPV Recovery", "Churn Reduction", "Composite"]
    top3 = results[:3]
    line_colors = ["#4ade80", "#38bdf8", "#fbbf24"]
    fill_colors = [
        "rgba(74,222,128,0.12)",
        "rgba(56,189,248,0.12)",
        "rgba(251,191,36,0.12)",
    ]
    fig = go.Figure()
    for r, line_col, fill_col in zip(top3, line_colors, fill_colors):
        vals = [
            r.acceptance_prob * 100,
            min(r.recovery_rate * 100, 100),
            r.churn_reduction * 100,
            r.composite_score * 100,
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=r.display_name,
                line=dict(color=line_col, width=2),
                fillcolor=fill_col,
            )
        )

    dark_layout(fig, height=300)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100], gridcolor=GRID_COL, linecolor=GRID_COL
            ),
            angularaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    return fig


# ──────────────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────────────


def show_recovery_engine(customer_data: dict | None = None, standalone: bool = False):
    """
    Main entry point.
    customer_data: dict with keys: customer_id, income, expenses, principal, rate,
                   months, emi, assets, other_debts, cibil_score, risk_band
    standalone: True = renders full page config; False = embedded in dashboard
    """
    if standalone:
        st.set_page_config(
            page_title="Recovery Decision Engine", page_icon="⚡", layout="wide"
        )

    st.markdown(RDE_CSS, unsafe_allow_html=True)

    # ── DEFAULT CUSTOMER ──
    if customer_data is None:
        customer_data = {
            "customer_id": "C12345",
            "name": "Demo Customer",
            "income": 85000,
            "expenses": 50000,
            "principal": 500000,
            "rate": 0.14,
            "months": 24,
            "emi": 18500,
            "assets": 450000,
            "other_debts": [],
            "cibil_score": 680,
            "risk_band": "B2",
        }

    # Ensure no NaNs in base data before processing
    import math
    import pandas as pd

    def _s(v, d):
        try:
            val = float(v)
            return d if math.isnan(val) or math.isinf(val) else val
        except:
            return d

    base = {
        "customer_id": str(customer_data.get("customer_id", "C00000")),
        "name": str(customer_data.get("name", "Customer")),
        "income": _s(customer_data.get("income"), 85000.0),
        "expenses": _s(customer_data.get("expenses"), 50000.0),
        "principal": _s(customer_data.get("principal"), 500000.0),
        "rate": _s(customer_data.get("rate"), 0.14),
        "months": int(_s(customer_data.get("months"), 24)),
        "emi": _s(customer_data.get("emi"), 18500.0),
        "assets": _s(customer_data.get("assets"), 450000.0),
        "other_debts": customer_data.get("other_debts", []),
        "cibil_score": int(_s(customer_data.get("cibil_score"), 680)),
        "risk_band": str(customer_data.get("risk_band", "B2")),
        "city": str(customer_data.get("city", "N/A")),
        "total_loan": _s(customer_data.get("total_loan"), 500000.0),
    }
    cid = base["customer_id"]

    # ── SESSION STATE INIT ──
    sk = f"rde_{cid}"
    if f"{sk}_init" not in st.session_state:
        st.session_state[f"{sk}_emi_r"] = 0
        st.session_state[f"{sk}_tenure"] = 0
        st.session_state[f"{sk}_shock"] = 0.0
        st.session_state[f"{sk}_assets"] = True
        st.session_state[f"{sk}_icr"] = False
        st.session_state[f"{sk}_consolidate"] = True
        st.session_state[f"{sk}_selected"] = None
        st.session_state[f"{sk}_decision"] = False
        st.session_state[f"{sk}_sim"] = None
        st.session_state[f"{sk}_init"] = True

    # ── HEADER ──
    dicr_base = compute_dicr(base["income"], base["expenses"], base["emi"])
    acr_base = compute_acr(base["assets"], base["principal"])
    st.markdown(
        f"""
    <div class="rde-header">
        <h1>⚡ Recovery Decision Engine</h1>
        <p>Income <strong style="color:#4ade80">{fmt(base["income"])}</strong> &nbsp;·&nbsp;
           EMI <strong style="color:#38bdf8">{fmt(base["emi"])}</strong> &nbsp;·&nbsp;
           Total Loan <strong style="color:#fbbf24">{fmt(base["total_loan"])}</strong> &nbsp;·&nbsp;
           City <strong style="color:#a78bfa">{base["city"]}</strong> &nbsp;·&nbsp;
           CIBIL <strong>{base.get("cibil_score", 680)}</strong>
        </p>
    </div>""",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════
    # CONTROL PANEL
    # ════════════════════════════════════
    st.markdown("### 🎛️ Recovery Strategy Builder")

    # AI STRATEGY PROMPT BAR (Direct User Input)
    st.markdown(
        '<div style="margin-top: -10px; margin-bottom: 10px; color: #64748b; font-size: 0.85rem;">💬 Describe your scenario or ask the AI agent to compute a strategic intervention.</div>',
        unsafe_allow_html=True,
    )
    user_query = st.text_input(
        "💬 Ask AI Assistant (Custom Intervention / Insights)",
        placeholder="e.g., 'Draft a 15% EMI reduction with 6-mo tenure extension. Is it better than consolidating?'",
        key=f"{sk}_user_prompt",
    )
    if user_query:
        with st.spinner("🤖 Consulting Groq AI Consultant..."):
            from real_ai_engine import generate_response

            ai_prompt = f"""
            You are the Recovery Strategy Consultant for Barclays.
            Customer Profile: {json.dumps(base, default=str)}
            User Instruction/Query: "{user_query}"
            
            Based on the financial coefficients, analyze this proposed strategy.
            Comment on:
            1. NPV Impact (will this trigger massive loss reserves?)
            2. Customer Acceptance Chance
            3. Recommended Verdict (Approve, Refine, or Reject)
            Keep it strictly concise and professional.
            """
            res = generate_response(ai_prompt)
            if res.get("success"):
                st.markdown(
                    f"""
                <div style="background: rgba(14, 165, 233, 0.1); border: 1px solid #0ea5e9; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                    <div style="color: #0ea5e9; font-weight: 800; font-size: 0.9rem; margin-bottom: 0.4rem;">🤖 AI STRATEGY DESK</div>
                    <div style="color: #1e293b; font-size: 0.9rem; line-height: 1.5;">{res["response"]}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.error("Could not reach Groq AI Desk.")

    with st.container():
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
        with c1:
            emi_r_num = st.number_input(
                "EMI Reduction %",
                min_value=0,
                max_value=100,
                value=st.session_state.get(f"{sk}_emi_r", 0),
                step=5,
                key=f"{sk}_emi_r_w",
                help="Reduce customer's monthly EMI burden",
            )
            emi_r = emi_r_num / 100.0 if emi_r_num else 0.0

            tenure_ext = st.number_input(
                "Tenure Extension (months)",
                min_value=0,
                max_value=48,
                value=st.session_state.get(f"{sk}_tenure", 0),
                step=1,
                key=f"{sk}_tenure_w",
            )
        with c2:
            shock_opt = st.selectbox(
                "Income Shock Scenario",
                [
                    "None (Base Case)",
                    "-10% Salary Cut",
                    "-20% Income Drop",
                    "-40% / Job Loss",
                    "+10% Income Growth",
                ],
                key=f"{sk}_shock_sel",
            )
            shock_map = {
                "None (Base Case)": 0.0,
                "-10% Salary Cut": -0.10,
                "-20% Income Drop": -0.20,
                "-40% / Job Loss": -0.40,
                "+10% Income Growth": 0.10,
            }
            income_shock = shock_map[shock_opt]
            rate_adj_num = st.number_input(
                "Interest Rate Adjustment (%Pa)",
                min_value=-10,
                max_value=10,
                value=0,
                step=1,
                key=f"{sk}_rate_adj",
                help="Simulate rate renegotiation",
            )
            rate_adj = rate_adj_num / 100.0 if rate_adj_num else 0.0
        with c3:
            include_assets = st.toggle(
                "🏦 Include Liquid Assets",
                value=st.session_state.get(f"{sk}_assets", True),
                key=f"{sk}_assets_w",
            )
            icr_mode = st.toggle(
                "📊 ICR Mode (22% income)",
                value=st.session_state.get(f"{sk}_icr", False),
                key=f"{sk}_icr_w",
            )
            consolidate = st.toggle(
                "🔗 Consolidate Debts",
                value=st.session_state.get(f"{sk}_consolidate", True),
                key=f"{sk}_cons_w",
            )
        with c4:
            decision_mode = st.toggle(
                "🎯 Decision Mode",
                value=st.session_state.get(f"{sk}_decision", False),
                key=f"{sk}_dec_w",
            )
            mc_fast = st.selectbox("MC Runs (ICR)", [500, 1000, 5000], key=f"{sk}_mc")
            if st.button("🔄 Reset Controls", key=f"{sk}_reset"):
                # Reinitialize to defaults — do NOT delete, since widget value=
                # reads happen BEFORE the init guard on the next rerun.
                st.session_state[f"{sk}_emi_r"] = 0
                st.session_state[f"{sk}_tenure"] = 0
                st.session_state[f"{sk}_shock"] = 0.0
                st.session_state[f"{sk}_assets"] = True
                st.session_state[f"{sk}_icr"] = False
                st.session_state[f"{sk}_consolidate"] = True
                st.session_state[f"{sk}_decision"] = False
                st.session_state[f"{sk}_sim"] = None
                st.rerun()

    # ── Effective rate ──
    eff_rate = max(0.06, base.get("rate", 0.14) + rate_adj)
    ctrl = {
        "emi_reduction": emi_r,
        "tenure_ext": tenure_ext,
        "income_shock": income_shock,
        "include_assets": include_assets,
        "icr_mode": icr_mode,
        "rate": eff_rate,
    }

    # ── LIVE METRICS (fast, always visible) ──
    live = compute_live_metrics({**base, "rate": eff_rate}, ctrl)

    st.markdown("---")
    st.markdown("#### 📊 Live Simulation Metrics")

    prev_emi = base["emi"]
    prev_dicr = compute_dicr(base["income"], base["expenses"], base["emi"])

    def delta_html(new, old, fmt_fn, higher_is_better=True):
        diff = new - old
        if abs(diff) < 0.001:
            return '<span style="color:#64748b">—</span>'
        arrow = "↑" if diff > 0 else "↓"
        col = "#4ade80" if (diff > 0) == higher_is_better else "#f87171"
        return f'<span style="color:{col}">{arrow} {fmt_fn(abs(diff))}</span>'

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    metrics = [
        (
            m1,
            "Monthly EMI",
            fmt(live["emi"]),
            "lm-green" if live["emi"] < prev_emi else "lm-blue",
            delta_html(live["emi"], prev_emi, fmt, False),
        ),
        (
            m2,
            "DICR",
            f"{live['dicr']:.2f}x",
            "lm-green"
            if live["dicr"] > 1.5
            else "lm-yellow"
            if live["dicr"] > 1.0
            else "lm-red",
            delta_html(live["dicr"], prev_dicr, lambda x: f"{x:.2f}x"),
        ),
        (
            m3,
            "NPV Recovery",
            f"{live['recovery_rate'] * 100:.1f}%",
            "lm-green"
            if live["recovery_rate"] > 0.85
            else "lm-yellow"
            if live["recovery_rate"] > 0.65
            else "lm-red",
            delta_html(live["recovery_rate"], 0.85, lambda x: f"{x * 100:.1f}%"),
        ),
        (
            m4,
            "Default Prob",
            f"{live['default_prob'] * 100:.1f}%",
            "lm-green"
            if live["default_prob"] < 0.08
            else "lm-yellow"
            if live["default_prob"] < 0.20
            else "lm-red",
            delta_html(live["default_prob"], 0.08, lambda x: f"{x * 100:.1f}%", False),
        ),
        (
            m5,
            "Monthly Savings",
            fmt(max(0, live["savings"])),
            "lm-green" if live["savings"] > 0 else "lm-red",
            "",
        ),
        (m6, "Composite Score", f"{live['composite']:.3f}", "lm-blue", ""),
    ]
    for col, label, value, css_cls, delta in metrics:
        with col:
            st.markdown(
                f"""
            <div class="live-metric">
                <div class="lm-label">{label}</div>
                <div class="lm-value {css_cls}">{value}</div>
                <div class="lm-delta">{delta}&nbsp;</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ════════════════════════════════════
    # RUN FULL SIMULATION (v2.0 engine)
    # ════════════════════════════════════
    with st.spinner("⚙️ Running pathway simulations..."):
        config = load_engine_config()
        config["mc_runs"] = mc_fast
        config["discount_rate"] = 0.08

        pathways_to_run = ["emi_holiday", "graduated_emi", "icr", "asset_backed"]
        if consolidate and base.get("other_debts"):
            pathways_to_run.append("consolidation")

        customer = CustomerProfile(
            customer_id=cid,
            name=base.get("name", "Customer"),
            monthly_income=base["income"] * (1 + income_shock),
            essential_expenses=base["expenses"],
            principal=base["principal"],
            annual_rate=eff_rate,
            remaining_months=base["months"] + tenure_ext,
            emi=base["emi"],
            total_liquid_assets=base["assets"] if include_assets else 0.0,
            other_debts=base.get("other_debts", []),
            cibil_score=base.get("cibil_score", 680),
        )

        sim = simulate_all_pathways(customer, config, pathways_to_run)
        results = sim.results

    # ════════════════════════════════════
    # SECTION: SCENARIO COMPARISON BAR
    # ════════════════════════════════════
    rl = risk_level(live["default_prob"])
    rc, ri = RISK_COLOR[rl]
    scenario_color = {
        "None (Base Case)": "#4ade80",
        "-10% Salary Cut": "#a3e635",
        "-20% Income Drop": "#fbbf24",
        "-40% / Job Loss": "#f87171",
        "+10% Income Growth": "#818cf8",
    }[shock_opt]

    st.markdown(
        f"""
    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                border-radius:12px; padding:0.8rem 1.2rem; display:flex; align-items:center; gap:1.5rem; margin-bottom:1rem;">
        <span style="font-size:0.8rem; color:#64748b; font-weight:600;">ACTIVE SCENARIO</span>
        <span style="background:{scenario_color}22; border:1px solid {scenario_color}55; color:{scenario_color};
                     padding:0.3rem 0.9rem; border-radius:20px; font-weight:700; font-size:0.85rem;">{shock_opt}</span>
        <span style="font-size:0.8rem; color:#64748b;">Risk Level:</span>
        <span style="color:{rc}; font-weight:700;">{ri} {rl.upper()}</span>
        <span style="font-size:0.8rem; color:#64748b;">Recommended:</span>
        <span style="color:#38bdf8; font-weight:700;">{sim.recommended.replace("_", " ").title() if sim.recommended else "N/A"}</span>
    </div>""",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════
    # TABS: PATHWAYS | CHARTS | DECISION | OUTPUT
    # ════════════════════════════════════
    tab_cards, tab_charts, tab_decision, tab_output = st.tabs(
        [
            "🃏 Pathway Cards",
            "📈 Charts & Analysis",
            "🎯 Decision Mode",
            "🏦 Bank Output",
        ]
    )

    # ────────────────────
    # TAB 1: PATHWAY CARDS
    # ────────────────────
    with tab_cards:
        st.markdown("##### Click a pathway to explore its full simulation →")

        for rank, r in enumerate(results, 1):
            rank_cls = {1: "pc-rank-1", 2: "pc-rank-2", 3: "pc-rank-3"}.get(
                rank, "pc-rank-n"
            )
            rank_lbl = {1: "🥇 BEST", 2: "🥈 #2", 3: "🥉 #3"}.get(rank, f"#{rank}")
            is_rec = r.pathway_name == sim.recommended
            tags_html = "".join(
                f'<span class="pc-tag tag-{"best" if "Best" in t or "Max" in t or "Low" in t else "save" if "₹" in t else "low"}">{t}</span>'
                for t in PATHWAY_TAGS.get(r.pathway_name, [])
            )
            policy_ok = r.policy_result.get("passed", True) if r.policy_result else True
            policy_html = (
                f'<span class="policy-ok">✅ Policy OK</span>'
                if policy_ok
                else f'<span class="policy-fail">⚠️ Policy Flag</span>'
            )

            savings_str = (
                f"Save {fmt(r.monthly_savings)}/mo"
                if r.monthly_savings > 0
                else f"EMI: {fmt(r.new_emi)}/mo"
            )
            card_border = "style='border-color:#4ade8866;'" if is_rec else ""

            with st.expander(
                f"{'⭐' if is_rec else '  '} #{rank}  {r.display_name}  |  Score: {r.composite_score:.3f}  |  {savings_str}",
                expanded=(rank == 1),
            ):
                left, right = st.columns([3, 2])
                with left:
                    st.markdown(
                        f"""
                    <div style="margin-bottom:0.8rem;">
                        <span class="pc-rank {rank_cls}">{rank_lbl}</span>
                        {policy_html}
                        {'<span class="pc-tag tag-best">⭐ RECOMMENDED</span>' if is_rec else ""}
                    </div>
                    <div style="font-size:0.95rem; color:#334155; margin-bottom:0.6rem; font-weight:500;">{r.description}</div>
                    <div style="font-size:0.9rem; color:#0f172a; margin-bottom:0.8rem; font-weight:600;">📋 {r.action}</div>
                    <div class="pc-benefit">💰 {r.immediate_relief}</div>
                    <div style="margin-top:0.8rem;">{tags_html}</div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Policy violations
                    if r.policy_result and not policy_ok:
                        violations = r.policy_result.get("violations", [])
                        for v in violations:
                            st.markdown(
                                f'<div class="expl-block" style="color:#0f172a; border-left-color:#f87171; background:rgba(248,113,113,0.1);">⚠️ <strong>Policy:</strong> {v}</div>',
                                unsafe_allow_html=True,
                            )

                    # Short explainability
                    if r.short_explanation:
                        st.markdown(
                            f'<div class="expl-block" style="color:#0f172a; background:rgba(0,163,224,0.05); border-left-color:#00A3E0;">💡 {r.short_explanation}</div>',
                            unsafe_allow_html=True,
                        )

                with right:

                    def safe_val(v):
                        try:
                            import math

                            if v is None or math.isnan(float(v)):
                                return 0.0
                            return min(max(float(v), 0.0), 1.0)
                        except:
                            return 0.0

                    metrics_html = f"""
                    <div style="display:flex; justify-content:space-between; margin-bottom:1.5rem; background:#f8fafc; padding:1.2rem; border-radius:12px; border:1px solid #e2e8f0; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                        <div style="text-align:center;">
                            <div style="font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase; margin-bottom:0.3rem;">New EMI</div>
                            <div style="font-size:1.5rem; color:#00539B; font-weight:800; letter-spacing:-0.5px;">{fmt(r.new_emi)}</div>
                        </div>
                        <div style="text-align:center; border-left:1px solid #e2e8f0; border-right:1px solid #e2e8f0; padding:0 1.5rem;">
                            <div style="font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase; margin-bottom:0.3rem;">Tenure</div>
                            <div style="font-size:1.5rem; color:#00539B; font-weight:800; letter-spacing:-0.5px;">{r.new_tenure_months} <span style="font-size:1rem;">mo</span></div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase; margin-bottom:0.3rem;">Recovery</div>
                            <div style="font-size:1.5rem; color:#00A3E0; font-weight:800; letter-spacing:-0.5px;">{safe_val(r.recovery_rate) * 100:.0f}%</div>
                        </div>
                    </div>
                    """
                    st.markdown(metrics_html, unsafe_allow_html=True)

                    st.progress(
                        safe_val(r.acceptance_prob),
                        text=f"Acceptance: {safe_val(r.acceptance_prob) * 100:.1f}%",
                    )
                    st.progress(
                        safe_val(r.recovery_rate),
                        text=f"NPV Recovery: {safe_val(r.recovery_rate) * 100:.1f}%",
                    )
                    st.progress(
                        safe_val(r.churn_reduction),
                        text=f"Churn Reduction: {safe_val(r.churn_reduction) * 100:.1f}%",
                    )

                # Monte Carlo (ICR)
                if r.mc_result:
                    st.markdown("---")
                    st.markdown(
                        "<strong style='color:#0f172a'>🎲 Monte Carlo Results (ICR Income Simulation)</strong>",
                        unsafe_allow_html=True,
                    )
                    mc = r.mc_result

                    mc_html = f"""
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem; background:#f1f5f9; padding:1rem; border-radius:10px; border:1px solid #cbd5e1;">
                        <div style="text-align:center;">
                            <div style="font-size:0.7rem; color:#475569; font-weight:700; text-transform:uppercase;">Mean NPV</div>
                            <div style="font-size:1.3rem; color:#0f172a; font-weight:800;">{fmt(mc["mean_npv"])}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.7rem; color:#475569; font-weight:700; text-transform:uppercase;">Std Dev</div>
                            <div style="font-size:1.3rem; color:#0f172a; font-weight:800;">{fmt(mc["std_npv"])}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.7rem; color:#475569; font-weight:700; text-transform:uppercase;">5th Pctl</div>
                            <div style="font-size:1.3rem; color:#0f172a; font-weight:800;">{fmt(mc["p5"])}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.7rem; color:#475569; font-weight:700; text-transform:uppercase;">95th Pctl</div>
                            <div style="font-size:1.3rem; color:#0f172a; font-weight:800;">{fmt(mc["p95"])}</div>
                        </div>
                    </div>
                    """
                    st.markdown(mc_html, unsafe_allow_html=True)
                    st.caption(
                        f"📊 {mc['n_runs']:,} simulations · Seed: {mc['seed']} · P(recovery>70%) = **{mc['prob_above_threshold'] * 100:.1f}%**"
                    )

                # Audit detail
                with st.expander("📋 Audit & Explainability Trail"):
                    st.markdown(f"**Model Explanation:**\n\n{r.explainability}")
                    if r.audit:
                        st.caption(
                            f"Simulation ID: `{r.audit.get('simulation_id', 'N/A')}` | Timestamp: {r.audit.get('timestamp', 'N/A')} | Version: {r.audit.get('model_version', '2.0.0')}"
                        )
                        if st.button(
                            f"📥 Export Audit JSON", key=f"audit_{r.pathway_name}_{cid}"
                        ):
                            st.download_button(
                                "Download",
                                data=str(r.audit),
                                file_name=f"audit_{r.pathway_name}_{cid}.json",
                                mime="application/json",
                                key=f"dl_{r.pathway_name}",
                            )

                # CTA buttons
                bt1, bt2, bt3 = st.columns(3)
                with bt1:
                    if st.button(
                        f"✅ Offer Now", key=f"offer_{rank}_{cid}", type="primary"
                    ):
                        st.session_state[f"{sk}_selected"] = r.pathway_name
                        # Store for AI Hub message flow (step 3 recovery message)
                        hub_sk = f"hub_{cid}"
                        st.session_state[f"{hub_sk}_pathway_name"] = r.pathway_name
                        st.session_state[f"{hub_sk}_pathway_details"] = {
                            "new_emi": float(r.new_emi),
                            "monthly_savings": float(r.monthly_savings),
                            "new_tenure_months": int(r.new_tenure_months),
                            "recovery_rate": float(r.recovery_rate),
                            "display_name": r.display_name,
                        }
                        st.success(
                            f"✅ '{r.display_name}' selected! Go to **🤖 AI Hub** → send Step 3 recovery message to customer."
                        )
                with bt2:
                    if st.button(f"📋 Save Draft", key=f"draft_{rank}_{cid}"):
                        st.session_state[f"{sk}_draft"] = r
                        st.info("Draft saved.")
                with bt3:
                    if st.button(f"⚖️ Compare", key=f"compare_{rank}_{cid}"):
                        st.session_state[f"{sk}_compare"] = r.pathway_name

    # ────────────────────
    # TAB 2: CHARTS
    # ────────────────────
    with tab_charts:
        col_cf, col_rad = st.columns([3, 2])

        with col_cf:
            st.markdown("##### 📊 Monthly Cash Flow — Income vs EMI vs Expenses")
            best_r = results[0] if results else None
            live_for_chart = dict(live)
            if best_r:
                live_for_chart["emi"] = best_r.new_emi
                live_for_chart["months"] = best_r.new_tenure_months
            fig_cf = cashflow_chart(base, live_for_chart, shock_opt)
            st.plotly_chart(fig_cf, use_container_width=True)

        with col_rad:
            st.markdown("##### 🕸️ Top-3 Pathway Comparison (Radar)")
            if len(results) >= 2:
                fig_r = composite_radar(results)
                st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("---")

        col_bar, col_stress = st.columns(2)

        with col_bar:
            st.markdown("##### 📈 Recovery Rate Across Pathways")
            fig_b = npv_distribution_chart(results)
            st.plotly_chart(fig_b, use_container_width=True)

        with col_stress:
            st.markdown("##### 🌊 Stress Scenario Impact on Composite Score")
            shocks = [0.0, -0.10, -0.20, -0.40]
            shock_labels = ["Base", "-10%", "-20%", "-40%(Job Loss)"]
            scenario_colors = ["#4ade80", "#a3e635", "#fbbf24", "#f87171"]
            if results:
                best_pw = results[0]
                comp_by_shock = []
                for sh in shocks:
                    lv = compute_live_metrics(
                        {**base, "rate": eff_rate}, {**ctrl, "income_shock": sh}
                    )
                    comp_by_shock.append(lv["composite"] * 100)
                fig_stress = go.Figure()
                fig_stress.add_trace(
                    go.Bar(
                        x=shock_labels,
                        y=comp_by_shock,
                        marker_color=scenario_colors,
                        text=[f"{v:.1f}" for v in comp_by_shock],
                        textposition="auto",
                        textfont=dict(color="#000", weight="bold"),
                    )
                )
                fig_stress.add_hline(
                    y=60,
                    line_dash="dot",
                    line_color="#fbbf24",
                    annotation_text="Min threshold (60)",
                )
                dark_layout(fig_stress, height=260)
                fig_stress.update_layout(
                    yaxis_title="Composite Score", showlegend=False
                )
                st.plotly_chart(fig_stress, use_container_width=True)

        # ── 12-Month Amortization preview ──
        st.markdown("---")
        st.markdown("##### 📅 Amortization Schedule Preview (Best Pathway)")
        if results:
            br = results[0]
            try:
                _, sched = compute_amortization(
                    base["principal"], eff_rate, br.new_tenure_months
                )
                preview = sched[: min(12, len(sched))]
                import pandas as pd

                df_sched = pd.DataFrame(preview)
                df_sched.columns = [
                    "Month",
                    "EMI (₹)",
                    "Interest (₹)",
                    "Principal Paid (₹)",
                    "Balance (₹)",
                ]
                for c in [
                    "EMI (₹)",
                    "Interest (₹)",
                    "Principal Paid (₹)",
                    "Balance (₹)",
                ]:
                    df_sched[c] = df_sched[c].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(df_sched, use_container_width=True, height=300)
            except Exception:
                st.info("Amortization preview unavailable for this configuration.")

    # ────────────────────
    # TAB 3: DECISION MODE
    # ────────────────────
    with tab_decision:
        st.markdown(
            """
        <div class="decision-banner">
            🎯 Decision Mode — Compare pathways side-by-side and deploy the winning plan
        </div>""",
            unsafe_allow_html=True,
        )

        if len(results) < 2:
            st.warning("Need at least 2 pathways to compare.")
        else:
            names_avail = [r.display_name for r in results]
            compare_picks = st.multiselect(
                "Select 2–3 pathways to compare",
                names_avail,
                default=names_avail[: min(3, len(names_avail))],
                max_selections=3,
                key=f"{sk}_cmp_sel",
            )
            compare_rs = [r for r in results if r.display_name in compare_picks]

            if compare_rs:
                # Side-by-side metric table
                import pandas as pd

                rows = {
                    "New EMI (₹)": [fmt(r.new_emi) for r in compare_rs],
                    "Tenure (mo)": [str(r.new_tenure_months) for r in compare_rs],
                    "Recovery Rate": [
                        f"{min(r.recovery_rate * 100, 99):.1f}%" for r in compare_rs
                    ],
                    "Acceptance Prob": [
                        f"{r.acceptance_prob * 100:.1f}%" for r in compare_rs
                    ],
                    "Churn Reduction": [
                        f"{r.churn_reduction * 100:.1f}%" for r in compare_rs
                    ],
                    "Composite Score": [f"{r.composite_score:.4f}" for r in compare_rs],
                    "Monthly Savings": [
                        fmt(max(0, r.monthly_savings)) for r in compare_rs
                    ],
                    "Total Interest": [fmt(r.total_interest) for r in compare_rs],
                    "Policy OK": [
                        "✅" if (r.policy_result or {}).get("passed", True) else "❌"
                        for r in compare_rs
                    ],
                }
                df_cmp = pd.DataFrame(
                    rows, index=[r.display_name for r in compare_rs]
                ).T
                st.dataframe(df_cmp, use_container_width=True)

                # Live ranking chart
                st.markdown("##### 📊 Live Composite Ranking")
                fig_dec = go.Figure()
                dec_colors = ["#4ade80", "#38bdf8", "#fbbf24"]
                for r, col in zip(compare_rs, dec_colors):
                    fig_dec.add_trace(
                        go.Bar(
                            x=[r.display_name],
                            y=[r.composite_score * 100],
                            name=r.display_name,
                            marker_color=col,
                            text=[f"{r.composite_score * 100:.1f}"],
                            textposition="auto",
                        )
                    )
                dark_layout(fig_dec, height=250)
                fig_dec.update_layout(
                    barmode="group", yaxis_title="Composite Score", showlegend=False
                )
                st.plotly_chart(fig_dec, use_container_width=True)

                # Deploy button
                st.markdown("---")
                winner = max(compare_rs, key=lambda r: r.composite_score)
                st.markdown(
                    f"**🏆 Recommended Winner:** `{winner.display_name}` (Score: {winner.composite_score:.4f})"
                )
                deploy_col1, deploy_col2 = st.columns([2, 1])
                with deploy_col1:
                    st.markdown(
                        f'<div class="expl-block">💡 {winner.short_explanation}</div>',
                        unsafe_allow_html=True,
                    )
                with deploy_col2:
                    if st.button(
                        f"🚀 Deploy Plan: {winner.display_name}",
                        type="primary",
                        key=f"{sk}_deploy",
                    ):
                        st.session_state[f"{sk}_selected"] = winner.pathway_name
                        st.success(
                            f"🎯 Plan deployed! Offer queued for `{winner.display_name}`."
                        )

    # ────────────────────
    # TAB 4: BANK OUTPUT
    # ────────────────────
    with tab_output:
        selected_pname = st.session_state.get(f"{sk}_selected")
        selected_r = next(
            (r for r in results if r.pathway_name == selected_pname),
            results[0] if results else None,
        )

        if not selected_r:
            st.info("Run a simulation and select a pathway to generate bank output.")
        else:
            st.markdown(f"**Generating offer for: `{selected_r.display_name}`**")

            now = datetime.now()
            config_used = load_engine_config()

            # RM Email
            st.markdown("##### 📧 Relationship Manager Brief")
            rm_text = generate_rm_email_text(
                selected_r.pathway_name,
                {**base, "name": base.get("name", cid)},
                {
                    "new_emi": selected_r.new_emi,
                    "monthly_savings": selected_r.monthly_savings,
                    "recovery_rate": selected_r.recovery_rate,
                    "composite": selected_r.composite_score,
                    "total_interest": selected_r.total_interest,
                    "npv": selected_r.npv,
                },
                config_used,
            )
            st.text_area(
                "RM Email (editable)", value=rm_text, height=260, key=f"{sk}_rm_email"
            )

            st.download_button(
                "📥 Download RM Brief",
                rm_text,
                file_name=f"rm_brief_{cid}_{selected_r.pathway_name}.txt",
                mime="text/plain",
                key=f"{sk}_dl_rm",
            )

            st.markdown("---")

            # Formal Offer Letter
            st.markdown("##### 🏦 Formal Recovery Offer Letter")
            offer_id = f"OFF-{cid[-4:]}-{now.strftime('%Y%m%d%H%M')}"
            policy_violations = (selected_r.policy_result or {}).get("violations", [])
            conditions_html = "\n".join(
                f"  • {v}"
                for v in [
                    "Quarterly income verification check",
                    "No new unsecured credit for 12 months",
                    "EMI to be paid by due date each month",
                    "Subject to bank policy approval",
                ]
            )

            offer_letter = f"""
BARCLAYS BANK — RECOVERY PATHWAY OFFER
═══════════════════════════════════════════

Offer ID   : {offer_id}
Date       : {now.strftime("%d-%m-%Y")}
Customer   : {cid}
Pathway    : {selected_r.display_name}
Status     : DRAFT — Pending RM Approval

── FINANCIAL TERMS ─────────────────────────
New Monthly EMI    : {fmt(selected_r.new_emi)}
New Tenure         : {selected_r.new_tenure_months} months
Total Interest     : {fmt(selected_r.total_interest)}
Monthly Savings    : {fmt(max(0, selected_r.monthly_savings))}

── RECOVERY METRICS ─────────────────────────
NPV                : {fmt(selected_r.npv)}
Recovery Rate      : {min(selected_r.recovery_rate * 100, 99):.1f}%
Acceptance Prob    : {selected_r.acceptance_prob * 100:.1f}%
Composite Score    : {selected_r.composite_score:.4f}
Policy Status      : {"PASS ✅" if not policy_violations else "FLAG ⚠️ — " + "; ".join(policy_violations)}

── CONDITIONS ───────────────────────────────
{conditions_html}

── MODEL PROVENANCE ─────────────────────────
Model Version      : {config_used.get("model_version", "2.0.0")}
Simulation Run     : {selected_r.audit.get("simulation_id", "N/A") if selected_r.audit else "N/A"}
Timestamp          : {now.isoformat()}
Discount Rate      : {config_used.get("discount_rate", 0.08) * 100:.0f}% p.a.

═══════════════════════════════════════════
This document is system-generated by PDIE Recovery Engine v2.0
For compliance queries contact: pdie-compliance@barclays.com
"""
            st.markdown(
                f'<div class="offer-letter"><pre>{offer_letter}</pre></div>',
                unsafe_allow_html=True,
            )

            st.download_button(
                "📄 Download Offer Letter",
                offer_letter,
                file_name=f"offer_{offer_id}.txt",
                mime="text/plain",
                key=f"{sk}_dl_offer",
            )

            # Repayment schedule download
            st.markdown("---")
            st.markdown("##### 📅 Full Repayment Schedule")
            try:
                import pandas as pd

                _, full_sched = compute_amortization(
                    base["principal"], eff_rate, selected_r.new_tenure_months
                )
                df_full = pd.DataFrame(full_sched)
                df_full.columns = [
                    "Month",
                    "EMI",
                    "Interest",
                    "Principal_Paid",
                    "Balance",
                ]
                st.dataframe(
                    df_full.style.format(
                        {
                            "EMI": "₹{:,.0f}",
                            "Interest": "₹{:,.0f}",
                            "Principal_Paid": "₹{:,.0f}",
                            "Balance": "₹{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                    height=350,
                )
                csv = df_full.to_csv(index=False)
                st.download_button(
                    "📥 Download Schedule CSV",
                    csv,
                    file_name=f"schedule_{cid}_{selected_r.pathway_name}.csv",
                    mime="text/csv",
                    key=f"{sk}_dl_sched",
                )
            except Exception as e:
                st.warning(f"Schedule unavailable: {e}")


# ──────────────────────────────────────────────────────
# STANDALONE ENTRY
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    show_recovery_engine(standalone=True)

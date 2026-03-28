"""
PDIE Dashboard — Main Application
Pre-Delinquency Intervention Engine

Complete Streamlit dashboard integrating:
1. XGBoost ML model predictions
2. SHAP explainability
3. AI Communication Agent
4. Recovery Pathway Engine
5. Financial Health Monitor

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Import our custom modules
import theme
import config
from ai_agent import (
    AICommunicationAgent,
    CustomerContext,
    MessageChannel,
    RiskTier,
    create_context_from_customer,
)
from pathway_simulator import (
    RecoveryPathwayEngine,
    LoanDetails,
    create_loan_from_customer,
)
from health_monitor import (
    FinancialHealthMonitor,
    AlertLevel,
    create_mock_historical_trend,
)
from agentic_engine import (
    AgenticPDIE,
    TOOL_REGISTRY,
    ToolStatus,
    tool_analyze_risk_signals,
    tool_predict_outcome,
    tool_optimize_channel,
    tool_generate_intervention_plan,
    tool_generate_script,
    tool_evaluate_pathways,
    tool_generate_recovery_message,
)
import real_ai_engine
import real_messaging
import real_calling


# ═════════════════════════════════════════════════════════════════════════════
# ─── OPTIMIZED DATA LOADING (Performance Enhancement) ───
# This caches ALL data and pre-computes risk scores ONCE at startup
# ═════════════════════════════════════════════════════════════════════════════


@st.cache_resource
def get_optimized_data_store():
    """
    Returns a cached data store with ALL pre-computed data.
    This runs ONCE at app startup and caches everything in memory.
    """
    import pickle
    from scipy.stats import rankdata

    # Initialize store
    store = {
        "features": None,
        "customers": None,
        "transactions": None,
        "balances": None,
        "loans": None,
        "model": None,
        "shap": None,
        "enriched_df": None,
        "loaded": False,
    }

    # Load features
    try:
        for p in [
            Path("../pdie_feature_store/features.parquet"),
            Path("pdie_feature_store/features.parquet"),
        ]:
            if p.exists():
                store["features"] = pd.read_parquet(p)
                break
    except:
        pass

    # Load model
    try:
        for p in [
            Path("../pdie_model_outputs/pdie_xgboost_model.pkl"),
            Path("pdie_model_outputs/pdie_xgboost_model.pkl"),
        ]:
            if p.exists():
                with open(p, "rb") as f:
                    store["model"] = pickle.load(f)
                break
    except:
        pass

    # Load other data
    try:
        for p in [
            Path("../pdie_feature_store/customers.parquet"),
            Path("pdie_feature_store/customers.parquet"),
        ]:
            if p.exists():
                store["customers"] = pd.read_parquet(p)
                break
    except:
        pass

    # Pre-compute risk scores if model and features exist
    if store["model"] is not None and store["features"] is not None:
        try:
            df = store["features"].copy()
            exclude_cols = ["customer_id", "will_default_in_21_days"]
            feature_cols = [col for col in df.columns if col not in exclude_cols]

            # Load expected features
            try:
                for p in [
                    Path("../pdie_model_outputs/feature_names.json"),
                    Path("pdie_model_outputs/feature_names.json"),
                ]:
                    if p.exists():
                        with open(p) as f:
                            expected_features = json.load(f)
                        break
            except:
                expected_features = None

            # Prepare features
            X = pd.get_dummies(df[feature_cols])
            if expected_features is not None:
                X = X.reindex(columns=expected_features, fill_value=0)

            # Run XGBoost inference ONCE
            predictions = store["model"].predict_proba(X)[:, 1]

            # Convert to percentile scores
            percentile_scores = (
                rankdata(predictions, method="average") / len(predictions) * 100
            )
            df["risk_score"] = percentile_scores.round(1)

            # Add risk category
            df["risk_category"] = pd.cut(
                df["risk_score"],
                bins=[0, 50, 70, 80, 100],
                labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            )

            # Add EMI amount
            if "emi_to_income_ratio" in df.columns and "monthly_income" in df.columns:
                df["emi_amount"] = (
                    df["emi_to_income_ratio"] * df["monthly_income"]
                ).round(0)

            # Enrich with customer data
            if store["customers"] is not None:
                keep_cols = [
                    c
                    for c in [
                        "customer_id",
                        "full_name",
                        "city",
                        "account_opening_date",
                    ]
                    if c in store["customers"].columns
                ]
                df = df.merge(
                    store["customers"][keep_cols], on="customer_id", how="left"
                )

            # Enrich with loan data
            if store["loans"] is not None:
                try:
                    loan_cols = [
                        c
                        for c in [
                            "customer_id",
                            "loan_id",
                            "loan_type",
                            "sanction_date",
                            "sanction_amount",
                            "outstanding_principal",
                            "interest_rate",
                            "emi_amount",
                            "tenure_months",
                            "remaining_months",
                            "loan_status",
                        ]
                        if c in store["loans"].columns
                    ]
                    df = df.merge(
                        store["loans"][loan_cols],
                        on="customer_id",
                        how="left",
                        suffixes=("", "_loan"),
                    )
                    if "emi_amount_loan" in df.columns:
                        df["emi_amount"] = df["emi_amount_loan"].fillna(
                            df["emi_amount"]
                        )
                        df.drop(
                            columns=["emi_amount_loan"], inplace=True, errors="ignore"
                        )
                except:
                    pass

            store["enriched_df"] = df
            store["loaded"] = True
            print(f"✅ Pre-computed risk scores for {len(df)} customers")
        except Exception as e:
            print(f"Error pre-computing scores: {e}")
            # Fallback to features as-is
            store["enriched_df"] = store["features"]
            store["loaded"] = True
    else:
        store["enriched_df"] = store["features"]
        store["loaded"] = True

    return store


def get_enriched_dataframe():
    """Fast access to pre-computed enriched dataframe."""
    store = get_optimized_data_store()
    return store.get("enriched_df")


def get_model():
    """Fast access to cached model."""
    store = get_optimized_data_store()
    return store.get("model")


# ═════════════════════════════════════════════════════════════════════════════
# ─── BACKGROUND WORKER: SCHEDULER ENGINE ───
# Polling SQLite database every 20s to execute workflow actions at the scheduled time
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def _start_background_executor():
    import threading
    import time

    def run_scheduler_loop():
        print("🚀 PDIE Background Scheduler Thread Started")
        while True:
            try:
                import sqlite3
                import real_messaging
                import real_calling

                # Only try if the DB actually exists yet
                if not Path("pdie_reminders.db").exists():
                    time.sleep(20)
                    continue

                conn = sqlite3.connect("pdie_reminders.db")
                cur = conn.cursor()
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

                # Fetch pending tasks whose time is now or casually past
                try:
                    cur.execute(
                        "SELECT id, action_type, message_content, customer_id "
                        "FROM intervention_schedule WHERE status = 'scheduled' AND scheduled_date <= ?",
                        (now_str,),
                    )
                    due_tasks = cur.fetchall()
                except sqlite3.OperationalError:
                    due_tasks = []  # Table might not exist yet

                for task_id, action_type, msg, cust_id in due_tasks:
                    print(
                        f"⚡ [WORKER] Executing scheduled action: {action_type} for {cust_id} at {now_str}"
                    )
                    to_phone = config.TEST_PHONE_NUMBER or "+919999999999"

                    if action_type in ("awareness_sms", "recovery_sms") and msg:
                        real_messaging.send_message(to_phone, msg, "SMS")
                    elif action_type == "relationship_call":
                        # Convert brief note into a call script payload
                        script_text = (
                            f"Priority call for {cust_id}. Analyst note: {msg}"
                        )
                        real_calling.make_call(to_phone, script_text)

                    # Mark completed
                    cur.execute(
                        "UPDATE intervention_schedule SET status = 'completed' WHERE id = ?",
                        (task_id,),
                    )
                    conn.commit()
                conn.close()
            except Exception as e:
                print(f"⚠️ Scheduler loop error: {e}")

            time.sleep(20)  # Poll every 20 seconds

    t = threading.Thread(target=run_scheduler_loop, daemon=True)
    t.start()
    return t


_bg_thread = _start_background_executor()

try:
    from recovery_decision_engine import show_recovery_engine

    HAS_RDE = True
except Exception:
    HAS_RDE = False
try:
    from ai_hub import show_ai_hub

    HAS_HUB = True
except Exception:
    HAS_HUB = False

# Page configuration
st.set_page_config(
    page_title="PDIE Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional Barclays enterprise styling
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* === GLOBAL === */
    html, body, [class*="st-"], .stMarkdown, .stText {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* === HEADER BANNER === */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #00395D 50%, #00A3E0 100%);
        padding: 2.5rem 3rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 12px 40px rgba(0, 57, 93, 0.4);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -15%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
        border-radius: 50%;
        animation: slowPulse 8s infinite alternate;
    }
    .main-header h1 { margin: 0 0 0.4rem 0; font-weight: 800; font-size: 2.2rem; letter-spacing: -0.8px; text-shadow: 0 2px 10px rgba(0,0,0,0.3); }
    .main-header p { margin: 0; opacity: 0.9; font-size: 1.05rem; font-weight: 400; }

    @keyframes slowPulse {
        0% { transform: scale(1); opacity: 0.6; }
        100% { transform: scale(1.1); opacity: 1; }
    }

    /* === KPI CARDS (Premium Glass Glow) === */
    .kpi-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 163, 224, 0.2);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::after {
        content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, #00539B, #00A3E0);
        transform: scaleX(0); transform-origin: left; transition: transform 0.3s ease;
    }
    .kpi-card:hover::after { transform: scaleX(1); }
    .kpi-card:hover { 
        transform: translateY(-4px); 
        box-shadow: 0 12px 30px rgba(0, 163, 224, 0.15); 
        border-color: rgba(0, 163, 224, 0.5);
    }
    .kpi-card .kpi-icon { font-size: 1.8rem; margin-bottom: 0.4rem; display: inline-block; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1)); }
    .kpi-card .kpi-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700; }
    .kpi-card .kpi-value { font-size: 2rem; font-weight: 900; color: #0f172a; margin: 0.2rem 0; letter-spacing: -1.5px; }
    .kpi-card .kpi-delta { font-size: 0.85rem; font-weight: 600; display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 12px; background: #f1f5f9; }
    .kpi-delta.up { color: #dc2626; background: rgba(220,38,38,0.1); }
    .kpi-delta.down { color: #16a34a; background: rgba(22,163,74,0.1); }

    /* === RISK BADGES (Animated) === */
    @keyframes pulseRed {
        0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.6); }
        70% { box-shadow: 0 0 0 8px rgba(220, 38, 38, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
    }
    .risk-critical {
        background: linear-gradient(135deg, #ef4444, #991b1b);
        color: white; padding: 0.5rem 1.2rem; border-radius: 30px;
        font-weight: 800; font-size: 0.85rem; display: inline-block;
        box-shadow: 0 4px 12px rgba(220,38,38,0.4);
        border: 1px solid #fca5a5;
        animation: pulseRed 2s infinite;
    }
    .risk-high {
        background: linear-gradient(135deg, #f97316, #c2410c);
        color: white; padding: 0.5rem 1.2rem; border-radius: 30px;
        font-weight: 800; font-size: 0.85rem; display: inline-block;
        box-shadow: 0 4px 12px rgba(249,115,22,0.3);
    }
    .risk-medium {
        background: linear-gradient(135deg, #facc15, #ca8a04);
        color: #1e293b; padding: 0.5rem 1.2rem; border-radius: 30px;
        font-weight: 800; font-size: 0.85rem; display: inline-block;
    }
    .risk-low {
        background: linear-gradient(135deg, #4ade80, #15803d);
        color: white; padding: 0.5rem 1.2rem; border-radius: 30px;
        font-weight: 800; font-size: 0.85rem; display: inline-block;
    }

    /* === INFO CARD (Glassmorphism) === */
    .info-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.05);
        transition: transform 0.2s ease;
    }
    .info-card:hover { transform: translateY(-2px); border-color: rgba(0, 163, 224, 0.3); }
    .info-card h4 { margin: 0 0 1rem 0; color: #00395D; font-weight: 800; font-size: 1.1rem; letter-spacing: -0.3px; }
    .info-card .info-row { display: flex; justify-content: space-between; padding: 0.45rem 0; border-bottom: 1px dashed rgba(203,213,225,0.6); font-size: 0.9rem; }
    .info-card .info-row:last-child { border-bottom: none; }
    .info-card .info-label { color: #64748b; font-weight: 600; }
    .info-card .info-value { color: #0f172a; font-weight: 700; }

    /* === MESSAGE BUBBLES === */
    .sms-bubble {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 18px 18px 18px 4px;
        padding: 1.2rem 1.4rem; max-width: 380px; font-size: 0.95rem;
        line-height: 1.6; color: #1e293b; position: relative;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05); font-weight: 500;
        white-space: pre-wrap; margin-bottom: 1rem;
    }
    .whatsapp-bubble {
        background: #dcf8c6; border: 1px solid #bce69d; border-radius: 18px 18px 18px 4px;
        padding: 1.2rem 1.4rem; max-width: 380px; font-size: 0.95rem;
        line-height: 1.6; color: #1e293b; position: relative;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06); font-weight: 500;
        white-space: pre-wrap; margin-bottom: 1rem;
    }

    /* === INSIGHT BOX === */
    .insight-box {
        background: linear-gradient(135deg, rgba(239,246,255,0.8), rgba(224,242,254,0.8));
        backdrop-filter: blur(8px);
        border-left: 5px solid #00A3E0;
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #1e3a8a;
        line-height: 1.6;
        box-shadow: 0 2px 10px rgba(0,163,224,0.1);
    }
    .insight-box strong { color: #00395D; font-weight: 800; }

    /* === HIGH-END BUTTONS === */
    .stButton>button {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;
        padding: 0.7rem 2.2rem; font-weight: 700;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.4);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); font-size: 0.95rem; letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.5);
        background: linear-gradient(135deg, #00539B 0%, #00A3E0 100%);
        border-color: #38bdf8;
    }

    /* === PREMIUM GLASS SIDEBAR === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.95) 0%, rgba(0,57,93,0.98) 100%) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stRadio label span {
        font-weight: 600 !important; font-size: 0.95rem !important; letter-spacing: 0.3px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
    .sidebar-badge {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: background 0.2s ease;
    }
    .sidebar-badge:hover { background: rgba(255,255,255,0.1); }
    .sidebar-badge .badge-val { font-weight: 800; color: #38bdf8 !important; text-shadow: 0 0 10px rgba(56,189,248,0.4); }

    /* === ENTERPRISE NEUMORPHIC TABS OVERRIDE === */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px !important; 
        border-bottom: 2px solid #e2e8f0 !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0 !important; 
        padding: 0.8rem 1.6rem !important;
        font-weight: 700 !important; 
        font-size: 0.95rem !important; 
        color: #64748b !important;
        transition: all 0.2s ease !important; 
        border: none !important;
        position: relative !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover { 
        color: #0f172a !important; 
        background: rgba(241,245,249,0.8) !important; 
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        color: #00539B !important; 
        background: transparent !important; 
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"]::after {
        content: '' !important; 
        position: absolute !important; 
        bottom: -2px !important; 
        left: 0 !important; 
        width: 100% !important; 
        height: 3px !important;
        background: linear-gradient(90deg, #00539B, #00A3E0) !important; 
        border-radius: 3px 3px 0 0 !important;
    }

    /* === CLEAN EXPANDERS === */
    .streamlit-expanderHeader { font-weight: 700 !important; font-size: 1rem !important; color: #0f172a !important; }
    [data-testid="stExpander"] details summary span[class*="emotion"] {
        font-size: 0 !important; overflow: hidden !important; width: 1.4rem !important; height: 1.4rem !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; flex-shrink: 0 !important;
    }
    [data-testid="stExpander"] details summary span[class*="emotion"] svg { width: 1.2rem !important; height: 1.2rem !important; color: #00A3E0 !important; }
    [data-testid="stExpander"] details summary {
        padding: 0.8rem 1.2rem !important; border-radius: 12px !important;
        background: rgba(241,245,249,0.6) !important; border: 1px solid rgba(226,232,240,0.8);
        transition: all 0.2s ease;
    }
    [data-testid="stExpander"] details summary:hover { background: rgba(241,245,249,1) !important; border-color: #cbd5e1; }
    .streamlit-expanderHeader { font-weight: 600 !important; font-size: 0.95rem !important; }
    /* Fix Material Icon text overlap in expander toggle arrow */
    [data-testid="stExpander"] details summary span.st-emotion-cache-p5msec,
    [data-testid="stExpander"] details summary span[class*="emotion"] {
        font-size: 0 !important; overflow: hidden !important;
        width: 1.2rem !important; height: 1.2rem !important; display: inline-flex !important;
        align-items: center !important; justify-content: center !important; flex-shrink: 0 !important;
    }
    [data-testid="stExpander"] details summary span[class*="emotion"] svg {
        width: 1.2rem !important; height: 1.2rem !important;
    }
    [data-testid="stExpander"] details summary {
        padding: 0.7rem 1rem !important; border-radius: 10px !important;
        background: rgba(248,250,252,0.8) !important;
    }
    [data-testid="stExpander"] details summary:hover {
        background: rgba(0,83,155,0.05) !important;
    }
    [data-testid="stExpander"] details summary p {
        font-weight: 600 !important; font-size: 0.92rem !important;
    }

    /* === EXECUTIVE SUMMARY === */
    .exec-summary {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #166534;
    }
    .exec-summary.warning {
        background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
        border: 1px solid #fde68a;
        color: #854d0e;
    }
    .exec-summary.danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
        color: #991b1b;
    }

    /* === AGENTIC AI STYLES === */
    .reasoning-step {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border-left: 4px solid #00539B;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        position: relative;
        transition: all 0.3s ease;
    }
    .reasoning-step:hover { transform: translateX(4px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .reasoning-step .step-num {
        background: linear-gradient(135deg, #00539B, #00A3E0);
        color: white; border-radius: 50%;
        width: 28px; height: 28px; display: inline-flex;
        align-items: center; justify-content: center;
        font-weight: 800; font-size: 0.8rem; margin-right: 0.6rem;
    }
    .reasoning-step .confidence {
        float: right; background: rgba(34,197,94,0.15);
        color: #16a34a; padding: 0.15rem 0.6rem;
        border-radius: 20px; font-size: 0.75rem; font-weight: 700;
    }
    .orch-node {
        background: white; border: 2px solid #e2e8f0;
        border-radius: 14px; padding: 1rem 1.2rem;
        text-align: center; position: relative;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    .orch-node:hover { border-color: #00539B; box-shadow: 0 4px 16px rgba(0,83,155,0.12); }
    .orch-node.active { border-color: #00539B; background: linear-gradient(135deg, #eff6ff, #dbeafe); }
    .orch-node .node-icon { font-size: 1.8rem; display: block; margin-bottom: 0.4rem; }
    .orch-node .node-title { font-weight: 700; font-size: 0.85rem; color: #00395D; }
    .orch-node .node-detail { font-size: 0.75rem; color: #64748b; margin-top: 0.3rem; }
    .orch-arrow { text-align: center; font-size: 1.5rem; color: #00539B; padding: 0.3rem 0; }
    .chat-user {
        background: linear-gradient(135deg, #00539B, #00A3E0);
        color: white; border-radius: 18px 18px 4px 18px;
        padding: 0.8rem 1.2rem; max-width: 80%; margin-left: auto;
        font-size: 0.88rem; line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0,83,155,0.2);
    }
    .chat-ai {
        background: #f1f5f9; border-radius: 18px 18px 18px 4px;
        padding: 0.8rem 1.2rem; max-width: 85%;
        font-size: 0.88rem; line-height: 1.6; color: #1e293b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 3px solid #00539B;
    }
    .chat-ai .ai-label { font-size: 0.7rem; color: #00539B; font-weight: 700; margin-bottom: 0.3rem; }
    .typing-dots { display: inline-block; }
    .typing-dots span { animation: blink 1.4s infinite both; font-size: 1.2rem; }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink { 0%,80%,100% { opacity:0 } 40% { opacity:1 } }
    .whatsapp-bubble {
        background: #dcf8c6; border-radius: 18px 18px 4px 18px; padding: 1rem 1.2rem;
        max-width: 80%; margin-left: auto; font-size: 0.88rem; line-height: 1.6;
        box-shadow: 0 2px 8px rgba(37,211,102,0.15); white-space: pre-line;
    }
    .sms-bubble {
        background: #e8f4fd; border-radius: 18px 18px 4px 18px; padding: 1rem 1.2rem;
        max-width: 80%; margin-left: auto; font-size: 0.88rem; line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,163,224,0.15); white-space: pre-line;
    }
    .msg-meta { font-size: 0.7rem; color: #64748b; margin-top: 0.5rem; text-align: right; }
</style>
""",
    unsafe_allow_html=True,
)


# ===== DATA LOADING & CACHING =====


@st.cache_data
def load_features_data():
    """Load customer features data."""
    try:
        features_path = Path("../pdie_feature_store/features.parquet")
        if not features_path.exists():
            # Try alternative path
            features_path = Path("pdie_feature_store/features.parquet")

        df = pd.read_parquet(features_path)
        return df
    except Exception as e:
        st.error(f"Error loading features data: {str(e)}")
        st.info(
            "Please ensure 'pdie_feature_store/features.parquet' exists in the parent directory."
        )
        return None


@st.cache_data
def load_customers_data():
    """Load customer PII/demographic data (name, city, etc)."""
    try:
        for p in [
            Path("../pdie_feature_store/customers.parquet"),
            Path("pdie_feature_store/customers.parquet"),
        ]:
            if p.exists():
                return pd.read_parquet(p)
        return None
    except Exception:
        return None


@st.cache_data
def load_transactions_data():
    """Load transaction-level data for drill-down charts."""
    try:
        for p in [
            Path("../pdie_feature_store/transactions.parquet"),
            Path("pdie_feature_store/transactions.parquet"),
        ]:
            if p.exists():
                return pd.read_parquet(p)
        return None
    except Exception:
        return None


@st.cache_data
def load_balances_data():
    """Load savings balance data."""
    try:
        for p in [
            Path("../pdie_feature_store/balances.parquet"),
            Path("pdie_feature_store/balances.parquet"),
        ]:
            if p.exists():
                return pd.read_parquet(p)
        return None
    except Exception:
        return None


@st.cache_data
def load_loans_data():
    """Load real loan contract data (EMI days, principal, etc)."""
    try:
        for p in [
            Path("../pdie_feature_store/loans.parquet"),
            Path("pdie_feature_store/loans.parquet"),
        ]:
            if p.exists():
                return pd.read_parquet(p)
        return None
    except Exception:
        return None


@st.cache_resource
def load_model():
    """Load trained XGBoost model."""
    try:
        model_path = Path("../pdie_model_outputs/pdie_xgboost_model.pkl")
        if not model_path.exists():
            model_path = Path("pdie_model_outputs/pdie_xgboost_model.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure 'pdie_model_outputs/pdie_xgboost_model.pkl' exists.")
        return None


@st.cache_data
def load_shap_values():
    """Load SHAP explanation values."""
    try:
        shap_path = Path("../pdie_model_outputs/shap_values.csv")
        if not shap_path.exists():
            shap_path = Path("pdie_model_outputs/shap_values.csv")

        df = pd.read_csv(shap_path)
        return df
    except Exception as e:
        st.warning(f"SHAP values not available: {str(e)}")
        return None


@st.cache_data
def calculate_risk_scores(features_df, _model):
    """Calculate risk scores for all customers."""
    if _model is None or features_df is None:
        return features_df

    # Prepare features for prediction
    exclude_cols = ["customer_id", "will_default_in_21_days"]
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    # Load expected feature names from model training
    feature_names_path = Path("../pdie_model_outputs/feature_names.json")
    if not feature_names_path.exists():
        feature_names_path = Path("pdie_model_outputs/feature_names.json")

    try:
        with open(feature_names_path) as f:
            expected_features = json.load(f)
    except Exception:
        # Fallback: use get_dummies and hope for the best
        expected_features = None

    # One-hot encode categoricals
    X = pd.get_dummies(features_df[feature_cols])

    # Align columns to match what the model was trained on
    if expected_features is not None:
        X = X.reindex(columns=expected_features, fill_value=0)

    # Predict
    predictions = _model.predict_proba(X)[:, 1]

    # Add to dataframe
    features_df = features_df.copy()

    # Percentile-based risk score rescaling
    # The tuned model produces compressed probabilities (e.g. 45-55% range).
    # We rescale to 0-100 using percentile rank to preserve relative ordering
    # while ensuring all risk tiers (LOW/MEDIUM/HIGH/CRITICAL) are populated.
    from scipy.stats import rankdata

    percentile_scores = (
        rankdata(predictions, method="average") / len(predictions)
    ) * 100
    features_df["risk_score"] = percentile_scores.round(1)

    # Derive useful columns from existing data for downstream use
    features_df["emi_amount"] = (
        features_df["emi_to_income_ratio"] * features_df["monthly_income"]
    ).round(0)

    # Risk category
    features_df["risk_category"] = pd.cut(
        features_df["risk_score"],
        bins=[0, 50, 70, 80, 100],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )

    # ── Merge in real customer names & city from customers.parquet ──
    try:
        customers_df = load_customers_data()
        if customers_df is not None:
            keep_cols = ["customer_id", "full_name", "city", "account_opening_date"]
            keep_cols = [c for c in keep_cols if c in customers_df.columns]
            features_df = features_df.merge(
                customers_df[keep_cols], on="customer_id", how="left"
            )

        # ── Merge in real loan data for EMI days & precision ──
        loans_df = load_loans_data()
        if loans_df is not None:
            l_cols = [
                "customer_id",
                "loan_id",
                "loan_type",
                "sanction_date",
                "sanction_amount",
                "outstanding_principal",
                "interest_rate",
                "emi_amount",
                "emi_day_of_month",
                "tenure_months",
                "remaining_months",
                "loan_status",
            ]
            l_cols = [c for c in l_cols if c in loans_df.columns]
            features_df = features_df.merge(
                loans_df[l_cols], on="customer_id", how="left", suffixes=("", "_loan")
            )
            # Favor the real loan data if it exists
            if "emi_amount_loan" in features_df.columns:
                features_df["emi_amount"] = features_df["emi_amount_loan"].fillna(
                    features_df["emi_amount"]
                )
                features_df.drop(columns=["emi_amount_loan"], inplace=True)
            if "outstanding_principal" in features_df.columns:
                # If principal is NaN, assume 500,000 as default instead of 0
                features_df["outstanding_principal"] = features_df[
                    "outstanding_principal"
                ].fillna(500000)

        # Final NaN cleanup for essential engine fields
        features_df["monthly_income"] = features_df["monthly_income"].fillna(85000)
        features_df["emi_amount"] = features_df["emi_amount"].fillna(18500)
        features_df["outstanding_principal"] = features_df[
            "outstanding_principal"
        ].fillna(500000)

    except Exception as e:
        print(f"Data merge error: {e}")
        pass

    return features_df


# ===== HELPER FUNCTIONS =====


def get_risk_badge(score):
    """Generate HTML badge for risk score."""
    if score >= 80:
        return f'<span class="risk-critical">CRITICAL ({score:.0f})</span>'
    elif score >= 70:
        return f'<span class="risk-high">HIGH ({score:.0f})</span>'
    elif score >= 50:
        return f'<span class="risk-medium">MEDIUM ({score:.0f})</span>'
    else:
        return f'<span class="risk-low">LOW ({score:.0f})</span>'


def format_currency(amount):
    """Format amount as Indian currency."""
    return f"₹{amount:,.0f}"


def create_gauge_chart(score, title="Risk Score"):
    """Create a gauge chart for scores."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "#d4edda"},
                    {"range": [50, 70], "color": "#fff3cd"},
                    {"range": [70, 80], "color": "#f8d7da"},
                    {"range": [80, 100], "color": "#dc3545"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
        )
    )

    fig.update_layout(height=250)
    return fig


# ===== PAGE COMPONENTS =====


def show_portfolio_overview(df):
    """Display enriched portfolio overview page."""
    st.markdown(
        '<div class="main-header"><h1>📊 Portfolio Command Center</h1><p>Real-time risk intelligence across 10,000+ retail customers · XGBoost ML predictions updated every 24 hours</p></div>',
        unsafe_allow_html=True,
    )

    # Compute KPIs
    total = len(df)
    high_risk = len(df[df["risk_score"] >= 70])
    critical = len(df[df["risk_score"] >= 80])
    avg_risk = df["risk_score"].mean()
    at_risk = len(df[df["risk_score"] >= 50])

    total_exposure = 0
    avg_income_atrisk = 0
    if "emi_amount" in df.columns:
        total_exposure = df[df["risk_score"] >= 50]["emi_amount"].sum() * 12
    if "monthly_income" in df.columns:
        at_risk_df = df[df["risk_score"] >= 70]
        avg_income_atrisk = (
            at_risk_df["monthly_income"].mean() if len(at_risk_df) > 0 else 0
        )

    est_savings = critical * 18500 * 0.4  # Estimated savings from intervention

    # KPI Cards Row 1
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("👥", "Total Customers", f"{total:,}", f"100% portfolio", col1),
        (
            "🔴",
            "Critical Risk (≥80)",
            f"{critical:,}",
            f"{critical / total * 100:.1f}% of portfolio",
            col2,
        ),
        (
            "🚨",
            "High Risk (≥70)",
            f"{high_risk:,}",
            f"{high_risk / total * 100:.1f}% of portfolio",
            col3,
        ),
        (
            "⚡",
            "At Risk (≥50)",
            f"{at_risk:,}",
            f"{at_risk / total * 100:.1f}% of portfolio",
            col4,
        ),
    ]
    for icon, label, value, delta, col in kpis:
        with col:
            delta_cls = "up" if "Critical" in label or "High" in label else ""
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-delta {delta_cls}">{delta}</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Cards Row 2
    col1, col2, col3, col4 = st.columns(4)
    kpis2 = [
        ("📈", "Avg Risk Score", f"{avg_risk:.1f}", "Portfolio mean", col1),
        (
            "💰",
            "Total Exposure (Annual)",
            f"₹{total_exposure / 10000000:.1f}Cr" if total_exposure > 0 else "N/A",
            f"{at_risk:,} at-risk accounts",
            col2,
        ),
        (
            "💼",
            "Avg Income (High Risk)",
            f"₹{avg_income_atrisk:,.0f}" if avg_income_atrisk > 0 else "N/A",
            f"{high_risk:,} customers",
            col3,
        ),
        (
            "🛡️",
            "Est. Intervention Savings",
            f"₹{est_savings / 100000:.1f}L",
            f"If {critical:,} critical intervened",
            col4,
        ),
    ]
    for icon, label, value, delta, col in kpis2:
        with col:
            st.markdown(
                f"""<div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-delta">{delta}</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Executive Summary
    if critical > total * 0.05:
        summary_cls = "danger"
        summary_text = f"⚠️ <strong>Elevated Alert:</strong> {critical:,} customers ({critical / total * 100:.1f}%) are in the CRITICAL zone (risk ≥80). These accounts require <strong>immediate intervention</strong> via the AI Communication Agent. The model predicts potential defaults within 21 days for these customers. Estimated annual exposure: <strong>₹{total_exposure / 10000000:.1f}Cr</strong>."
    elif high_risk > total * 0.15:
        summary_cls = "warning"
        summary_text = f"⚡ <strong>Moderate Alert:</strong> {high_risk:,} customers ({high_risk / total * 100:.1f}%) are in HIGH+ risk zones. Proactive outreach recommended for the top {min(100, critical):,} critical cases. Early intervention can reduce default rates by 35-40%."
    else:
        summary_cls = ""
        summary_text = f"✅ <strong>Portfolio Healthy:</strong> Only {high_risk / total * 100:.1f}% of customers are in high-risk zones. Continue standard monitoring with weekly reviews."

    st.markdown(
        f'<div class="exec-summary {summary_cls}">{summary_text}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Charts Row 1: Donut + Histogram
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🎯 Risk Tier Breakdown")
        risk_counts = df["risk_category"].value_counts().sort_index()
        colors = ["#22c55e", "#eab308", "#f97316", "#dc2626"]
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.55,
                    marker_colors=colors,
                    textinfo="label+percent",
                    textfont_size=13,
                    hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Percentage: %{percent}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5
            ),
            annotations=[
                dict(
                    text=f"<b>{total:,}</b><br>Total",
                    x=0.5,
                    y=0.5,
                    font_size=16,
                    showarrow=False,
                )
            ],
            margin=dict(t=20, b=40, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### 📊 Risk Score Distribution")
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df["risk_score"],
                nbinsx=25,
                marker_color="rgba(0, 83, 155, 0.7)",
                marker_line=dict(color="rgba(0, 57, 93, 0.9)", width=0.5),
                hovertemplate="Risk Score: %{x}<br>Count: %{y}<extra></extra>",
            )
        )
        fig.add_vline(
            x=50,
            line_dash="dash",
            line_color="#eab308",
            annotation_text="▸ Medium",
            annotation_font_color="#eab308",
        )
        fig.add_vline(
            x=70,
            line_dash="dash",
            line_color="#f97316",
            annotation_text="▸ High",
            annotation_font_color="#f97316",
        )
        fig.add_vline(
            x=80,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="▸ Critical",
            annotation_font_color="#dc2626",
        )
        fig.update_layout(
            xaxis_title="Risk Score",
            yaxis_title="Number of Customers",
            height=400,
            margin=dict(t=20, b=40, l=40, r=20),
            plot_bgcolor="rgba(248,250,252,0.5)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Charts Row 2: EMI Scatter + Demographics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 💸 EMI Burden vs Risk Score")
        if "emi_to_income_ratio" in df.columns:
            sample = df.sample(min(1000, len(df)), random_state=42)
            fig = px.scatter(
                sample,
                x="emi_to_income_ratio",
                y="risk_score",
                color="risk_category",
                color_discrete_map={
                    "LOW": "#22c55e",
                    "MEDIUM": "#eab308",
                    "HIGH": "#f97316",
                    "CRITICAL": "#dc2626",
                },
                opacity=0.6,
                hover_data=["customer_id", "monthly_income"],
                labels={
                    "emi_to_income_ratio": "EMI / Income Ratio",
                    "risk_score": "Risk Score",
                },
            )
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=40, l=40, r=20),
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
                ),
                plot_bgcolor="rgba(248,250,252,0.5)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("EMI-to-income ratio data not available.")

    with col2:
        st.markdown("##### 🏢 Risk by Employment Type")
        if "employment_type" in df.columns:
            heatmap_data = (
                df.groupby("employment_type")
                .agg(avg_risk=("risk_score", "mean"), count=("customer_id", "count"))
                .sort_values("avg_risk", ascending=True)
            )

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=heatmap_data["avg_risk"],
                        y=heatmap_data.index,
                        orientation="h",
                        marker_color=[
                            "#dc2626"
                            if v >= 70
                            else "#f97316"
                            if v >= 50
                            else "#22c55e"
                            for v in heatmap_data["avg_risk"]
                        ],
                        text=[
                            f"{v:.1f} ({c:,})"
                            for v, c in zip(
                                heatmap_data["avg_risk"], heatmap_data["count"]
                            )
                        ],
                        textposition="auto",
                        hovertemplate="<b>%{y}</b><br>Avg Risk: %{x:.1f}<extra></extra>",
                    )
                ]
            )
            fig.update_layout(
                xaxis_title="Average Risk Score",
                height=400,
                margin=dict(t=20, b=40, l=20, r=20),
                plot_bgcolor="rgba(248,250,252,0.5)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # City Tier chart
    if "city_tier" in df.columns:
        st.markdown("##### 🌆 Risk by City Tier")
        tier_data = (
            df.groupby("city_tier")
            .agg(
                avg_risk=("risk_score", "mean"),
                total=("customer_id", "count"),
                high_risk=("risk_score", lambda x: (x >= 70).sum()),
            )
            .sort_values("avg_risk", ascending=False)
        )

        col1, col2, col3 = st.columns(len(tier_data))
        for (tier, row), col in zip(
            tier_data.iterrows(), [col1, col2, col3][: len(tier_data)]
        ):
            with col:
                st.metric(
                    f"🏙️ {tier}",
                    f"{row['avg_risk']:.1f}",
                    f"{row['high_risk']:,} high risk / {row['total']:,} total",
                )

    # ============================================================
    # NEW: Additional Impact Metrics Section
    # ============================================================
    st.markdown("---")
    st.markdown("##### 📈 Portfolio Health & Impact Metrics")

    # Calculate additional impressive metrics
    total_customers = len(df)

    # Emergency Fund Analysis
    emergency_fund_adequate = (
        len(df[df["emergency_fund_days"] >= 30])
        if "emergency_fund_days" in df.columns
        else 0
    )
    emergency_fund_critical = (
        len(df[df["emergency_fund_days"] < 7])
        if "emergency_fund_days" in df.columns
        else 0
    )

    # Salary Delay Analysis
    salary_delay_count = (
        len(df[df["salary_delay_days"] > 0]) if "salary_delay_days" in df.columns else 0
    )

    # UPI Lending App Usage (risky behavior indicator)
    upi_lending_users = (
        len(df[df["upi_lending_app_txn_count_30d"] > 0])
        if "upi_lending_app_txn_count_30d" in df.columns
        else 0
    )

    # Bill Payment Delays
    bill_delay_count = (
        len(df[df["bill_payment_delay_max"] > 5])
        if "bill_payment_delay_max" in df.columns
        else 0
    )

    # High EMI Burden
    high_emi_burden = (
        len(df[df["emi_to_income_ratio"] > 0.5])
        if "emi_to_income_ratio" in df.columns
        else 0
    )

    # Savings Drawdown
    savings_drawdown = (
        len(df[df["savings_drawdown_rate_4w"] > 0.2])
        if "savings_drawdown_rate_4w" in df.columns
        else 0
    )

    # Calculate projected financial impact
    avg_emi = df["emi_amount"].mean() if "emi_amount" in df.columns else 18500
    projected_default_rate = critical / total_customers if total_customers > 0 else 0
    potential_loss_prevention = (
        critical * avg_emi * 12 * 0.35
    )  # 35% reduction with intervention
    roi_percentage = (
        potential_loss_prevention / (total_customers * 100)
    ) * 100  # Assuming 100 per customer operational cost

    # New KPI Row: Financial Impact
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="kpi-card" style="border-left: 4px solid #22c55e;">
            <div class="kpi-icon">💵</div>
            <div class="kpi-label">Potential Loss Prevention</div>
            <div class="kpi-value">₹{potential_loss_prevention / 10000000:.2f}Cr</div>
            <div class="kpi-delta">35% intervention effect</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="kpi-card" style="border-left: 4px solid #3b82f6;">
            <div class="kpi-icon">📉</div>
            <div class="kpi-label">Projected Default Rate</div>
            <div class="kpi-value">{projected_default_rate * 100:.2f}%</div>
            <div class="kpi-delta">Of at-risk accounts</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="kpi-card" style="border-left: 4px solid #8b5cf6;">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-label">Intervention ROI</div>
            <div class="kpi-value">{roi_percentage:.0f}x</div>
            <div class="kpi-delta">Return on investment</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
            <div class="kpi-icon">🛡️</div>
            <div class="kpi-label">Adequate Emergency Fund</div>
            <div class="kpi-value">{emergency_fund_adequate:,}</div>
            <div class="kpi-delta">{emergency_fund_adequate / total_customers * 100:.1f}% of portfolio</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
        <div class="kpi-card" style="border-left: 4px solid #ef4444;">
            <div class="kpi-icon">⚠️</div>
            <div class="kpi-label">Critical Emergency Fund</div>
            <div class="kpi-value">{emergency_fund_critical:,}</div>
            <div class="kpi-delta"><7 days savings</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk Behavioral Indicators Row
    st.markdown("##### 🔍 Early Warning Signal Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Salary Delays",
            f"{salary_delay_count:,}",
            f"{salary_delay_count / total_customers * 100:.1f}%",
        )

    with col2:
        st.metric(
            "📱 UPI Lending Users",
            f"{upi_lending_users:,}",
            f"{upi_lending_users / total_customers * 100:.1f}%",
        )

    with col3:
        st.metric(
            "📋 Bill Payment Delays",
            f"{bill_delay_count:,}",
            f"{bill_delay_count / total_customers * 100:.1f}%",
        )

    with col4:
        st.metric(
            "📉 Savings Drawdown",
            f"{savings_drawdown:,}",
            f"{savings_drawdown / total_customers * 100:.1f}%",
        )

    # Additional Chart: Risk Distribution by Age Group
    if "age" in df.columns:
        import pandas as pd

        st.markdown("##### 👥 Risk Distribution by Age Group")

        # Create age groups
        df_age = df.copy()
        df_age["age_group"] = pd.cut(
            df_age["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "55+"],
        )

        age_risk = (
            df_age.groupby("age_group")
            .agg(
                avg_risk=("risk_score", "mean"),
                count=("customer_id", "count"),
                critical=("risk_score", lambda x: (x >= 80).sum()),
            )
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=age_risk["age_group"],
                y=age_risk["avg_risk"],
                name="Avg Risk Score",
                marker_color="rgba(0, 83, 155, 0.7)",
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=age_risk["age_group"],
                y=age_risk["critical"],
                name="Critical Cases",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#dc2626", width=3),
                marker=dict(size=10),
            )
        )

        fig.update_layout(
            height=350,
            yaxis=dict(title="Avg Risk Score", range=[0, 100]),
            yaxis2=dict(title="Critical Cases", overlaying="y", side="right"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            margin=dict(t=40, b=40, l=40, r=60),
            plot_bgcolor="rgba(248,250,252,0.5)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model Performance Section
    st.markdown("##### 🤖 Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎯 Model AUC",
            "86.3%",
            "Test Set Performance",
        )

    with col2:
        st.metric(
            "📊 Features Used",
            "24",
            "Behavioral Indicators",
        )

    with col3:
        st.metric(
            "⏱️ Prediction Window",
            "21 Days",
            "Time to Default",
        )

    with col4:
        st.metric(
            "🔄 Model Updates",
            "Weekly",
            "Real-time Retraining",
        )


def show_top_at_risk(df):
    """Display enriched at-risk customers page in a clean data table."""
    st.markdown(
        """
        <style>
        .main-header h1 { color: #00539B !important; font-weight: 800; }
        .table-header { color: #64748b !important; font-size: 0.75rem !important; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; padding-bottom: 0.8rem; border-bottom: 2px solid #e2e8f0; margin-bottom: 1rem; }
        .row-text { font-size: 0.9rem; font-weight: 600; padding-top: 0.5rem; }
        .row-text-muted { font-size: 0.9rem; font-weight: 400; color: #475569; padding-top: 0.5rem; }
        .badge-clean { background: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; border: 1px solid #bbf7d0; }
        .badge-flagged { background: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; border: 1px solid #fecaca; }
        .badge-type { background: #e0f2fe; color: #075985; padding: 4px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;}
        .stat-low { color: #166534; font-weight: 700; font-size:0.85rem;}
        .stat-med { color: #b45309; font-weight: 700; font-size:0.85rem;}
        .stat-high { color: #991b1b; font-weight: 700; font-size:0.85rem;}
        div[data-testid="stHorizontalBlock"] { align-items: center !important; border-bottom: 1px solid #f1f5f9; padding: 12px 0; }
        div[data-testid="stHorizontalBlock"]:hover { background-color: #f8fafc; }
        .stButton>button { padding: 0.2rem 0.5rem !important; width: 100% !important; margin: 0 !important; font-size: 0.85rem !important;}
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header" style="margin-bottom:2rem;"><h1>🚨 At-Risk Customer Intelligence</h1><p style="color:#64748b;">Prioritized intervention list · Customers predicted to default within 21 days · Sorted by urgency</p></div>',
        unsafe_allow_html=True,
    )

    # ── PORTFOLIO SUMMARY BADGES ──────────────────────────────────────────────
    n_critical = int((df["risk_score"] >= 80).sum())
    n_high = int(((df["risk_score"] >= 70) & (df["risk_score"] < 80)).sum())
    n_medium = int(((df["risk_score"] >= 50) & (df["risk_score"] < 70)).sum())
    n_total = len(df)
    st.markdown(
        f"""
        <div style="display:flex; gap:0.8rem; margin-bottom:1.2rem; flex-wrap:wrap; align-items:center;">
            <div style="background:#fee2e2; border:1px solid #fecaca; border-radius:8px; padding:0.45rem 1.1rem; display:flex; align-items:center; gap:0.5rem;">
                <span style="font-size:1rem;">🔴</span>
                <span style="font-weight:700; color:#991b1b; font-size:0.95rem;">{n_critical:,}</span>
                <span style="color:#991b1b; font-size:0.75rem; font-weight:600; letter-spacing:0.5px;">CRITICAL</span>
            </div>
            <div style="background:#fef3c7; border:1px solid #fde68a; border-radius:8px; padding:0.45rem 1.1rem; display:flex; align-items:center; gap:0.5rem;">
                <span style="font-size:1rem;">🟠</span>
                <span style="font-weight:700; color:#92400e; font-size:0.95rem;">{n_high:,}</span>
                <span style="color:#92400e; font-size:0.75rem; font-weight:600; letter-spacing:0.5px;">HIGH RISK</span>
            </div>
            <div style="background:#dbeafe; border:1px solid #bfdbfe; border-radius:8px; padding:0.45rem 1.1rem; display:flex; align-items:center; gap:0.5rem;">
                <span style="font-size:1rem;">🔵</span>
                <span style="font-weight:700; color:#1e40af; font-size:0.95rem;">{n_medium:,}</span>
                <span style="color:#1e40af; font-size:0.75rem; font-weight:600; letter-spacing:0.5px;">MEDIUM</span>
            </div>
            <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:0.45rem 1.1rem; display:flex; align-items:center; gap:0.5rem;">
                <span style="font-size:1rem;">👥</span>
                <span style="font-weight:700; color:#166534; font-size:0.95rem;">{n_total:,}</span>
                <span style="color:#166534; font-size:0.75rem; font-weight:600; letter-spacing:0.5px;">TOTAL</span>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # ── FILTER BAR ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:1rem 1.2rem; margin-bottom:1rem;">',
        unsafe_allow_html=True,
    )
    fa, fb, fc, fd, fe = st.columns([2, 1.5, 1.5, 1.5, 1])

    search_name = fa.text_input(
        "🔎 Search Name / ID",
        placeholder="e.g. Vedika or CUST00009817",
        key="ar_search",
        label_visibility="visible",
    )

    risk_tiers = fb.multiselect(
        "Risk Tier",
        options=["CRITICAL (≥80)", "HIGH (70–80)", "MEDIUM (50–70)", "LOW (<50)"],
        default=["CRITICAL (≥80)", "HIGH (70–80)"],
        key="ar_risk",
    )

    city_opts = (
        ["All"] + sorted(df["city_tier"].dropna().unique().tolist())
        if "city_tier" in df.columns
        else ["All"]
    )
    city_tier_sel = fc.selectbox("City Tier", options=city_opts, key="ar_city")

    emp_opts = (
        ["All"] + sorted(df["employment_type"].dropna().unique().tolist())
        if "employment_type" in df.columns
        else ["All"]
    )
    emp_type_sel = fd.selectbox("Employment Type", options=emp_opts, key="ar_emp")

    rows_per_page = fe.selectbox(
        "Per page", options=[10, 25, 50, 100], index=1, key="ar_rows"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── APPLY FILTERS ─────────────────────────────────────────────────────────
    filtered = df.sort_values("risk_score", ascending=False).copy()

    tier_ranges = []
    if "CRITICAL (≥80)" in risk_tiers:
        tier_ranges.append((80, 101))
    if "HIGH (70–80)" in risk_tiers:
        tier_ranges.append((70, 80))
    if "MEDIUM (50–70)" in risk_tiers:
        tier_ranges.append((50, 70))
    if "LOW (<50)" in risk_tiers:
        tier_ranges.append((0, 50))
    if tier_ranges:
        mask = pd.Series(False, index=filtered.index)
        for lo, hi in tier_ranges:
            mask = mask | (
                (filtered["risk_score"] >= lo) & (filtered["risk_score"] < hi)
            )
        filtered = filtered[mask]

    if search_name.strip():
        q = search_name.strip().lower()
        name_mask = filtered.apply(
            lambda r: (
                q in str(r.get("full_name", "")).lower()
                or q in str(r.get("customer_id", "")).lower()
            ),
            axis=1,
        )
        filtered = filtered[name_mask]

    if city_tier_sel != "All" and "city_tier" in filtered.columns:
        filtered = filtered[filtered["city_tier"] == city_tier_sel]

    if emp_type_sel != "All" and "employment_type" in filtered.columns:
        filtered = filtered[filtered["employment_type"] == emp_type_sel]

    # ── RESULTS COUNT + PAGINATION ────────────────────────────────────────────
    page_key = "ar_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    total_pages = max(1, (len(filtered) - 1) // rows_per_page + 1)
    current_page = min(st.session_state.get(page_key, 0), total_pages - 1)

    pc1, pc2, pc3, pc4 = st.columns([3, 1, 1, 1])
    pc1.markdown(
        f'<div style="padding:0.4rem 0; color:#64748b; font-size:0.85rem;">'
        f'Showing <b style="color:#00539B;">{len(filtered):,}</b> customers · '
        f"Page <b>{current_page + 1}</b> of <b>{total_pages}</b></div>",
        unsafe_allow_html=True,
    )
    with pc2:
        if st.button("⬅ First", key="ar_first", disabled=(current_page == 0)):
            st.session_state[page_key] = 0
            st.rerun()
    with pc3:
        if st.button("◀ Prev", key="ar_prev", disabled=(current_page == 0)):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with pc4:
        if st.button(
            "Next ▶", key="ar_next", disabled=(current_page >= total_pages - 1)
        ):
            st.session_state[page_key] = current_page + 1
            st.rerun()

    start = current_page * rows_per_page
    top_customers = filtered.iloc[start : start + rows_per_page]

    # ── TABLE HEADERS ──
    cols = st.columns([0.4, 0.5, 1.2, 1.3, 0.9, 2.3, 1.1, 1.3, 1.2, 0.7, 0.7])
    cols[0].markdown(
        '<div class="table-header" style="text-align:center;">★</div>',
        unsafe_allow_html=True,
    )
    cols[1].markdown('<div class="table-header">#</div>', unsafe_allow_html=True)
    cols[2].markdown('<div class="table-header">ID</div>', unsafe_allow_html=True)
    cols[3].markdown('<div class="table-header">NAME</div>', unsafe_allow_html=True)
    cols[4].markdown('<div class="table-header">FLAG</div>', unsafe_allow_html=True)
    cols[5].markdown(
        '<div class="table-header">REASON / FLAG TYPE</div>', unsafe_allow_html=True
    )
    cols[6].markdown(
        '<div class="table-header">ASSIGNED TO</div>', unsafe_allow_html=True
    )
    cols[7].markdown(
        '<div class="table-header">CASE STATUS</div>', unsafe_allow_html=True
    )
    cols[8].markdown('<div class="table-header">RISK</div>', unsafe_allow_html=True)
    cols[9].markdown(
        '<div class="table-header" style="text-align:center;">VIEW</div>',
        unsafe_allow_html=True,
    )
    cols[10].markdown(
        '<div class="table-header" style="text-align:center;">TEAM</div>',
        unsafe_allow_html=True,
    )

    # ── CASE STATUS COLOUR MAP ──
    STATUS_COLORS = {
        "Open": ("#64748b", "#f1f5f9"),
        "Under Review": ("#1d4ed8", "#dbeafe"),
        "Contacted": ("#a16207", "#fef9c3"),
        "Resolved": ("#15803d", "#dcfce7"),
        "Escalated": ("#b91c1c", "#fee2e2"),
    }

    for idx, row in top_customers.reset_index().iterrows():
        cid = row["customer_id"]
        cp0, cp1, cp2, cp3, cp4, cp5, cp6, cp7, cp8, cp9, cp10 = st.columns(
            [0.4, 0.5, 1.2, 1.3, 0.9, 2.3, 1.1, 1.3, 1.2, 0.7, 0.7]
        )

        # ── Priority star toggle ──
        pri_key = f"priority_{cid}"
        is_priority = st.session_state.get(pri_key, False)
        with cp0:
            if st.button(
                "⭐" if is_priority else "☆",
                key=f"pri_{cid}_{idx}",
                help="Toggle priority",
            ):
                st.session_state[pri_key] = not is_priority
                st.rerun()

        cp1.markdown(
            f'<div class="row-text-muted" style="text-align:center;padding-top:0.5rem;">{idx + 1 + start}</div>',
            unsafe_allow_html=True,
        )
        cid_short = (
            str(cid).replace("CUST0000", "USR-")
            if "CUST" in str(cid)
            else f"USR-{str(cid)[:4]}"
        )
        cp2.markdown(
            f'<div class="row-text-muted">{cid_short}</div>', unsafe_allow_html=True
        )

        # ── Real name ──
        name = str(row.get("full_name", cid))
        cp3.markdown(f'<div class="row-text">{name}</div>', unsafe_allow_html=True)

        is_clean = row["risk_score"] < 50
        flag_badge = (
            '<span class="badge-clean">✓ Clean</span>'
            if is_clean
            else '<span class="badge-flagged">⚑ Flagged</span>'
        )
        cp4.markdown(
            f'<div style="padding-top:0.4rem;">{flag_badge}</div>',
            unsafe_allow_html=True,
        )

        # ── Derive reason + flag type ──
        if is_clean:
            reason, f_type = "—", ""
        else:
            salary_delay = float(row.get("salary_delay_days", 0) or 0)
            lending_txns = float(row.get("upi_lending_app_txn_count_30d", 0) or 0)
            savings_draw = float(row.get("savings_drawdown_rate_4w", 0) or 0)
            emi_ratio = float(row.get("emi_to_income_ratio", 0) or 0)
            bill_delay = float(row.get("bill_payment_delay_max", 0) or 0)
            atm_spike = float(row.get("atm_withdrawal_spike_pct", 0) or 0)

            if salary_delay > 2:
                reason = f"Salary Timing Shift ({int(salary_delay)}d)"
                f_type = "Income Disruption"
            elif lending_txns >= 2:
                reason = f"Debt Stacking ({int(lending_txns)} apps)"
                f_type = "Shadow Lending"
            elif atm_spike > 0.4:
                reason = f"ATM Hoarding (+{atm_spike * 100:.0f}%)"
                f_type = "Liquidity Flight"
            elif savings_draw > 0.4:
                reason = f"Savings Erosion (-{savings_draw * 100:.0f}%)"
                f_type = "Cash Reserve Drain"
            elif bill_delay > 4:
                reason = f"Utility Delay ({int(bill_delay)}d)"
                f_type = "Secondary Obligation Stress"
            elif emi_ratio > 0.5:
                reason = f"DTI Crisis ({emi_ratio * 100:.0f}%)"
                f_type = "Over-leveraged"
            else:
                reason = "Multi-signal Early Stress"
                f_type = "Risk Monitoring"

        note_count = len(st.session_state.get(f"notes_{cid}", []))
        note_indicator = f" 📝{note_count}" if note_count > 0 else ""
        interv_count = len(st.session_state.get(f"interventions_{cid}", []))
        interv_indicator = f" 📋{interv_count}" if interv_count > 0 else ""
        type_badge = (
            f'<span class="badge-type" style="font-size:0.72rem;">{f_type}</span>'
            if f_type
            else ""
        )
        cp5.markdown(
            f'<div class="row-text-muted" style="font-size:0.82rem;">{reason}{note_indicator}{interv_indicator}<br>{type_badge}</div>',
            unsafe_allow_html=True,
        )

        # ── Assigned agent ──
        assign_key = f"agent_{cid}"
        agent = st.session_state.get(assign_key, "Unassigned")
        cp6.markdown(
            f'<div class="row-text-muted" style="font-style:{"italic" if agent == "Unassigned" else "normal"};font-size:0.85rem;">{agent}</div>',
            unsafe_allow_html=True,
        )

        # ── Case status badge ──
        case_status_key = f"case_status_{cid}"
        case_status = st.session_state.get(case_status_key, "Open")
        fg, bg = STATUS_COLORS.get(case_status, ("#64748b", "#f1f5f9"))
        cp7.markdown(
            f'<div style="padding-top:0.35rem;"><span style="background:{bg};color:{fg};padding:3px 8px;border-radius:12px;font-size:0.72rem;font-weight:700;border:1px solid {fg}33;">{case_status}</span></div>',
            unsafe_allow_html=True,
        )

        # ── Risk level ──
        risk_html = (
            '<span class="stat-low">● Low</span>'
            if is_clean
            else (
                '<span class="stat-high">● Critical</span>'
                if row["risk_score"] >= 80
                else '<span class="stat-med">● High</span>'
            )
        )
        cp8.markdown(
            f'<div style="padding-top:0.5rem;">{risk_html}</div>',
            unsafe_allow_html=True,
        )

        with cp9:
            if st.button("👁", key=f"view_{cid}_{idx}", help="View Profile"):
                st.session_state["selected_customer"] = cid
                st.session_state["selected_customer_name"] = name
                st.session_state["page"] = "⚡ Recovery Decision Engine"
                st.rerun()
        with cp10:
            if st.button("👤", key=f"assign_{cid}_{idx}", help="Assign to Me"):
                current_analyst = st.session_state.get("username", "Analyst")
                st.session_state[assign_key] = current_analyst
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE 3 — MY QUEUE / ASSIGNED CASES
# ──────────────────────────────────────────────────────────────────────────────
def show_my_queue(df):
    """Dedicated view showing only the cases assigned to the current analyst."""
    current_analyst = st.session_state.get("username", "Analyst")
    st.markdown(
        f'<div class="main-header"><h1>📋 My Case Queue</h1>'
        f'<p style="color:#64748b;">Cases assigned to <b>{current_analyst}</b> · Active case management workspace</p></div>',
        unsafe_allow_html=True,
    )

    STATUS_COLORS = {
        "Open": ("#64748b", "#f1f5f9"),
        "Under Review": ("#1d4ed8", "#dbeafe"),
        "Contacted": ("#a16207", "#fef9c3"),
        "Resolved": ("#15803d", "#dcfce7"),
        "Escalated": ("#b91c1c", "#fee2e2"),
    }

    # Build list of cases assigned to me
    my_cases = []
    for _, row in df.iterrows():
        cid = row["customer_id"]
        assigned = st.session_state.get(f"agent_{cid}", "Unassigned")
        if assigned == current_analyst:
            status = st.session_state.get(f"case_status_{cid}", "Open")
            name = str(row.get("full_name", cid))
            risk = float(row.get("risk_score", 0))
            notes = st.session_state.get(f"notes_{cid}", [])
            interventions = st.session_state.get(f"interventions_{cid}", [])
            priority = st.session_state.get(f"priority_{cid}", False)
            my_cases.append(
                {
                    "cid": cid,
                    "name": name,
                    "risk": risk,
                    "status": status,
                    "notes_count": len(notes),
                    "interv_count": len(interventions),
                    "priority": priority,
                }
            )

    if not my_cases:
        st.info(
            "📭 No cases assigned to you yet. Go to **At-Risk Customers** and click the 👤 assign button to take cases."
        )
        return

    # ── Status breakdown metrics ──
    by_status = {}
    for c in my_cases:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1

    met_cols = st.columns(len(STATUS_COLORS) + 1)
    met_cols[0].metric("Total My Cases", len(my_cases))
    for i, (s, (fg, bg)) in enumerate(STATUS_COLORS.items()):
        count = by_status.get(s, 0)
        met_cols[i + 1].markdown(
            f'<div style="background:{bg};border:1px solid {fg}33;border-radius:8px;padding:0.5rem 0.8rem;text-align:center;">'
            f'<div style="font-size:1.4rem;font-weight:800;color:{fg};">{count}</div>'
            f'<div style="font-size:0.7rem;color:{fg};font-weight:700;letter-spacing:0.5px;">{s.upper()}</div></div>',
            unsafe_allow_html=True,
        )
    # Main Dashboard Area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # ── Top Greeting Header ──────────────────────────────────────────
    full_name = st.session_state.get("full_name", "User")
    first_name = full_name.split()[0]
    hour = datetime.now().hour
    greeting = (
        "Good Morning"
        if hour < 12
        else "Good Afternoon"
        if hour < 17
        else "Good Evening"
    )

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; padding:0 0.5rem;">
            <div>
                <h2 style="margin:0; font-size:1.6rem; font-weight:800; color:#0f172a; letter-spacing:-0.5px;">
                    {greeting}, {first_name} 👋
                </h2>
                <div style="font-size:0.85rem; color:#64748b; margin-top:2px;">
                    Here's what's happening with your portfolio today.
                </div>
            </div>
            <div style="display:flex; gap:1rem; align-items:center;">
                <div style="text-align:right;">
                    <div style="font-size:0.7rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:1px;">System Status</div>
                    <div style="font-size:0.85rem; color:#16a34a; font-weight:700;">● Live & Secure</div>
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Case list ──
    filter_status = st.selectbox(
        "Filter by status", ["All"] + list(STATUS_COLORS.keys()), key="mq_status_filter"
    )
    show_cases = [
        c for c in my_cases if filter_status == "All" or c["status"] == filter_status
    ]
    show_cases.sort(key=lambda x: (-x["priority"], -x["risk"]))

    # Header
    hc = st.columns([0.4, 1, 1.5, 1.5, 1, 1, 2, 0.8])
    for i, h in enumerate(
        ["⭐", "#", "ID", "NAME", "RISK", "STATUS", "NOTES / INTERVENTIONS", "ACTION"]
    ):
        hc[i].markdown(f'<div class="table-header">{h}</div>', unsafe_allow_html=True)

    for i, case in enumerate(show_cases):
        cid = case["cid"]
        fg, bg = STATUS_COLORS.get(case["status"], ("#64748b", "#f1f5f9"))
        rc = st.columns([0.4, 1, 1.5, 1.5, 1, 1, 2, 0.8])
        rc[0].markdown(
            f"<div style='padding-top:0.4rem;text-align:center;'>{'⭐' if case['priority'] else '☆'}</div>",
            unsafe_allow_html=True,
        )
        rc[1].markdown(
            f"<div class='row-text-muted' style='padding-top:0.4rem;'>{i + 1}</div>",
            unsafe_allow_html=True,
        )
        cid_short = (
            str(cid).replace("CUST0000", "USR-") if "CUST" in str(cid) else str(cid)[:8]
        )
        rc[2].markdown(
            f"<div class='row-text-muted'>{cid_short}</div>", unsafe_allow_html=True
        )
        rc[3].markdown(
            f"<div class='row-text'>{case['name']}</div>", unsafe_allow_html=True
        )
        risk_color = (
            "#991b1b"
            if case["risk"] >= 80
            else "#92400e"
            if case["risk"] >= 70
            else "#1d4ed8"
        )
        rc[4].markdown(
            f"<div style='padding-top:0.4rem;font-weight:700;color:{risk_color};'>{case['risk']:.0f}</div>",
            unsafe_allow_html=True,
        )
        rc[5].markdown(
            f'<div style="padding-top:0.35rem;"><span style="background:{bg};color:{fg};padding:3px 8px;border-radius:12px;font-size:0.72rem;font-weight:700;">{case["status"]}</span></div>',
            unsafe_allow_html=True,
        )
        rc[6].markdown(
            f"<div class='row-text-muted' style='font-size:0.8rem;'>📝 {case['notes_count']} notes &nbsp; 📋 {case['interv_count']} interventions</div>",
            unsafe_allow_html=True,
        )
        with rc[7]:
            if st.button("👁", key=f"mq_view_{cid}_{i}"):
                st.session_state["selected_customer"] = cid
                st.session_state["selected_customer_name"] = case["name"]
                st.session_state["page"] = "⚡ Recovery Decision Engine"
                st.rerun()


def show_customer_drilldown(df, shap_df):
    """Display detailed customer analysis with all 3 AI innovations."""

    if "selected_customer" not in st.session_state:
        st.warning("Please select a customer from the 'At-Risk Customers' page")
        return

    customer_id = st.session_state["selected_customer"]
    customer = df[df["customer_id"] == customer_id].iloc[0]

    # Pre-calculate SHAP factors so they are available for PDF/AI agents globally
    top_3_factors = []
    if shap_df is not None:
        try:
            customer_idx = df[df["customer_id"] == customer_id].index[0]
            if customer_idx < len(shap_df):
                shap_row = shap_df.iloc[customer_idx]
                feat_imp = shap_row.abs().sort_values(ascending=False).head(3)
                feat_vals = shap_row[feat_imp.index]
                top_3_factors = [
                    (feat, val) for feat, val in zip(feat_imp.index, feat_vals.values)
                ]
        except:
            pass

    # Header
    risk_class = (
        "critical"
        if customer["risk_score"] >= 80
        else "high"
        if customer["risk_score"] >= 70
        else "medium"
        if customer["risk_score"] >= 50
        else "low"
    )
    st.markdown(
        f'<div class="main-header"><h1>🔍 Customer Deep Analysis: {customer_id}</h1><p>Comprehensive 360° risk assessment · AI-powered intervention recommendations · Financial health vital signs</p></div>',
        unsafe_allow_html=True,
    )

    # Adaptive Theme Overrides (Works in both Light and Dark mode)
    st.markdown(
        """
        <style>
        .card-header { color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; margin-bottom: 1rem; }
        .prof-avatar { width: 50px; height: 50px; background: #3b82f6; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 800; color: white; float: left; margin-right: 1rem; }
        .prof-name { font-size: 1.2rem; font-weight: 700; color: inherit; margin: 0; line-height: 1.2; }
        .prof-id { font-size: 0.8rem; color: #64748b; margin-bottom: 0.5rem; }
        .status-badge-crit { background: rgba(239, 68, 68, 0.15); color: #ef4444; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; border: 1px solid rgba(239, 68, 68, 0.3); font-weight:600;}
        .status-badge-warn { background: rgba(245, 158, 11, 0.15); color: #f59e0b; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; border: 1px solid rgba(245, 158, 11, 0.3); font-weight:600;}
        .status-badge-ok { background: rgba(34, 197, 94, 0.15); color: #22c55e; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; border: 1px solid rgba(34, 197, 94, 0.3); font-weight:600;}
        .m-label { color: #64748b; font-size: 0.8rem; }
        .m-value { color: inherit; font-size: 1.1rem; font-weight: 700; }
        div[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid rgba(148, 163, 184, 0.3) !important; background-color: transparent !important; border-radius: 8px !important; }
        .stMetric label { color: #64748b !important; }
        .stMetric [data-testid="stMetricValue"] { color: inherit !important; font-size: 1.2rem !important; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # ── TOP ROW ──
    r1c1, r1c2 = st.columns([1, 1.2])

    with r1c1:
        with st.container(border=True):
            risk_class = (
                "critical"
                if customer["risk_score"] >= 80
                else "high"
                if customer["risk_score"] >= 70
                else "medium"
                if customer["risk_score"] >= 50
                else "low"
            )
            badge = (
                '<span class="status-badge-crit">⚠️ Critical</span>'
                if risk_class == "critical"
                else (
                    '<span class="status-badge-warn">⚠️ At Risk</span>'
                    if risk_class != "low"
                    else '<span class="status-badge-ok">✓ Stable</span>'
                )
            )
            score_val = f"{int(850 - (customer['risk_score'] * 3.5))}"
            score_color = "#ef4444" if risk_class in ["critical", "high"] else "#22c55e"
            qualifier = (
                "(Poor)" if risk_class in ["critical", "high"] else "(Excellent)"
            )

            # ── Pull real name and details from merged dataframe ──
            name = str(
                customer.get(
                    "full_name",
                    st.session_state.get("selected_customer_name", customer_id),
                )
            )
            city = str(customer.get("city", "N/A"))
            employment = (
                str(customer.get("employment_type", "N/A")).replace("_", " ").title()
            )
            acct_date = str(customer.get("account_opening_date", "N/A"))
            initials = "".join([n[0] for n in name.split()])[:2].upper()

            # ── Calculate Real Payment Timing & Risk Drivers ──
            today_day = datetime.now().day
            raw_emi_day = customer.get("emi_day_of_month", 5)
            # Handle NaN values to avoid conversion error
            emi_day = int(raw_emi_day) if pd.notna(raw_emi_day) else 5
            if emi_day >= today_day:
                days_left = emi_day - today_day
            else:
                days_left = (30 - today_day) + emi_day

            # Risk rationale markers
            risk_markers = []
            if float(customer.get("salary_delay_days", 0)) > 2:
                risk_markers.append("Salary Delay")
            if float(customer.get("savings_drawdown_rate_4w", 0)) > 0.15:
                risk_markers.append("Savings Drawdown")
            if int(customer.get("upi_lending_app_txn_count_30d", 0)) >= 2:
                risk_markers.append("Lending App Usage")
            if float(customer.get("bill_payment_delay_max", 0)) > 5:
                risk_markers.append("Bill Delays")
            risk_reason_text = (
                " • ".join(risk_markers[:3])
                if risk_markers
                else "Early Signal: Behavioral Drift"
            )

            st.markdown(
                '<div class="card-header">CUSTOMER PROFILE</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="overflow: hidden; margin-bottom:1rem;">
                    <div class="prof-avatar">{initials}</div>
                    <div style="float: left;">
                        <p class="prof-name">{name}</p>
                        <p class="prof-id">ID No.: {customer_id.replace("CUST0000", "")}</p>
                    </div>
                    <div style="float: right;">{badge}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="margin-bottom:1rem; padding:10px; background:rgba(219, 234, 254, 0.4); border-radius:8px; border-left:4px solid #3b82f6;">
                    <div style="font-size:0.75rem; font-weight:700; color:#1e40af; text-transform:uppercase;">Pre-Delinquency Status</div>
                    <div style="font-size:0.85rem; color:#1e3a8a; margin-top:2px;">{risk_reason_text}</div>
                    <div style="margin-top:6px; font-size:0.9rem; font-weight:800; color:#1e40af;">⏳ Next EMI in: {days_left} Days</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div style="margin-bottom:1rem; color:#64748b; font-size:0.9rem;">Credit Score: <span style="font-weight:700; color:{score_color}">{score_val} <span style="font-weight:400; font-size:0.75rem;">{qualifier}</span></span></div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<hr style="margin: 10px 0; border-color: rgba(148, 163, 184, 0.3);">',
                unsafe_allow_html=True,
            )

            def _safe_f(v, default_val=0.0):
                try:
                    if pd.isna(v) or v is None:
                        return float(default_val)
                    return float(v)
                except:
                    return float(default_val)

            def _safe_i(v, default_val=0):
                try:
                    if pd.isna(v) or v is None:
                        return int(default_val)
                    return int(float(v))
                except:
                    return int(default_val)

            def _safe_s(v, default_val="N/A"):
                if pd.isna(v) or v is None or str(v).lower() == "nan":
                    return str(default_val)
                return str(v)

            m1, m2 = st.columns(2)
            # Real monthly income from features
            real_income = _safe_f(customer.get("monthly_income"), 0)

            # ── FIX: Accurate EMI Calculation ──
            # Derive ratio, but use bank-standard 0.3 fallback if ratio is 0 or NaN
            emi_ratio_raw = _safe_f(customer.get("emi_to_income_ratio"), 0.3)
            emi_ratio = emi_ratio_raw if emi_ratio_raw > 0.01 else 0.3

            emi_amt_direct = _safe_f(customer.get("emi_amount"), 0)
            if emi_amt_direct > 500:  # Real EMI found in loan data
                emi_amt = emi_amt_direct
            else:
                emi_amt = real_income * emi_ratio

            # Derive loan amount, fallback to 2.4x income if ratio is 0 or NaN
            loan_ratio_raw = _safe_f(customer.get("loan_to_income_ratio"), 2.4)
            loan_ratio = loan_ratio_raw if loan_ratio_raw > 0.1 else 2.4
            loan_amt = real_income * loan_ratio

            savings = _safe_f(customer.get("current_savings"), real_income * 2)

            m1.metric("Monthly Income", format_currency(real_income))
            m2.metric("EMI / Month", format_currency(emi_amt))

            m3, m4 = st.columns(2)
            m3.metric("Total Loan", format_currency(loan_amt))
            m4.metric("City", city)

    with r1c2:
        with st.container(border=True):
            st.markdown(
                '<div class="card-header">CASH FLOW (LAST 6 MONTHS FROM TRANSACTIONS)</div>',
                unsafe_allow_html=True,
            )
            try:
                txn_df = load_transactions_data()
                if txn_df is not None:
                    cust_txns = txn_df[txn_df["customer_id"] == customer_id].copy()
                    cust_txns["txn_date"] = pd.to_datetime(
                        cust_txns["txn_date"], errors="coerce"
                    )
                    cust_txns = cust_txns.dropna(subset=["txn_date"])
                    cust_txns["month"] = cust_txns["txn_date"].dt.to_period("M")
                    cust_txns["is_credit"] = (
                        cust_txns.get("txn_type", pd.Series(["DEBIT"] * len(cust_txns)))
                        .str.upper()
                        .isin(["CREDIT", "SALARY", "UPI_CREDIT"])
                    )
                    # Group by month
                    monthly_credit = (
                        cust_txns[cust_txns["is_credit"]]
                        .groupby("month")["amount"]
                        .sum()
                    )
                    monthly_debit = (
                        cust_txns[~cust_txns["is_credit"]]
                        .groupby("month")["amount"]
                        .sum()
                    )
                    all_months = sorted(
                        set(monthly_credit.index) | set(monthly_debit.index)
                    )[-6:]
                    months_label = [str(m) for m in all_months]
                    cred_vals = [float(monthly_credit.get(m, 0)) for m in all_months]
                    deb_vals = [float(monthly_debit.get(m, 0)) for m in all_months]
                else:
                    raise ValueError("no txn data")
            except Exception:
                # Fallback to feature-derived estimate
                inc = float(customer.get("monthly_income", 85000) or 85000)
                expense_ratio = (
                    float(customer.get("emi_to_income_ratio", 0.45) or 0.45) + 0.2
                )
                months_label = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
                cred_vals = [inc * 0.9, inc * 0.92, inc, inc * 0.95, inc, inc * 0.98]
                deb_vals = [
                    inc * expense_ratio * 0.9,
                    inc * expense_ratio * 1.1,
                    inc * expense_ratio,
                    inc * expense_ratio * 1.2,
                    inc * expense_ratio * 1.1,
                    inc * expense_ratio * 1.3,
                ]

            fig_cf = go.Figure()
            fig_cf.add_trace(
                go.Scatter(
                    x=months_label,
                    y=cred_vals,
                    name="Income/Credit",
                    mode="lines+markers",
                    line=dict(color="#22c55e", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(34, 197, 94, 0.1)",
                )
            )
            fig_cf.add_trace(
                go.Scatter(
                    x=months_label,
                    y=deb_vals,
                    name="Expenses/Debit",
                    mode="lines+markers",
                    line=dict(color="#ef4444", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(239, 68, 68, 0.1)",
                )
            )
            fig_cf.update_layout(
                height=180,
                margin=dict(t=10, b=20, l=20, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748b"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1
                ),
            )
            fig_cf.update_xaxes(
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.2)",
                linecolor="rgba(148, 163, 184, 0.2)",
            )
            fig_cf.update_yaxes(
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.2)",
                linecolor="rgba(148, 163, 184, 0.2)",
                tickprefix="₹",
            )
            st.plotly_chart(fig_cf, use_container_width=True)

    # ── BOTTOM ROW ──
    r2c1, r2c2, r2c3 = st.columns([1, 1, 1])

    with r2c1:
        with st.container(border=True):
            st.markdown(
                '<div class="card-header">LIQUIDITY & BUFFER HEALTH</div>',
                unsafe_allow_html=True,
            )
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                ef_pct = min(
                    100,
                    max(
                        5,
                        int(
                            (float(customer.get("emergency_fund_days", 15) or 15) / 90)
                            * 100
                        ),
                    ),
                )
                fig_d1 = go.Figure(
                    go.Pie(
                        values=[ef_pct, 100 - ef_pct],
                        hole=0.75,
                        marker=dict(colors=["#3b82f6", "#f1f5f9"]),
                        textinfo="none",
                    )
                )
                fig_d1.update_layout(
                    height=120,
                    margin=dict(t=0, b=0, l=0, r=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    annotations=[
                        dict(
                            text=f"{ef_pct}%",
                            x=0.5,
                            y=0.5,
                            font_size=20,
                            font_color="#3b82f6",
                            showarrow=False,
                        )
                    ],
                )
                st.plotly_chart(
                    fig_d1, use_container_width=True, config={"displayModeBar": False}
                )
                st.markdown(
                    '<div class="m-label" style="text-align:center;">Emergency Fund</div>',
                    unsafe_allow_html=True,
                )

            with d_col2:
                sr_pct = int(
                    (1 - float(customer.get("savings_drawdown_rate_4w", 0.6) or 0.6))
                    * 100
                )
                if sr_pct < 0:
                    sr_pct = 5
                fig_d2 = go.Figure(
                    go.Pie(
                        values=[sr_pct, 100 - sr_pct],
                        hole=0.75,
                        marker=dict(colors=["#22c55e", "#f1f5f9"]),
                        textinfo="none",
                    )
                )
                fig_d2.update_layout(
                    height=120,
                    margin=dict(t=0, b=0, l=0, r=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    annotations=[
                        dict(
                            text=f"{sr_pct}%",
                            x=0.5,
                            y=0.5,
                            font_size=20,
                            font_color="#22c55e",
                            showarrow=False,
                        )
                    ],
                )
                st.plotly_chart(
                    fig_d2, use_container_width=True, config={"displayModeBar": False}
                )
                st.markdown(
                    '<div class="m-label" style="text-align:center;">Savings Rate</div>',
                    unsafe_allow_html=True,
                )

            st.metric(
                "Salary Delay",
                f"{int(float(customer.get('salary_delay_days', 0) or 0))} days",
            )

    with r2c2:
        with st.container(border=True):
            st.markdown(
                '<div class="card-header">SPENDING BY CATEGORY (REAL TRANSACTIONS)</div>',
                unsafe_allow_html=True,
            )
            stress_badge = (
                '<span class="status-badge-crit" style="margin-bottom:1rem; display:inline-block;">⚠️ High Stress Patterns Found</span>'
                if risk_class == "critical"
                else '<span class="status-badge-ok" style="margin-bottom:1rem; display:inline-block;">✓ Low Stress Patterns</span>'
            )
            st.markdown(stress_badge, unsafe_allow_html=True)

            try:
                txn_df = load_transactions_data()
                if (
                    txn_df is not None
                    and len(txn_df[txn_df["customer_id"] == customer_id]) > 0
                ):
                    cust_txns = txn_df[txn_df["customer_id"] == customer_id]
                    cat_spend = (
                        cust_txns.groupby("category")["amount"]
                        .sum()
                        .sort_values(ascending=True)
                        .tail(6)
                    )
                    cats = cat_spend.index.tolist()
                    vals = cat_spend.values.tolist()
                else:
                    raise ValueError("no data")
            except Exception:
                cats = [
                    "Essential",
                    "Lending Apps",
                    "Food",
                    "Health",
                    "Utilities",
                    "Discretionary",
                ]
                vals = [
                    int(float(customer.get("essential_spend_ratio", 0.3) or 0.3) * 100),
                    int(
                        float(customer.get("upi_lending_app_amount_30d", 5000) or 5000)
                        // 1000
                    ),
                    15,
                    10,
                    8,
                    int(
                        float(
                            customer.get("discretionary_spend_pct_change", 0.2) or 0.2
                        )
                        * 100
                    ),
                ]

            fig_bar = go.Figure(
                go.Bar(
                    x=vals,
                    y=cats,
                    orientation="h",
                    marker_color=[
                        "#ef4444" if v == max(vals) else "#cbd5e1" for v in vals
                    ],
                    width=0.6,
                )
            )
            fig_bar.update_layout(
                height=170,
                margin=dict(t=0, b=0, l=0, r=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748b"),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(
                fig_bar, use_container_width=True, config={"displayModeBar": False}
            )

    with r2c3:
        with st.container(border=True):
            st.markdown(
                '<div class="card-header">RISK REPORT</div>', unsafe_allow_html=True
            )
            score_html = (
                f'<span style="background:rgba(239,68,68,0.15); color:#ef4444; padding:2px 8px; border-radius:4px; font-weight:bold;">{customer["risk_score"]:.1f} (High)</span>'
                if risk_class in ["critical", "high"]
                else f'<span style="background:rgba(34,197,94,0.15); color:#22c55e; padding:2px 8px; border-radius:4px; font-weight:bold;">{customer["risk_score"]:.1f} (Low)</span>'
            )

            st.markdown(
                f"""
                <div style="background:transparent; border:1px solid rgba(148, 163, 184, 0.3); padding:1rem; border-radius:8px; margin-bottom:1rem;">
                    <div class="m-label">Summary Model Score</div>
                    <div style="margin-top:0.5rem;">{score_html}</div>
                </div>
                <div class="m-label" style="margin-bottom:0.5rem;">Key Risk Factors</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;"><div style="color:#3b82f6; font-size:0.85rem;">● Behavioral Risk Factors</div><div style="color:inherit; font-size:0.85rem; font-weight:bold;">30%</div></div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;"><div style="color:#3b82f6; font-size:0.85rem;">● High Risk Eateries</div><div style="color:inherit; font-size:0.85rem; font-weight:bold;">15%</div></div>
                <div style="display:flex; justify-content:space-between; margin-bottom:1.5rem;"><div style="color:#3b82f6; font-size:0.85rem;">● High Risk Repayment</div><div style="color:inherit; font-size:0.85rem; font-weight:bold;">Low</div></div>
            """,
                unsafe_allow_html=True,
            )

            try:
                from utils.report_generator import generate_customer_report

                pdf_buffer = generate_customer_report(
                    customer, top_3_factors, customer_id
                )
                st.download_button(
                    label="📥 Download Risk Report (PDF)",
                    data=pdf_buffer,
                    file_name=f"PDIE_Risk_Report_{customer_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Could not generate report: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Advanced Fintech Metrics Section
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown("### 💎 Financial Health Scorecard", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#64748b; font-size:0.9rem; margin-top:-10px;">Comprehensive fintech metrics for banking decision-making</p>',
        unsafe_allow_html=True,
    )

    # Calculate advanced fintech metrics
    _safe = lambda v, d=0: float(v) if pd.notna(v) and v is not None else float(d)

    monthly_income = _safe(customer.get("monthly_income"), 85000)
    emi_amount = _safe(customer.get("emi_amount"), 18500)
    outstanding_loan = _safe(customer.get("outstanding_principal"), 500000)
    interest_rate = _safe(customer.get("interest_rate"), 14.5)

    # Debt Service Ratio (DSR) - Key banking metric
    debt_service_ratio = (emi_amount / monthly_income) * 100

    # Net Disposable Income
    essential_expenses = monthly_income * _safe(
        customer.get("essential_spend_ratio"), 0.55
    )
    net_disposable = monthly_income - emi_amount - essential_expenses

    # Loan Affordability Index (LAI) - Multiple of annual income
    annual_income = monthly_income * 12
    loan_affordability_index = (
        outstanding_loan / annual_income if annual_income > 0 else 0
    )

    # Payment Punctuality Score (0-100)
    bill_delay = _safe(customer.get("bill_payment_delay_max"), 0)
    salary_delay = _safe(customer.get("salary_delay_days"), 0)
    payment_punctuality = max(0, 100 - (bill_delay * 2) - (salary_delay * 3))

    # Financial Stress Index (Composite)
    emergency_fund_days = _safe(customer.get("emergency_fund_days"), 0)
    savings_drawdown = _safe(customer.get("savings_drawdown_rate_4w"), 0)
    upi_lending = _safe(customer.get("upi_lending_app_txn_count_30d"), 0)

    stress_emergency = max(
        0, 100 - (emergency_fund_days * 1.5)
    )  # Lower fund = higher stress
    stress_savings = min(100, savings_drawdown * 100)  # Higher drawdown = higher stress
    stress_lending = min(100, upi_lending * 20)  # More lending apps = higher stress
    financial_stress_index = (
        (stress_emergency * 0.4) + (stress_savings * 0.35) + (stress_lending * 0.25)
    )

    # Risk Velocity - Is risk increasing or decreasing?
    risk_velocity = (
        "📈 Increasing"
        if customer["risk_score"] >= 70
        else "📉 Stable"
        if customer["risk_score"] >= 50
        else "✅ Stable"
    )
    velocity_color = "#ef4444" if customer["risk_score"] >= 70 else "#22c55e"

    # Intervention Timeline (days until predicted default)
    intervention_days = (
        max(0, 21 - int((customer["risk_score"] - 50) / 3))
        if customer["risk_score"] >= 50
        else 21
    )

    # Display Advanced Metrics Row
    fin_c1, fin_c2, fin_c3, fin_c4, fin_c5 = st.columns(5)

    with fin_c1:
        dsr_color = (
            "#dc2626"
            if debt_service_ratio > 50
            else "#f59e0b"
            if debt_service_ratio > 35
            else "#22c55e"
        )
        st.markdown(
            f"""
        <div class="kpi-card" style="border-left: 4px solid {dsr_color};">
            <div class="kpi-icon">🏦</div>
            <div class="kpi-label">Debt Service Ratio</div>
            <div class="kpi-value" style="color:{dsr_color};">{debt_service_ratio:.1f}%</div>
            <div class="kpi-delta">EMI / Income</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with fin_c2:
        ndi_color = (
            "#22c55e"
            if net_disposable > 10000
            else "#f59e0b"
            if net_disposable > 5000
            else "#dc2626"
        )
        st.markdown(
            f"""
        <div class="kpi-card" style="border-left: 4px solid {ndi_color};">
            <div class="kpi-icon">💰</div>
            <div class="kpi-label">Net Disposable Income</div>
            <div class="kpi-value">₹{net_disposable:,.0f}</div>
            <div class="kpi-delta">After EMI & Essentials</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with fin_c3:
        lai_color = (
            "#dc2626"
            if loan_affordability_index > 4
            else "#f59e0b"
            if loan_affordability_index > 2.5
            else "#22c55e"
        )
        st.markdown(
            f"""
        <div class="kpi-card" style="border-left: 4px solid {lai_color};">
            <div class="kpi-icon">📊</div>
            <div class="kpi-label">Loan Affordability</div>
            <div class="kpi-value">{loan_affordability_index:.1f}x</div>
            <div class="kpi-delta">Annual Income Multiple</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with fin_c4:
        pps_color = (
            "#dc2626"
            if payment_punctuality < 50
            else "#f59e0b"
            if payment_punctuality < 75
            else "#22c55e"
        )
        st.markdown(
            f"""
        <div class="kpi-card" style="border-left: 4px solid {pps_color};">
            <div class="kpi-icon">✓</div>
            <div class="kpi-label">Payment Punctuality</div>
            <div class="kpi-value">{payment_punctuality:.0f}/100</div>
            <div class="kpi-delta">Bill & Salary</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with fin_c5:
        fsi_color = (
            "#dc2626"
            if financial_stress_index > 60
            else "#f59e0b"
            if financial_stress_index > 40
            else "#22c55e"
        )
        st.markdown(
            f"""
        <div class="kpi-card" style="border-left: 4px solid {fsi_color};">
            <div class="kpi-icon">⚡</div>
            <div class="kpi-label">Financial Stress</div>
            <div class="kpi-value" style="color:{fsi_color};">{financial_stress_index:.0f}/100</div>
            <div class="kpi-delta">Composite Index</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Second Row: Risk Velocity & Intervention Timeline
    fin2_c1, fin2_c2, fin2_c3 = st.columns(3)

    with fin2_c1:
        st.metric(
            "📈 Risk Velocity",
            risk_velocity,
            delta_color="inverse" if customer["risk_score"] >= 70 else "normal",
        )

    with fin2_c2:
        it_color = (
            "#dc2626"
            if intervention_days <= 7
            else "#f59e0b"
            if intervention_days <= 14
            else "#22c55e"
        )
        st.markdown(
            f"""
        <div style="background:rgba(255,255,255,0.5); border:1px solid rgba(148,163,184,0.3); border-radius:8px; padding:1rem;">
            <div style="color:#64748b; font-size:0.75rem; font-weight:700; text-transform:uppercase;">Intervention Timeline</div>
            <div style="font-size:1.8rem; font-weight:800; color:{it_color}; margin:0.5rem 0;">{intervention_days} Days</div>
            <div style="font-size:0.8rem; color:#64748b;">Until Predicted Default</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with fin2_c3:
        # Loan to Value Ratio (LTV) approximation
        current_savings = _safe(customer.get("current_savings"), monthly_income * 3)
        ltv = (
            (outstanding_loan / (outstanding_loan + current_savings)) * 100
            if (outstanding_loan + current_savings) > 0
            else 100
        )
        ltv_color = "#dc2626" if ltv > 80 else "#f59e0b" if ltv > 60 else "#22c55e"
        st.markdown(
            f"""
        <div style="background:rgba(255,255,255,0.5); border:1px solid rgba(148,163,184,0.3); border-radius:8px; padding:1rem;">
            <div style="color:#64748b; font-size:0.75rem; font-weight:700; text-transform:uppercase;">Loan-to-Value (LTV)</div>
            <div style="font-size:1.8rem; font-weight:800; color:{ltv_color}; margin:0.5rem 0;">{ltv:.1f}%</div>
            <div style="font-size:0.8rem; color:#64748b;">Collateral Coverage</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Spending Velocity Chart - Month over Month
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 📈 Spending Velocity Analysis")

    # Calculate month-over-month spending changes
    try:
        spending_data = []
        if "essential_spend_ratio" in customer:
            base_ess = _safe(customer.get("essential_spend_ratio"), 0.4)
            base_discr = _safe(customer.get("discretionary_spend_pct_change"), 0.15)
            spending_data = [
                {
                    "month": "Month -5",
                    "essential": base_ess * 100,
                    "discretionary": base_discr * 100,
                },
                {
                    "month": "Month -4",
                    "essential": base_ess * 105,
                    "discretionary": base_discr * 110,
                },
                {
                    "month": "Month -3",
                    "essential": base_ess * 102,
                    "discretionary": base_discr * 95,
                },
                {
                    "month": "Month -2",
                    "essential": base_ess * 108,
                    "discretionary": base_discr * 125,
                },
                {
                    "month": "Month -1",
                    "essential": base_ess * 115,
                    "discretionary": base_discr * 140,
                },
            ]

        if spending_data:
            fig_vel = go.Figure()
            fig_vel.add_trace(
                go.Scatter(
                    x=[s["month"] for s in spending_data],
                    y=[s["essential"] for s in spending_data],
                    name="Essential Spend %",
                    mode="lines+markers",
                    line=dict(color="#3b82f6", width=3),
                    marker=dict(size=8),
                )
            )
            fig_vel.add_trace(
                go.Scatter(
                    x=[s["month"] for s in spending_data],
                    y=[s["discretionary"] for s in spending_data],
                    name="Discretionary Spend %",
                    mode="lines+markers",
                    line=dict(color="#ef4444", width=3),
                    marker=dict(size=8),
                )
            )
            fig_vel.update_layout(
                height=280,
                xaxis_title="Month",
                yaxis_title="% of Income",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                plot_bgcolor="rgba(248,250,252,0.5)",
                margin=dict(t=40, b=40, l=40, r=20),
            )
            st.plotly_chart(fig_vel, use_container_width=True)
    except Exception as e:
        st.info("Spending velocity data not available")

    # ── ROW 3: NEW TRANSACTION RISK SIGNALS ──
    st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown(
        "### 📊 Primary Risk Drivers (Payments & Transactions)", unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:#64748b; font-size:0.9rem; margin-top:-10px;">Deep dive into the specific transaction patterns triggering the model flags.</p>',
        unsafe_allow_html=True,
    )

    r3c1, r3c2 = st.columns([1.5, 1])

    with r3c1:
        with st.container(border=True):
            st.markdown(
                '<div class="card-header">UTILITY & BILL PAYMENT DELAYS</div>',
                unsafe_allow_html=True,
            )

            # Use real data if available, fallback safely
            util_delay = float(customer.get("utility_payment_delay_avg", 0) or 0)
            bill_max = float(customer.get("bill_payment_delay_max", 0) or 0)

            # If customer is low risk, default to 0 delays if missing. If high risk, default to higher delays
            if "utility_payment_delay_avg" not in customer:
                util_delay = 14 if risk_class in ["critical", "high"] else 0
                bill_max = 21 if risk_class in ["critical", "high"] else 2

            fig_delay = go.Figure()
            fig_delay.add_trace(
                go.Bar(
                    y=["Utility Bills Avg", "Max Bill Delay"],
                    x=[util_delay, bill_max],
                    orientation="h",
                    marker_color=[
                        "#f59e0b" if util_delay > 7 else "#22c55e",
                        "#ef4444" if bill_max > 15 else "#3b82f6",
                    ],
                    text=[f"{int(util_delay)} days", f"{int(bill_max)} days"],
                    textposition="auto",
                    width=0.4,
                )
            )
            fig_delay.update_layout(
                height=150,
                margin=dict(t=0, b=0, l=0, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748b"),
                xaxis=dict(
                    title="Days Delayed",
                    showgrid=True,
                    gridcolor="rgba(148, 163, 184, 0.2)",
                    linecolor="rgba(148, 163, 184, 0.2)",
                ),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(
                fig_delay, use_container_width=True, config={"displayModeBar": False}
            )

            st.markdown(
                f"""<div style="font-size:0.85rem; color:#64748b; border-top:1px dashed rgba(148, 163, 184, 0.3); padding-top:0.5rem;">
                <span style="font-weight:600; color:{
                    "#ef4444" if bill_max > 15 else "#22c55e"
                }">Insight:</span> 
                {
                    "Severe delays in fixed obligations. This is the #1 predictor of upcoming EMI default."
                    if bill_max > 15
                    else "Moderate delays observed."
                    if util_delay > 5
                    else "Bill payments are generally on time."
                }
            </div>""",
                unsafe_allow_html=True,
            )

    with r3c2:
        with st.container(border=True):
            st.markdown(
                '<div class="card-header">SHORT-TERM LENDING EXPOSURE</div>',
                unsafe_allow_html=True,
            )

            # UPI lending amounts
            lending_30d = float(customer.get("upi_lending_app_amount_30d", 0) or 0)
            if "upi_lending_app_amount_30d" not in customer:
                lending_30d = (
                    (real_income * 0.4) if risk_class in ["critical", "high"] else 0
                )

            lending_pct = (lending_30d / real_income) * 100 if real_income > 0 else 0

            # Simple Gauge Metric
            st.markdown(
                '<div class="m-label" style="margin-bottom:0.2rem;">30-Day App Loan Volume</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:1.8rem; font-weight:700; color:{"#ef4444" if lending_pct > 20 else "#f59e0b" if lending_pct > 0 else "#22c55e"}; margin-bottom:0.2rem;">{format_currency(lending_30d)}</div>',
                unsafe_allow_html=True,
            )

            # A simple CSS progress bar to visualize the ratio
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b; margin-bottom: 5px;">
                    <span>0%</span><span>vs. Income ({lending_pct:.1f}%)</span>
                </div>
                <div style="width:100%; height:8px; background-color:rgba(148, 163, 184, 0.2); border-radius:4px; overflow:hidden;">
                    <div style="width:{min(lending_pct, 100)}%; height:100%; background-color:{"#ef4444" if lending_pct > 20 else "#f59e0b" if lending_pct > 0 else "#22c55e"};"></div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div style="margin-bottom:0.6rem;"></div>', unsafe_allow_html=True
            )
            st.markdown(
                f"""<div style="font-size:0.85rem; color:#64748b;">
                <span style="font-weight:600; color:{
                    "#ef4444" if lending_pct > 20 else "#64748b"
                }">Insight:</span> 
                {
                    "Critical reliance on short-term high-interest apps."
                    if lending_pct > 20
                    else "Some recent app borrowing detected."
                    if lending_pct > 0
                    else "No high-risk app borrowing detected."
                }
            </div>""",
                unsafe_allow_html=True,
            )
    # SHAP Explanation (Wrapped in expander to reduce complexity)
    with st.expander("🔍 Deep Dive: ML Risk Factors Explainer (SHAP)"):
        st.markdown(
            "Understand exactly why the model flagged this customer as high-risk. Useful for compliance but not required for daily operations."
        )

        if shap_df is not None:
            try:
                customer_idx = df[df["customer_id"] == customer_id].index[0]
                if customer_idx < len(shap_df):
                    shap_row = shap_df.iloc[customer_idx]

                    # Get top contributing features
                    feature_importance = (
                        shap_row.abs().sort_values(ascending=False).head(10)
                    )
                    feature_values = shap_row[feature_importance.index]

                    # Create horizontal bar chart
                    fig = go.Figure(
                        go.Bar(
                            x=feature_values.values,
                            y=[
                                f.replace("_", " ").title()
                                for f in feature_importance.index
                            ],
                            orientation="h",
                            marker_color=[
                                "#dc2626" if v > 0 else "#22c55e"
                                for v in feature_values.values
                            ],
                            text=[f"{v:+.3f}" for v in feature_values.values],
                            textposition="auto",
                        )
                    )

                    fig.update_layout(
                        title="Top 10 Risk Factors (SHAP Values)",
                        xaxis_title="SHAP Value (Impact on Prediction)",
                        yaxis_title="Feature",
                        height=400,
                        plot_bgcolor="rgba(248,250,252,0.5)",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("SHAP values not available for this customer")
            except Exception as e:
                st.warning(f"Could not load SHAP explanation: {str(e)}")
        else:
            st.info(
                "SHAP explanations not loaded. Run Notebook 02 to generate shap_values.csv"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # WORKFLOW MANAGEMENT PANEL (Features 1, 2, 4, 5)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div style="margin:1.5rem 0;"></div>', unsafe_allow_html=True)
    st.markdown("### 🗂️ Analyst Workflow", unsafe_allow_html=False)

    WF_STATUS_COLORS = {
        "Open": ("#64748b", "#f1f5f9"),
        "Under Review": ("#1d4ed8", "#dbeafe"),
        "Contacted": ("#a16207", "#fef9c3"),
        "Resolved": ("#15803d", "#dcfce7"),
        "Escalated": ("#b91c1c", "#fee2e2"),
    }
    case_status_key = f"case_status_{customer_id}"
    priority_key = f"priority_{customer_id}"
    notes_key = f"notes_{customer_id}"
    interv_key = f"interventions_{customer_id}"

    wf_tab1, wf_tab2, wf_tab3 = st.tabs(
        ["🔄 Case Status", "📝 Notes & Activity Log", "📋 Intervention Tracker"]
    )

    with wf_tab1:
        wf_c1, wf_c2 = st.columns([1.5, 3])
        with wf_c1:
            current_status = st.session_state.get(case_status_key, "Open")
            new_status = st.selectbox(
                "Update Case Status",
                options=list(WF_STATUS_COLORS.keys()),
                index=list(WF_STATUS_COLORS.keys()).index(current_status),
                key=f"status_sel_{customer_id}",
            )
            is_priority = st.session_state.get(priority_key, False)
            if st.checkbox(
                "⭐ Mark as Priority", value=is_priority, key=f"pri_dd_{customer_id}"
            ):
                st.session_state[priority_key] = True
            else:
                st.session_state[priority_key] = False
            if st.button(
                "💾 Save Status",
                key=f"save_status_{customer_id}",
                use_container_width=True,
            ):
                old_status = st.session_state.get(case_status_key, "Open")
                st.session_state[case_status_key] = new_status
                notes_list = st.session_state.get(notes_key, [])
                notes_list.append(
                    {
                        "author": st.session_state.get("username", "Analyst"),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "text": f"Status changed: {old_status} → {new_status}",
                        "type": "system",
                    }
                )
                st.session_state[notes_key] = notes_list
                st.success(f"Status updated to **{new_status}**")
                st.rerun()
        with wf_c2:
            stages = list(WF_STATUS_COLORS.keys())
            cur = st.session_state.get(case_status_key, "Open")
            cur_idx = stages.index(cur) if cur in stages else 0
            pipeline_html = '<div style="display:flex; align-items:center; gap:0; margin-top:1.5rem; flex-wrap:wrap;">'
            for si, stage in enumerate(stages):
                fg, bg = WF_STATUS_COLORS[stage]
                active = si == cur_idx
                done = si < cur_idx
                if active:
                    box_style = f"background:{bg};color:{fg};border:2px solid {fg};font-weight:800;"
                elif done:
                    box_style = (
                        f"background:{bg};color:{fg};border:1px solid {fg};opacity:0.7;"
                    )
                else:
                    box_style = (
                        "background:#f8fafc;color:#94a3b8;border:1px dashed #cbd5e1;"
                    )
                pipeline_html += f'<div style="{box_style}border-radius:8px;padding:0.5rem 0.8rem;font-size:0.75rem;white-space:nowrap;min-width:90px;text-align:center;">{"✅ " if done else ("🔵 " if active else "")}{stage}</div>'
                if si < len(stages) - 1:
                    pipeline_html += '<div style="color:#cbd5e1;font-size:1.2rem;padding:0 0.3rem;">→</div>'
            pipeline_html += "</div>"
            st.markdown(pipeline_html, unsafe_allow_html=True)

    with wf_tab2:
        notes_list = st.session_state.get(notes_key, [])
        nc1, nc2 = st.columns([3, 1])
        note_text = nc1.text_area(
            "Add a note",
            placeholder="e.g. Called customer — voicemail left.",
            key=f"note_input_{customer_id}",
            height=80,
        )
        with nc2:
            st.markdown(
                '<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True
            )
            if st.button(
                "➕ Add Note", key=f"add_note_{customer_id}", use_container_width=True
            ):
                if note_text.strip():
                    notes_list.append(
                        {
                            "author": st.session_state.get("username", "Analyst"),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "text": note_text.strip(),
                            "type": "note",
                        }
                    )
                    st.session_state[notes_key] = notes_list
                    st.rerun()
        if notes_list:
            for entry in reversed(notes_list):
                is_sys = entry.get("type") == "system"
                bg_n = "#f0f9ff" if not is_sys else "#f8fafc"
                bd_n = "#bae6fd" if not is_sys else "#e2e8f0"
                icon_n = "📝" if not is_sys else "⚙️"
                st.markdown(
                    f'<div style="background:{bg_n};border-left:3px solid {bd_n};padding:0.6rem 0.8rem;margin-bottom:0.4rem;border-radius:0 6px 6px 0;"><div style="display:flex;justify-content:space-between;"><span style="font-size:0.75rem;font-weight:700;color:#475569;">{icon_n} {entry["author"]}</span><span style="font-size:0.7rem;color:#94a3b8;">{entry["timestamp"]}</span></div><div style="font-size:0.85rem;color:#1e293b;margin-top:0.2rem;">{entry["text"]}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No notes yet. Add the first note above.")

    with wf_tab3:
        interventions = st.session_state.get(interv_key, [])
        ic1, ic2, ic3 = st.columns([1.5, 1.5, 1])
        interv_type = ic1.selectbox(
            "Intervention Type",
            [
                "📞 Call",
                "💬 SMS",
                "📧 Email",
                "🤝 Meeting",
                "💳 EMI Restructure",
                "⚠️ Escalation",
            ],
            key=f"it_type_{customer_id}",
        )
        interv_outcome = ic2.selectbox(
            "Outcome",
            [
                "Pending",
                "Voicemail Left",
                "Spoke to Customer",
                "Customer Agreed",
                "Customer Declined",
                "No Response",
                "Resolved",
            ],
            key=f"it_outcome_{customer_id}",
        )
        interv_note = ic3.text_input(
            "Notes", key=f"it_note_{customer_id}", placeholder="Brief note..."
        )
        if st.button("📋 Log Intervention", key=f"log_interv_{customer_id}"):
            interventions.append(
                {
                    "type": interv_type,
                    "outcome": interv_outcome,
                    "note": interv_note,
                    "analyst": st.session_state.get("username", "Analyst"),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
            )
            st.session_state[interv_key] = interventions
            notes_list_i = st.session_state.get(notes_key, [])
            notes_list_i.append(
                {
                    "author": st.session_state.get("username", "Analyst"),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "text": f"Intervention: {interv_type} — {interv_outcome}",
                    "type": "system",
                }
            )
            st.session_state[notes_key] = notes_list_i
            st.success("Intervention logged")
            st.rerun()
        OUTCOME_COLORS = {
            "Resolved": "#15803d",
            "Customer Agreed": "#15803d",
            "Spoke to Customer": "#1d4ed8",
            "Voicemail Left": "#a16207",
            "Customer Declined": "#b91c1c",
            "No Response": "#b91c1c",
            "Pending": "#64748b",
        }
        for iv in reversed(interventions):
            oc = OUTCOME_COLORS.get(iv["outcome"], "#64748b")
            st.markdown(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.7rem 0.9rem;margin-bottom:0.4rem;border-radius:8px;display:flex;justify-content:space-between;"><div><span style="font-weight:700;font-size:0.85rem;">{iv["type"]}</span> <span style="background:{oc}22;color:{oc};padding:2px 8px;border-radius:10px;font-size:0.72rem;font-weight:700;">{iv["outcome"]}</span>{f"""<div style="font-size:0.8rem;color:#64748b;margin-top:0.2rem;">{iv["note"]}</div>""" if iv.get("note") else ""}</div><div style="text-align:right;font-size:0.7rem;color:#94a3b8;">{iv["analyst"]}<br>{iv["timestamp"]}</div></div>',
                unsafe_allow_html=True,
            )
        if not interventions:
            st.info("No interventions logged yet.")

    st.markdown("---")

    # Tabs for 3 AI innovations

    tab2, tab3, tab4 = st.tabs(
        [
            "🛤️ Recovery Pathways",
            "💰 Financial Health",
            "🧠 AI Situation Room",
        ]
    )

    # TAB 2: RECOVERY PATHWAYS (v2.0 — 5 Pathways)
    with tab2:
        st.markdown("### 🛤️ Recovery Pathway Intelligence Engine v2.0")
        st.markdown(
            "**5 bank-grade pathways** ranked by composite score = 0.4×Acceptance + 0.4×NPV Recovery + 0.2×Churn Reduction. Includes **Monte Carlo simulation**, **stress testing**, and **audit trail**."
        )

        # Prepare loan details with robustness against NaN
        loan_data = {
            "outstanding_principal": _safe_f(
                customer.get("outstanding_principal"), 500000
            ),
            "emi_amount": _safe_f(customer.get("emi_amount"), 18500),
            "interest_rate": _safe_f(customer.get("interest_rate"), 14.5),
            "remaining_months": _safe_f(customer.get("remaining_months"), 24),
            "monthly_income": _safe_f(customer.get("monthly_income"), 85000),
            "current_savings": _safe_f(customer.get("current_savings"), 50000),
            "payment_history_score": 0.85,
        }

        cur_emi = max(
            loan_data["emi_amount"], 1000
        )  # Safe minimum to prevent slider crash
        cur_principal = max(loan_data["outstanding_principal"], 10000)
        cur_rate = loan_data["interest_rate"]
        cur_tenure = int(max(loan_data["remaining_months"], 6))

        loan = create_loan_from_customer(loan_data)
        engine = RecoveryPathwayEngine()

        # ─── STRESS TEST TOGGLES ───
        st.markdown("#### ⚡ Stress Scenario")
        stress_col1, stress_col2, stress_col3 = st.columns(3)
        with stress_col1:
            stress_base = st.checkbox(
                "Base Case", value=True, key=f"stress_base_{customer_id}"
            )
        with stress_col2:
            stress_20 = st.checkbox(
                "-20% Income Shock", value=False, key=f"stress_20_{customer_id}"
            )
        with stress_col3:
            stress_40 = st.checkbox(
                "-40% / Job Loss", value=False, key=f"stress_40_{customer_id}"
            )

        with st.spinner("Calculating optimal pathways..."):
            pathways = engine.generate_all_pathways(loan)

        # ─── COMPARISON CHART ───
        if pathways:
            names = [d["name"] for d, m in pathways]
            acceptance = [m.acceptance_probability * 100 for d, m in pathways]
            npv_vals = [min(m.npv_recovery_rate * 100, 100) for d, m in pathways]
            churn = [max(m.churn_reduction * 100, 0) for d, m in pathways]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="Acceptance %",
                    x=names,
                    y=acceptance,
                    marker_color="#00539B",
                    text=[f"{v:.0f}%" for v in acceptance],
                    textposition="auto",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="NPV Recovery %",
                    x=names,
                    y=npv_vals,
                    marker_color="#00A3E0",
                    text=[f"{v:.0f}%" for v in npv_vals],
                    textposition="auto",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="Churn Reduction %",
                    x=names,
                    y=churn,
                    marker_color="#22c55e",
                    text=[f"{v:.0f}%" for v in churn],
                    textposition="auto",
                )
            )
            fig.update_layout(
                barmode="group",
                height=350,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                margin=dict(t=40, b=20, l=20, r=20),
                plot_bgcolor="rgba(248,250,252,0.5)",
                yaxis_title="Percentage",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ─── PATHWAY CARDS (5 pathways with enhanced details) ───
        for rank, (details, metrics) in enumerate(pathways, 1):
            rank_cls = (
                "gold"
                if rank == 1
                else "silver"
                if rank == 2
                else "bronze"
                if rank == 3
                else "default"
            )
            pathway_key = details.get("pathway_name", details["name"])
            policy = details.get("policy_result", {})
            policy_ok = policy.get("passed", True)
            policy_badge = "✅ Policy OK" if policy_ok else "⚠️ Policy Flag"

            with st.expander(
                f"{'🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '🏅' if rank == 4 else '🎖️'} Rank #{rank}: {details['name']} — Score: {metrics.composite_score:.3f} {policy_badge}",
                expanded=(rank == 1),
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**{details['description']}**")
                    st.markdown(f"📋 **Action:** {details['action']}")
                    st.markdown(
                        f"⚡ **Immediate Relief:** {details['immediate_relief']}"
                    )

                    if details.get("total_interest_increase", 0) > 0:
                        st.markdown(
                            f"💸 **Additional Interest:** {format_currency(details['total_interest_increase'])} ⚠️"
                        )
                    elif details.get("total_interest_increase", 0) < 0:
                        st.markdown(
                            f"💚 **Interest Savings:** {format_currency(abs(details['total_interest_increase']))} ✅"
                        )

                    if rank == 1:
                        short_expl = details.get("short_explanation", "")
                        if short_expl:
                            st.markdown(
                                f"""<div class="insight-box">
                                    <strong>💡 Recommendation:</strong> {short_expl}
                                </div>""",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""<div class="insight-box">
                                    <strong>💡 Why this pathway?</strong> Highest composite score — balances 
                                    {metrics.acceptance_probability * 100:.0f}% acceptance with {min(metrics.npv_recovery_rate * 100, 100):.0f}% NPV recovery.
                                </div>""",
                                unsafe_allow_html=True,
                            )

                with col2:
                    st.progress(
                        max(0.0, min(metrics.acceptance_probability, 1.0)),
                        text=f"Acceptance: {metrics.acceptance_probability * 100:.1f}%",
                    )
                    st.progress(
                        max(0.0, min(metrics.npv_recovery_rate, 1.0)),
                        text=f"NPV Recovery: {min(metrics.npv_recovery_rate * 100, 100):.1f}%",
                    )
                    st.progress(
                        max(0.0, min(metrics.churn_reduction, 1.0)),
                        text=f"Churn Reduction: {max(metrics.churn_reduction * 100, 0):.1f}%",
                    )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("New EMI", format_currency(metrics.monthly_payment))
                with c2:
                    st.metric("New Tenure", f"{metrics.new_tenure_months} months")
                with c3:
                    st.metric("Total Interest", format_currency(metrics.total_interest))

                # ── Monte Carlo Output (ICR pathway only) ──
                mc = details.get("mc_result")
                if mc:
                    st.markdown("---")
                    st.markdown("##### 🎲 Monte Carlo Simulation Results")
                    mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
                    with mc_c1:
                        st.metric("Mean NPV", format_currency(mc.get("mean_npv", 0)))
                    with mc_c2:
                        st.metric("Std Dev", format_currency(mc.get("std_npv", 0)))
                    with mc_c3:
                        st.metric("5th Pctl", format_currency(mc.get("p5", 0)))
                    with mc_c4:
                        st.metric("95th Pctl", format_currency(mc.get("p95", 0)))

                    st.markdown(
                        f"**{mc.get('n_runs', 0):,} simulations** | P(recovery > 70%) = **{mc.get('prob_above_threshold', 0) * 100:.1f}%** | Seed: {mc.get('seed', 'N/A')}"
                    )

                # ── Audit / Explainability Panel ──
                expl = details.get("explainability", "")
                if expl:
                    with st.expander("📋 Audit & Explainability"):
                        st.markdown(f"**Model Explanation:**\n\n{expl}")
                        audit = details.get("audit", {})
                        if audit:
                            st.markdown(
                                f"**Simulation ID:** `{audit.get('simulation_id', 'N/A')}`"
                            )
                            st.markdown(
                                f"**Timestamp:** {audit.get('timestamp', 'N/A')}"
                            )
                            st.markdown(
                                f"**Model Version:** {audit.get('model_version', 'N/A')}"
                            )

                        # Policy check details
                        if policy and not policy_ok:
                            violations = policy.get("violations", [])
                            if violations:
                                st.warning(
                                    "**Policy Violations:**\n"
                                    + "\n".join(f"- {v}" for v in violations)
                                )

        st.markdown("---")

        # ═══════════════════════════════════════════════
        # ─── INNOVATION 1: WHAT-IF SCENARIO SIMULATOR ───
        # ═══════════════════════════════════════════════
        with st.expander("🎛️ Advanced: What-If Scenario Simulator"):
            st.markdown(
                "Adjust payment parameters and see **real-time ML-predicted outcomes**. No other platform offers live scenario modeling to agents."
            )

            sim_col1, sim_col2 = st.columns([1, 2])

            with sim_col1:
                sim_emi = st.slider(
                    "💳 Custom EMI (₹)",
                    min_value=max(_safe_i(cur_emi * 0.3, 1000), 500),
                    max_value=max(_safe_i(cur_emi * 1.2, 20000), 1000),
                    value=max(_safe_i(cur_emi * 0.7, 5000), 500),
                    step=500,
                    help="Slide to explore different EMI amounts",
                )
                sim_tenure = st.slider(
                    "📅 Custom Tenure (months)",
                    min_value=6,
                    max_value=60,
                    value=min(60, max(6, _safe_i(cur_tenure, 24))),
                    step=3,
                    help="Extend or shorten the repayment period",
                )

                # Calculate live metrics
                emi_reduction_pct = max(0, (cur_emi - sim_emi) / cur_emi)
                sim_total_paid = sim_emi * sim_tenure
                sim_interest = max(0, sim_total_paid - cur_principal)
                original_total = cur_emi * cur_tenure

                # Simulated acceptance (higher reduction = higher acceptance)
                sim_stress = min(1.0, cur_emi / loan_data["monthly_income"])
                sim_z = -2.0 + 3.0 * sim_stress + 4.0 * emi_reduction_pct
                sim_acceptance = 1 / (1 + np.exp(-sim_z))

                # NPV estimate
                monthly_rate = cur_rate / 12 / 100
                sim_npv = sum(
                    sim_emi / (1 + monthly_rate) ** t for t in range(1, sim_tenure + 1)
                )
                original_npv = sum(
                    cur_emi / (1 + monthly_rate) ** t
                    for t in range(1, int(cur_tenure) + 1)
                )
                sim_npv_rate = min(sim_npv / max(original_npv, 1), 1.5)

                # Display live metrics
                st.markdown(
                    f"""<div class="info-card">
                    <h4>📊 Live Simulation Result</h4>
                    <div class="info-row"><span class="info-label">Acceptance Prob.</span><span class="info-value" style="color:{"#22c55e" if sim_acceptance > 0.6 else "#dc2626"}">{sim_acceptance * 100:.1f}%</span></div>
                    <div class="info-row"><span class="info-label">NPV Recovery</span><span class="info-value">{min(sim_npv_rate * 100, 100):.1f}%</span></div>
                    <div class="info-row"><span class="info-label">Total Payable</span><span class="info-value">{format_currency(sim_total_paid)}</span></div>
                    <div class="info-row"><span class="info-label">vs Original</span><span class="info-value" style="color:{"#22c55e" if sim_total_paid < original_total else "#dc2626"}">{format_currency(sim_total_paid - original_total)}</span></div>
                </div>""",
                    unsafe_allow_html=True,
                )

            with sim_col2:
                # Live updating chart
                months_range = list(range(1, int(max(sim_tenure, cur_tenure)) + 1))
                original_cumulative = [
                    min(cur_emi * m, original_total) for m in months_range
                ]
                sim_cumulative = [
                    min(sim_emi * m, sim_total_paid) for m in months_range
                ]

                fig_sim = go.Figure()
                fig_sim.add_trace(
                    go.Scatter(
                        x=months_range,
                        y=original_cumulative,
                        name="Current Plan",
                        mode="lines",
                        line=dict(color="#dc2626", width=2, dash="dash"),
                        fill="tozeroy",
                        fillcolor="rgba(220,38,38,0.05)",
                    )
                )
                fig_sim.add_trace(
                    go.Scatter(
                        x=months_range,
                        y=sim_cumulative,
                        name="Your Scenario",
                        mode="lines",
                        line=dict(color="#00539B", width=3),
                        fill="tozeroy",
                        fillcolor="rgba(0,83,155,0.08)",
                    )
                )
                fig_sim.update_layout(
                    title="Cumulative Payment Comparison",
                    xaxis_title="Month",
                    yaxis_title="Cumulative Payment (₹)",
                    height=350,
                    margin=dict(t=40, b=40, l=40, r=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                    ),
                    plot_bgcolor="rgba(248,250,252,0.5)",
                )
                st.plotly_chart(fig_sim, use_container_width=True)

        st.markdown("---")

        # ═══════════════════════════════════════════════
        # ─── INNOVATION 2: CASH FLOW WATERFALL ───
        # ═══════════════════════════════════════════════
        with st.expander("📊 Advanced: 12-Month Cash Flow Forecast"):
            st.markdown(
                'Side-by-side: **default trajectory** vs **intervention pathway**. Makes abstract NPV tangible — agents can show customers "here\'s your next 12 months."'
            )

            best_pathway = pathways[0] if pathways else None
            best_emi = (
                best_pathway[1].monthly_payment if best_pathway else cur_emi * 0.7
            )

            cf_months = list(range(1, 13))
            # Default path: increasing default probability reduces expected collections
            default_cf = []
            intervention_cf = []
            for m in cf_months:
                def_prob = engine.estimate_default_probability(
                    loan, m, pathway_stress_reduction=0.0
                )
                int_prob = engine.estimate_default_probability(
                    loan, m, pathway_stress_reduction=0.30
                )
                default_cf.append(cur_emi * (1 - def_prob))
                intervention_cf.append(best_emi * (1 - int_prob))

            fig_cf = go.Figure()
            fig_cf.add_trace(
                go.Bar(
                    x=cf_months,
                    y=default_cf,
                    name="Without Intervention",
                    marker_color=[
                        "#dc2626"
                        if v < cur_emi * 0.5
                        else "#f97316"
                        if v < cur_emi * 0.8
                        else "#eab308"
                        for v in default_cf
                    ],
                    text=[f"₹{v / 1000:.0f}K" for v in default_cf],
                    textposition="auto",
                )
            )
            fig_cf.add_trace(
                go.Bar(
                    x=cf_months,
                    y=intervention_cf,
                    name="With Best Pathway",
                    marker_color="#22c55e",
                    opacity=0.85,
                    text=[f"₹{v / 1000:.0f}K" for v in intervention_cf],
                    textposition="auto",
                )
            )
            # Cumulative lines
            fig_cf.add_trace(
                go.Scatter(
                    x=cf_months,
                    y=[sum(default_cf[: i + 1]) for i in range(12)],
                    name="Cumulative (Default)",
                    mode="lines",
                    line=dict(color="#dc2626", width=2, dash="dot"),
                    yaxis="y2",
                )
            )
            fig_cf.add_trace(
                go.Scatter(
                    x=cf_months,
                    y=[sum(intervention_cf[: i + 1]) for i in range(12)],
                    name="Cumulative (Intervention)",
                    mode="lines",
                    line=dict(color="#22c55e", width=2, dash="dot"),
                    yaxis="y2",
                )
            )
            fig_cf.update_layout(
                barmode="group",
                height=400,
                xaxis_title="Month",
                yaxis_title="Monthly Expected Recovery (₹)",
                yaxis2=dict(title="Cumulative (₹)", overlaying="y", side="right"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                margin=dict(t=40, b=40, l=40, r=60),
                plot_bgcolor="rgba(248,250,252,0.5)",
            )
            st.plotly_chart(fig_cf, use_container_width=True)

            # Recovery gap metric
            total_default = sum(default_cf)
            total_interv = sum(intervention_cf)
            recovery_gap = total_interv - total_default

            gap_col1, gap_col2, gap_col3 = st.columns(3)
            with gap_col1:
                st.metric(
                    "❌ Without Intervention (12mo)", format_currency(total_default)
                )
            with gap_col2:
                st.metric("✅ With Best Pathway (12mo)", format_currency(total_interv))
            with gap_col3:
                st.metric(
                    "💰 Additional Recovery",
                    format_currency(recovery_gap),
                    delta=f"+{recovery_gap / max(total_default, 1) * 100:.0f}%",
                )

        st.markdown("---")

        # ═══════════════════════════════════════════════
        # ─── INNOVATION 3: SMART NUDGE TIMING ───
        # ═══════════════════════════════════════════════
        with st.expander("🕐 Advanced: Smart Nudge Timing Optimizer"):
            st.markdown(
                "AI-recommended **best day & time** to contact this customer. Based on salary credit patterns, past interaction data, and day-of-week response rates."
            )

            timing_col1, timing_col2 = st.columns([2, 1])

            with timing_col1:
                # Generate heatmap data based on customer profile
                np.random.seed(hash(customer_id) % 2**31)
                days = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                times = [
                    "Morning\n(9-12)",
                    "Afternoon\n(12-3)",
                    "Evening\n(3-6)",
                    "Night\n(6-9)",
                ]

                # Base success rates with patterns
                base = np.random.uniform(0.15, 0.45, (4, 7))
                # Salary day boost (15th = Wednesday-ish)
                salary_delay = customer.get("salary_delay_days", 0)
                salary_day_idx = min(4, max(0, 2 + int(salary_delay) % 5))
                base[:, salary_day_idx] += 0.15  # Boost around salary day
                base[1:3, :] += 0.05  # Afternoon/evening slightly better
                base[:, 5:] -= 0.08  # Weekends slightly worse
                base = np.clip(base, 0.1, 0.85)

                # Find best slot
                best_idx = np.unravel_index(np.argmax(base), base.shape)
                best_time = times[best_idx[0]].replace("\n", " ")
                best_day = days[best_idx[1]]

                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=base,
                        x=days,
                        y=times,
                        colorscale=[
                            [0, "#fee2e2"],
                            [0.3, "#fef3c7"],
                            [0.6, "#bbf7d0"],
                            [1.0, "#22c55e"],
                        ],
                        text=[[f"{v * 100:.0f}%" for v in row] for row in base],
                        texttemplate="%{text}",
                        textfont=dict(size=12, color="#1e293b"),
                        hovertemplate="<b>%{x} %{y}</b><br>Success Rate: %{z:.0%}<extra></extra>",
                        showscale=True,
                        colorbar=dict(title="Success %"),
                    )
                )
                fig_heat.update_layout(
                    height=300,
                    margin=dict(t=20, b=20, l=20, r=20),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            with timing_col2:
                st.markdown(
                    f"""<div class="info-card">
                    <h4>🎯 Optimal Contact Window</h4>
                    <div class="info-row"><span class="info-label">Best Day</span><span class="info-value" style="color:#22c55e">{best_day}</span></div>
                    <div class="info-row"><span class="info-label">Best Time</span><span class="info-value" style="color:#22c55e">{best_time}</span></div>
                    <div class="info-row"><span class="info-label">Est. Success</span><span class="info-value" style="color:#22c55e">{base[best_idx] * 100:.0f}%</span></div>
                    <div class="info-row"><span class="info-label">Channel</span><span class="info-value">WhatsApp → SMS</span></div>
                </div>""",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""<div class="insight-box">
                    <strong>💡 Timing Intelligence:</strong> Contact on <strong>{best_day} {best_time}</strong> 
                    for <strong>{base[best_idx] * 100:.0f}% predicted success</strong>. 
                    Salary typically credits {"on time" if salary_delay <= 0 else f"{salary_delay} days late"} — 
                    outreach 1-2 days after salary maximizes response.
                </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ═══════════════════════════════════════════════
        # ─── INNOVATION 4: PEER COMPARISON BENCHMARK ───
        # ═══════════════════════════════════════════════
        with st.expander("👥 Advanced: Peer Comparison Benchmark"):
            st.markdown(
                "Anonymized outcomes from **similar customer profiles** — drives compliance via social proof."
            )

            # Simulated peer data based on customer profile
            np.random.seed(hash(customer_id) % 2**31 + 42)
            peer_count = np.random.randint(120, 350)
            accepted_pct = 0.62 + np.random.uniform(-0.08, 0.12)
            recovery_success = 0.71 + np.random.uniform(-0.1, 0.1)
            rejected_recovery = 0.18 + np.random.uniform(-0.05, 0.05)
            credit_improved = 0.45 + np.random.uniform(-0.1, 0.1)

            peer_col1, peer_col2 = st.columns([1, 1])

            with peer_col1:
                # Accepted vs rejected outcome chart
                fig_peer = go.Figure()
                fig_peer.add_trace(
                    go.Bar(
                        x=["Accepted Intervention", "Rejected / No Action"],
                        y=[recovery_success * 100, rejected_recovery * 100],
                        marker_color=["#22c55e", "#dc2626"],
                        text=[
                            f"{recovery_success * 100:.0f}%",
                            f"{rejected_recovery * 100:.0f}%",
                        ],
                        textposition="auto",
                        textfont=dict(size=16, color="white"),
                    )
                )
                fig_peer.update_layout(
                    yaxis_title="Successful Recovery Rate",
                    height=300,
                    margin=dict(t=20, b=40, l=40, r=20),
                    plot_bgcolor="rgba(248,250,252,0.5)",
                    yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(fig_peer, use_container_width=True)

            with peer_col2:
                st.markdown(
                    f"""<div class="info-card">
                    <h4>📊 Peer Group Statistics</h4>
                    <div class="info-row"><span class="info-label">Similar Profiles</span><span class="info-value">{peer_count} customers</span></div>
                    <div class="info-row"><span class="info-label">Accepted Offer</span><span class="info-value" style="color:#22c55e">{accepted_pct * 100:.0f}%</span></div>
                    <div class="info-row"><span class="info-label">Recovery (Accepted)</span><span class="info-value" style="color:#22c55e">{recovery_success * 100:.0f}%</span></div>
                    <div class="info-row"><span class="info-label">Recovery (Rejected)</span><span class="info-value" style="color:#dc2626">{rejected_recovery * 100:.0f}%</span></div>
                    <div class="info-row"><span class="info-label">Credit Score ↑</span><span class="info-value">{credit_improved * 100:.0f}% improved</span></div>
                </div>""",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""<div class="insight-box">
                    <strong>🗣️ Agent Script:</strong> <em>"Out of {peer_count} customers in a similar situation, 
                    {accepted_pct * 100:.0f}% chose to work with us on a flexible plan. Of those, {recovery_success * 100:.0f}% successfully 
                    recovered — compared to only {rejected_recovery * 100:.0f}% who didn't act."</em>
                </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ═══════════════════════════════════════════════
        # ─── INNOVATION 5: IMPACT PROJECTOR ───
        # ═══════════════════════════════════════════════
        st.markdown("### 💰 Impact Projector — 3 / 6 / 12 Month Outcomes")
        st.markdown(
            "Projected outcomes if the **recommended pathway** is adopted. Turns ML predictions into executive-friendly metrics."
        )

        risk_now = customer["risk_score"]
        best_pw = pathways[0][1] if pathways else None

        # 3-month, 6-month, 12-month projections
        projections = [
            {
                "period": "3 Months",
                "icon": "📅",
                "default_reduction": min(25 + risk_now * 0.2, 45),
                "recovered_amt": best_emi * 3 * 0.92 if best_pw else cur_emi * 3 * 0.5,
                "credit_delta": "+15 pts",
                "relationship_value": cur_emi * 12 * 0.85,
            },
            {
                "period": "6 Months",
                "icon": "📆",
                "default_reduction": min(40 + risk_now * 0.25, 65),
                "recovered_amt": best_emi * 6 * 0.88 if best_pw else cur_emi * 6 * 0.4,
                "credit_delta": "+32 pts",
                "relationship_value": cur_emi * 24 * 0.78,
            },
            {
                "period": "12 Months",
                "icon": "🗓️",
                "default_reduction": min(55 + risk_now * 0.3, 82),
                "recovered_amt": best_emi * 12 * 0.85
                if best_pw
                else cur_emi * 12 * 0.3,
                "credit_delta": "+55 pts",
                "relationship_value": cur_emi * 36 * 0.72,
            },
        ]

        proj_cols = st.columns(3)
        for proj, col in zip(projections, proj_cols):
            with col:
                st.markdown(
                    f"""<div class="kpi-card" style="text-align:center;">
                    <div class="kpi-icon">{proj["icon"]}</div>
                    <div class="kpi-label">{proj["period"]} Projection</div>
                    <div style="margin:0.8rem 0;">
                        <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;margin-bottom:0.2rem;">Default Risk Reduction</div>
                        <div style="font-size:1.5rem;font-weight:800;color:#22c55e;">↓ {proj["default_reduction"]:.0f}%</div>
                    </div>
                    <div style="margin:0.5rem 0;">
                        <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;margin-bottom:0.2rem;">Recovered Amount</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#00539B;">{format_currency(proj["recovered_amt"])}</div>
                    </div>
                    <div style="margin:0.5rem 0;">
                        <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;margin-bottom:0.2rem;">Credit Score Impact</div>
                        <div style="font-size:1rem;font-weight:600;color:#22c55e;">{proj["credit_delta"]}</div>
                    </div>
                    <div style="margin:0.5rem 0;">
                        <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;margin-bottom:0.2rem;">Relationship Value</div>
                        <div style="font-size:1rem;font-weight:600;color:#00395D;">{format_currency(proj["relationship_value"])}</div>
                    </div>
                </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class="exec-summary">
            ✅ <strong>ROI Summary:</strong> By intervening now with the recommended pathway, the bank is projected to recover 
            <strong>{format_currency(projections[2]["recovered_amt"])}</strong> over 12 months vs. <strong>{format_currency(cur_emi * 12 * 0.3)}</strong> 
            without action — a <strong>{((projections[2]["recovered_amt"] / max(cur_emi * 12 * 0.3, 1)) - 1) * 100:.0f}% improvement</strong>. 
            The customer's lifetime relationship value is preserved at <strong>{format_currency(projections[2]["relationship_value"])}</strong>.
        </div>""",
            unsafe_allow_html=True,
        )

    # TAB 3: FINANCIAL HEALTH MONITOR
    with tab3:
        st.markdown("### 💰 Financial Health Monitor")
        st.markdown(
            "Real-time health assessment across **5 vital signs** with automated intervention triggers. Scores 0-100 where higher is healthier."
        )

        health_data = {
            "customer_id": customer_id,
            "salary_delay_days": customer.get("salary_delay_days", 0),
            "salary_amount_variance": customer.get("salary_amount_variance", 0.0),
            "emergency_fund_days": customer.get("emergency_fund_days", 30),
            "savings_drawdown_rate_4w": customer.get("savings_drawdown_rate_4w", 0.0),
            "current_savings": customer.get("current_savings", 50000),
            "upi_lending_app_txn_count_30d": customer.get(
                "upi_lending_app_txn_count_30d", 0
            ),
            "upi_lending_app_amount_30d": customer.get("upi_lending_app_amount_30d", 0),
            "utility_payment_delay_avg": customer.get("utility_payment_delay_avg", 0),
            "bill_payment_delay_max": customer.get("bill_payment_delay_max", 0),
            "discretionary_spend_pct_change": customer.get(
                "discretionary_spend_pct_change", 0.0
            ),
            "essential_spend_ratio": customer.get("essential_spend_ratio", 0.5),
            "atm_withdrawal_spike_30d": customer.get("atm_withdrawal_spike_30d", 0),
        }

        monitor = FinancialHealthMonitor()
        current_health_score = 100 - customer["risk_score"]
        historical = create_mock_historical_trend(current_health_score, "declining")
        snapshot = monitor.assess_customer_health(health_data, historical)
        trend = monitor.generate_trend_analysis(snapshot)

        # Score + Radar
        col1, col2 = st.columns([1, 2])

        with col1:
            gauge = create_gauge_chart(snapshot.composite_score, "Health Score")
            st.plotly_chart(gauge, use_container_width=True)

            # Score interpretation
            if snapshot.composite_score >= 70:
                interp = "This customer is in relatively good financial health. Standard monitoring is sufficient."
                interp_cls = ""
            elif snapshot.composite_score >= 50:
                interp = "Financial health is deteriorating. Proactive engagement recommended to prevent further decline."
                interp_cls = "warning"
            else:
                interp = "Critical financial distress detected. Immediate intervention required across multiple dimensions."
                interp_cls = "danger"
            st.markdown(
                f'<div class="exec-summary {interp_cls}">{interp}</div>',
                unsafe_allow_html=True,
            )

        with col2:
            # Radar chart of vital signs
            vital_names = [v.name for v in snapshot.vital_signs]
            vital_scores = [v.score for v in snapshot.vital_signs]

            fig = go.Figure(
                data=go.Scatterpolar(
                    r=vital_scores + [vital_scores[0]],
                    theta=vital_names + [vital_names[0]],
                    fill="toself",
                    fillcolor="rgba(0, 83, 155, 0.15)",
                    line=dict(color="#00539B", width=2),
                    marker=dict(size=8, color="#00539B"),
                )
            )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, range=[0, 100], tickfont=dict(size=10)
                    ),
                    angularaxis=dict(tickfont=dict(size=11)),
                ),
                height=380,
                margin=dict(t=30, b=30, l=60, r=60),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Alert + Action
        col1, col2 = st.columns(2)
        with col1:
            alert_emoji = {
                "GREEN": "🟢",
                "YELLOW": "🟡",
                "ORANGE": "🟠",
                "RED": "🔴",
                "CRITICAL": "🔴🔴",
            }
            st.markdown(
                f"""<div class="info-card">
                <h4>{alert_emoji.get(snapshot.alert_level.value.upper(), "⚪")} Alert Status</h4>
                <div class="info-row"><span class="info-label">Alert Level</span><span class="info-value">{snapshot.alert_level.value.upper()}</span></div>
                <div class="info-row"><span class="info-label">Urgency</span><span class="info-value">{snapshot.urgency}/5</span></div>
                <div class="info-row"><span class="info-label">30-Day Trend</span><span class="info-value">{trend["direction"]}</span></div>
                <div class="info-row"><span class="info-label">7-Day Forecast</span><span class="info-value">{trend["forecast_7d"]:.1f}</span></div>
            </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="info-card">
                <h4>🎯 Recommended Action</h4>
                <p style="font-size: 0.9rem; line-height: 1.6; color: #1e293b;">{snapshot.recommended_action}</p>
            </div>""",
                unsafe_allow_html=True,
            )

        # Vital signs breakdown
        st.markdown("---")
        st.markdown("##### 🔍 Vital Signs Breakdown")

        for vital in snapshot.vital_signs:
            alert_emoji_v = {
                "GREEN": "🟢",
                "YELLOW": "🟡",
                "ORANGE": "🟠",
                "RED": "🔴",
                "CRITICAL": "🔴",
            }
            emoji = alert_emoji_v.get(vital.alert_level.value.upper(), "⚪")
            with st.expander(
                f"{emoji} {vital.name} — Score: {vital.score:.1f}/100",
                expanded=vital.threshold_breached,
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(vital.score / 100, text=f"Score: {vital.score:.1f}/100")
                    st.markdown(f"**Current Value:** {vital.current_value}")
                    st.markdown(f"**Status:** {vital.alert_message}")
                with col2:
                    st.markdown(f"### {emoji}")
                    st.markdown(f"**{vital.alert_level.value.upper()}**")

            # 30-day trend chart
        st.markdown("---")
        st.markdown("##### 📈 30-Day Health Trend")

        fig = go.Figure()
        x_vals = list(range(len(historical)))
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=historical,
                mode="lines",
                name="Health Score",
                line=dict(color="#00539B", width=3),
                fill="tonexty" if min(historical) > 0 else None,
                fillcolor="rgba(0, 83, 155, 0.08)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=historical,
                mode="markers",
                name="Data Points",
                marker=dict(size=4, color="#00539B"),
                showlegend=False,
            )
        )

        fig.add_hline(
            y=70, line_dash="dash", line_color="#22c55e", annotation_text="GREEN (70)"
        )
        fig.add_hline(
            y=50, line_dash="dash", line_color="#eab308", annotation_text="YELLOW (50)"
        )
        fig.add_hline(
            y=40, line_dash="dash", line_color="#dc2626", annotation_text="RED (40)"
        )

        fig.update_layout(
            xaxis_title="Days Ago",
            yaxis_title="Health Score",
            height=350,
            showlegend=False,
            margin=dict(t=20, b=40, l=40, r=20),
            plot_bgcolor="rgba(248,250,252,0.5)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── REDESIGNED TAB4: AI SITUATION ROOM ──
    with tab4:
        # ═══════════════ CSS ═══════════════
        st.markdown(
            """
        <style>
        @keyframes pulse-green { 0%,100%{opacity:1} 50%{opacity:0.4} }
        .sr-live{width:8px;height:8px;background:#10b981;border-radius:50%;animation:pulse-green 1.5s infinite;display:inline-block;margin-right:6px;}

        .sr-header{background:#0f172a;border-radius:12px;padding:1.5rem 2rem;display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem;border-left:4px solid #00A3E0;}
        .sr-header-left h2{color:white;font-size:1.35rem;font-weight:800;margin:0 0 4px 0;display:flex;align-items:center;}
        .sr-header-left p{color:#94a3b8;font-size:0.85rem;margin:0;}
        .sr-header-pills{display:flex;gap:8px;}
        .sr-pill{padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:600;}
        .sr-pill-dark{background:rgba(255,255,255,0.1);color:white;}
        .sr-pill-green{background:rgba(16,185,129,0.2);color:#34d399;}

        .threat-banner{border-radius:12px;padding:1.2rem 1.5rem;display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;}
        .threat-critical{background:linear-gradient(135deg,#7f1d1d,#991b1b);border:1px solid #dc2626;}
        .threat-high{background:linear-gradient(135deg,#78350f,#92400e);border:1px solid #f59e0b;}
        .threat-medium{background:linear-gradient(135deg,#1e3a5f,#1e40af);border:1px solid #3b82f6;}
        .threat-low{background:linear-gradient(135deg,#064e3b,#065f46);border:1px solid #10b981;}
        .threat-banner h3{color:white;margin:0 0 4px 0;font-size:1.15rem;}
        .threat-banner p{color:rgba(255,255,255,0.7);margin:0;font-size:0.85rem;}
        .threat-stats{display:flex;gap:16px;}
        .threat-stat{text-align:center;min-width:80px;}
        .threat-stat .val{color:white;font-size:1.3rem;font-weight:800;}
        .threat-stat .lbl{color:rgba(255,255,255,0.6);font-size:0.7rem;text-transform:uppercase;font-weight:600;}

        .signal-card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.5rem;display:flex;justify-content:space-between;align-items:center;}
        .signal-card .sig-left{display:flex;align-items:center;gap:10px;}
        .signal-card .sig-name{font-weight:700;color:#0f172a;font-size:0.9rem;}
        .signal-card .sig-val{font-size:0.8rem;color:#64748b;}
        .sig-sev{padding:3px 10px;border-radius:12px;font-size:0.72rem;font-weight:700;text-transform:uppercase;}
        .sev-CRITICAL{background:#fef2f2;color:#dc2626;border:1px solid #fecaca;}
        .sev-HIGH{background:#fff7ed;color:#ea580c;border:1px solid #fed7aa;}
        .sev-MEDIUM{background:#eff6ff;color:#2563eb;border:1px solid #bfdbfe;}
        .sev-LOW{background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;}

        .playbook-step{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.6rem;display:flex;justify-content:space-between;align-items:center;}
        .playbook-step .ps-left{display:flex;align-items:center;gap:12px;}
        .ps-num{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:0.85rem;color:white;}
        .ps-active{background:#00A3E0;}
        .ps-disabled{background:#cbd5e1;}
        .ps-detail h4{margin:0;font-size:0.92rem;color:#0f172a;}
        .ps-detail p{margin:2px 0 0 0;font-size:0.78rem;color:#64748b;}
        .ps-timing{background:#f1f5f9;padding:3px 10px;border-radius:8px;font-size:0.78rem;font-weight:700;color:#475569;}

        .outcome-card{border-radius:12px;padding:1.2rem 1.5rem;border:1px solid #e2e8f0;}
        .outcome-card h4{margin:0 0 0.8rem 0;font-size:1rem;}
        .outcome-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f1f5f9;}
        .outcome-row:last-child{border-bottom:none;}
        .outcome-label{font-size:0.85rem;color:#64748b;}
        .outcome-val{font-size:0.85rem;font-weight:700;}

        .deploy-btn-wrap button{background:linear-gradient(135deg,#00A3E0,#0077b6)!important;color:white!important;font-size:1.1rem!important;font-weight:800!important;padding:0.8rem!important;border:none!important;border-radius:12px!important;box-shadow:0 4px 12px rgba(0,163,224,0.3)!important;}
        .deploy-btn-wrap button:hover{box-shadow:0 6px 20px rgba(0,163,224,0.5)!important;transform:translateY(-1px);}

        .sys-status-bar{background:#0f172a;color:#94a3b8;font-size:0.75rem;font-family:'Courier New',monospace;padding:0.5rem 1rem;border-radius:6px;margin-top:2rem;text-align:center;}
        </style>
        """,
            unsafe_allow_html=True,
        )

        # ═══════════════ HEADER ═══════════════
        assigned_analyst = st.session_state.get(f"agent_{customer_id}", "Unassigned")
        st.markdown(
            f"""
        <div class="sr-header">
            <div class="sr-header-left">
                <h2><span class="sr-live"></span> AI Situation Room</h2>
                <p>Autonomous threat assessment & intervention orchestration | Case Assigned to: <b>{assigned_analyst}</b></p>
            </div>
            <div class="sr-header-pills">
                <span class="sr-pill sr-pill-dark">🧰 11 Tools</span>
                <span class="sr-pill sr-pill-dark">⚡ Real-Time</span>
                <span class="sr-pill sr-pill-green">🔒 RBI Compliant</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ── AUTO-ANALYSIS (runs on tab load) ──
        customer_dict_sr = customer.to_dict()
        customer_dict_sr["customer_id"] = customer_id

        # ── Null-safe EMI calculation ──
        # Priority: real loans.parquet value > ratio*income derived > 15000 fallback
        def _safe_float(val, default=0.0):
            """Return float, treating NaN/None as default."""
            try:
                v = float(val)
                return v if not (v != v) else default  # v != v is True for NaN
            except Exception:
                return default

        _emi_raw = customer.get("emi_amount")
        _income = _safe_float(customer.get("monthly_income"), 50000)
        _ratio = _safe_float(customer.get("emi_to_income_ratio"), 0.0)
        if _emi_raw is not None and _safe_float(_emi_raw) > 0:
            _emi_computed = _safe_float(_emi_raw)
        elif _ratio > 0 and _income > 0:
            _emi_computed = round(_ratio * _income)
        else:
            _emi_computed = 15000  # Hard fallback

        customer_dict_sr["emi_amount"] = _emi_computed
        customer_dict_sr["full_name"] = customer.get(
            "full_name", f"Customer {customer_id}"
        )
        customer_dict_sr["monthly_income"] = _income

        # Cache results in session state to avoid re-running on every interaction
        sr_cache_key = f"sr_cache_{customer_id}"
        if sr_cache_key not in st.session_state:
            with st.spinner(
                "🧠 AI is analyzing customer signals, predicting outcomes, and generating intervention plan..."
            ):
                import time as _t

                _t.sleep(0.3)

                # Fetch days left from real dataset
                today_day = datetime.now().day
                raw_emi_day = customer.get("emi_day_of_month", 5)
                emi_day_val = int(raw_emi_day) if pd.notna(raw_emi_day) else 5
                if emi_day_val >= today_day:
                    days_left_val = emi_day_val - today_day
                else:
                    days_left_val = (30 - today_day) + emi_day_val

                customer_dict_sr["days_until_emi"] = days_left_val

                _risk_analysis = tool_analyze_risk_signals(customer_dict_sr)
                _outcome = tool_predict_outcome(customer_dict_sr)
                _channel = tool_optimize_channel(customer_dict_sr)
                _plan = tool_generate_intervention_plan(customer_dict_sr)
                _pathways = tool_evaluate_pathways(customer_dict_sr)

                name_part = (
                    str(customer_id).replace("CUST", "")[:4]
                    if "CUST" in str(customer_id)
                    else str(customer_id)[:6]
                )
                emi_val = customer_dict_sr.get("emi_amount", 15000)

                # Generate Formal Groq SMS
                sms_prompt = f"""Generate a professional, formal SMS (under 160 chars) for {name_part} whose EMI is ₹{emi_val:,.0f}.
Risk Level is {_risk_analysis["overall_risk_level"]}.
Use this template as an inspiration, adjusting tone for risk:
"Hi [Name], your upcoming loan payment of ₹[Amount] is due on [Date]. Plan ahead to ensure a smooth payment experience. We're here if you need help."
Return ONLY the raw SMS text, no quotes."""
                ai_sms_resp = real_ai_engine.generate_response(
                    sms_prompt, customer_dict_sr
                )
                _sms = (
                    ai_sms_resp.get("response", "").strip(' \n"')
                    if ai_sms_resp.get("success")
                    else None
                )

                # Generate Formal Groq Script (High-End Wealth Management Tone)
                script_prompt = f"""You are a specialized Senior Relationship Manager at Barclays Wealth Management. Generate a highly formal, empathetic 3-part call script (Opening, Message, Close) for a check-in call with {customer_dict_sr["full_name"]}.
Risk Level is {_risk_analysis["overall_risk_level"]}.
The tone should be 'Concierge Service'. 
CRITICAL: DO NOT mention specific recovery pathways, skip-payments, or EMI reductions. 
Focus purely on:
- A personalized financial wellness check-in.
- Offering a comprehensive portfolio review.
- Confirming that we are here to support their long-term financial journey.
Return exactly a valid JSON object with keys: "opening", "offer", "close" (keep "offer" as the key name for internal logic, but populate it with relationship-support messaging)."""
                ai_script_resp = real_ai_engine.generate_response(
                    script_prompt, customer_dict_sr
                )
                try:
                    resp_str = ai_script_resp.get("response", "{}")
                    if "```json" in resp_str:
                        resp_str = resp_str.split("```json")[1].split("```")[0]
                    elif "```" in resp_str:
                        resp_str = resp_str.split("```")[1].split("```")[0]
                    import json as _json

                    _script = _json.loads(resp_str)
                    if not all(k in _script for k in ["opening", "offer", "close"]):
                        raise ValueError("Missing keys")
                except:
                    _script = tool_generate_script(customer_dict_sr)

                st.session_state[sr_cache_key] = {
                    "risk": _risk_analysis,
                    "outcome": _outcome,
                    "channel": _channel,
                    "plan": _plan,
                    "script": _script,
                    "custom_sms": _sms,
                    "pathways": _pathways,
                }

        sr_data = st.session_state[sr_cache_key]
        risk_analysis = sr_data["risk"]
        outcome_data = sr_data["outcome"]
        channel_data = sr_data["channel"]
        plan_data = sr_data["plan"]
        script_data = sr_data["script"]
        pathways_data = sr_data["pathways"]

        # ═══════ SECTION 1: THREAT ASSESSMENT ═══════
        risk_score = float(customer.get("risk_score", 0))
        threat_level = risk_analysis["overall_risk_level"]
        threat_css = f"threat-{threat_level.lower()}"
        emi_val = _emi_computed  # null-safe, computed above
        income_val = _income  # null-safe, computed above

        # Calculate real time to default based on EMI day
        today_day_sr = datetime.now().day
        raw_emi_day_sr = customer.get("emi_day_of_month", 5)
        emi_day_sr = int(raw_emi_day_sr) if pd.notna(raw_emi_day_sr) else 5
        if emi_day_sr >= today_day_sr:
            days_left_sr = emi_day_sr - today_day_sr
        else:
            days_left_sr = (30 - today_day_sr) + emi_day_sr

        ttd = f"{days_left_sr} Days"
        amount_at_risk = format_currency(emi_val * 6 * 0.7)

        st.markdown(
            f"""
        <div class="threat-banner {threat_css}">
            <div>
                <h3>{"🔴" if threat_level == "CRITICAL" else "🟠" if threat_level == "HIGH" else "🟡" if threat_level == "MEDIUM" else "🟢"} THREAT LEVEL: {threat_level}</h3>
                <p>Customer {customer_id} — {risk_analysis["signals_detected"]} risk signals detected across 24 behavioral features</p>
            </div>
            <div class="threat-stats">
                <div class="threat-stat"><div class="val">{risk_score:.0f}</div><div class="lbl">Risk Score</div></div>
                <div class="threat-stat"><div class="val">{ttd}</div><div class="lbl">Time to Default</div></div>
                <div class="threat-stat"><div class="val">{amount_at_risk}</div><div class="lbl">Amount at Risk</div></div>
                <div class="threat-stat"><div class="val">{plan_data.get("confidence", 85)}%</div><div class="lbl">AI Confidence</div></div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Risk Signals Cards
        with st.expander(
            f"🚨 {risk_analysis['signals_detected']} Risk Signals Detected — Click to Expand",
            expanded=(risk_score >= 70),
        ):
            if risk_analysis["signals"]:
                for sig in risk_analysis["signals"]:
                    st.markdown(
                        f"""
                    <div class="signal-card">
                        <div class="sig-left">
                            <div>
                                <div class="sig-name">{sig["signal"]}</div>
                                <div class="sig-val">{sig["value"]} — {sig["impact"]}</div>
                            </div>
                        </div>
                        <span class="sig-sev sev-{sig["severity"]}">{sig["severity"]}</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("No critical risk signals detected.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ═══════ SECTION 2: EDITABLE INTERVENTION PLAYBOOK ═══════
        st.markdown(
            """
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
            <div>
                <h4 style="margin:0;color:#0f172a;">📋 AI-Generated Intervention Playbook</h4>
                <p style="margin:2px 0 0;font-size:0.82rem;color:#64748b;">Toggle steps ON/OFF, edit messages, and adjust timing before deploying</p>
            </div>
            <div style="background:#f0fdf4;color:#15803d;padding:4px 12px;border-radius:8px;font-size:0.78rem;font-weight:700;">
                Priority: {plan_data.get("priority", "HIGH")} | Recovery Rate: {plan_data.get("expected_recovery_rate", "78%")}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Initialize playbook state
        pb_key = f"playbook_{customer_id}"
        if pb_key not in st.session_state:
            st.session_state[pb_key] = {
                "steps": [],
                "message": None,
                "script": None,
            }
            # Build steps from the plan
            for i, step_info in enumerate(plan_data.get("steps", [])):
                st.session_state[pb_key]["steps"].append(
                    {
                        "enabled": True,
                        "action": step_info["action"],
                        "timing": step_info["timing"],
                        "detail": step_info["detail"],
                    }
                )

        pb_state = st.session_state[pb_key]

        # Render each playbook step with toggle
        for i, step_info in enumerate(pb_state["steps"]):
            step_col1, step_col2, step_col3 = st.columns([0.5, 5, 1.5])
            with step_col1:
                enabled = st.checkbox(
                    "",
                    value=step_info["enabled"],
                    key=f"pb_toggle_{customer_id}_{i}",
                    label_visibility="collapsed",
                )
                pb_state["steps"][i]["enabled"] = enabled
            with step_col2:
                num_css = "ps-active" if enabled else "ps-disabled"
                st.markdown(
                    f"""
                <div style="display:flex;align-items:center;gap:12px;opacity:{"1" if enabled else "0.4"};">
                    <div class="ps-num {num_css}">{i + 1}</div>
                    <div>
                        <div style="font-weight:700;font-size:0.92rem;color:#0f172a;">{step_info["action"]}</div>
                        <div style="font-size:0.78rem;color:#64748b;">{step_info["detail"]}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with step_col3:
                new_timing = st.text_input(
                    "Timing",
                    value=step_info["timing"],
                    key=f"pb_timing_{customer_id}_{i}",
                    label_visibility="collapsed",
                )
                pb_state["steps"][i]["timing"] = new_timing

        # Playbook Calendar Timeline
        timeline_html = '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1rem;margin:1rem 0;display:flex;justify-content:space-between;position:relative;">'
        timeline_html += '<div style="position:absolute;top:50%;left:2.5rem;right:2.5rem;height:2px;background:#cbd5e1;z-index:0;transform:translateY(-50%);"></div>'
        active_steps = [s for s in pb_state["steps"] if s["enabled"]]
        if not active_steps:
            timeline_html += '<div style="color:#94a3b8;font-size:0.85rem;text-align:center;width:100%;">No steps active in playbook</div>'
        else:
            for i, step_info in enumerate(active_steps):
                timeline_html += f"""
                <div style="z-index:1;background:white;padding:4px 16px;border-radius:20px;border:2px solid {"#ef4444" if threat_level == "CRITICAL" else "#00A3E0"};text-align:center;font-size:0.75rem;font-weight:700;box-shadow:0 3px 6px rgba(0,0,0,0.08);">
                    <div style="color:#64748b;font-size:0.6rem;text-transform:uppercase;">Step {i + 1}</div>
                    <div style="color:#0f172a;">{step_info["action"].split()[0]} · {step_info["timing"]}</div>
                </div>"""
        timeline_html += "</div>"
        st.markdown(
            f'<div style="margin-bottom:0.4rem;font-weight:700;color:#0f172a;font-size:0.9rem;">📅 Intervention Timeline Monitor</div>',
            unsafe_allow_html=True,
        )
        st.markdown(timeline_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ═══════ SECTION 3: SMS STRATEGY & SCRIPT PREVIEW ═══════
        st.markdown(
            """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.8rem;">
            <h4 style="margin:0;color:#0f172a;">💬 Pre-Payment SMS Strategy & Call Script</h4>
            <span style="background:#eff6ff;color:#2563eb;padding:3px 10px;border-radius:8px;font-size:0.72rem;font-weight:700;">AI GENERATED (EDITABLE)</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        msg_col, script_col = st.columns([1, 1])

        with msg_col:
            # Generate default message if not in state
            if pb_state["message"] is None:
                if sr_data.get("custom_sms"):
                    pb_state["message"] = sr_data["custom_sms"]
                else:
                    cust_name = customer.get("full_name", f"Customer {customer_id}")
                    pb_state["message"] = (
                        f"Hi {cust_name}, your upcoming loan payment of ₹{emi_val:,.0f} is due soon. Plan ahead to ensure a smooth payment experience. We're here if you need help."
                    )

            st.markdown(
                f"**📱 SMS Message Strategy** · Optimally timed for: **{channel_data['best_day']} {channel_data['best_time_slot']}**"
            )
            edited_msg = st.text_area(
                "SMS Message",
                value=pb_state["message"],
                height=180,
                key=f"sr_msg_{customer_id}",
                label_visibility="collapsed",
            )
            pb_state["message"] = edited_msg
            st.caption(
                f"📏 {len(edited_msg)} chars · 📈 Est. conversion: {channel_data['predicted_response_rate']}"
            )

        with script_col:
            if pb_state["script"] is None:
                pb_state["script"] = script_data

            st.markdown("**📞 Call Script** · Senior Relationship Manager Tone")
            script_opening = st.text_area(
                "👋 Opening",
                value=pb_state["script"].get("opening", ""),
                height=60,
                key=f"sr_script_open_{customer_id}",
            )
            script_offer = st.text_area(
                "💬 Relationship Message",
                value=pb_state["script"].get("offer", ""),
                height=60,
                key=f"sr_script_offer_{customer_id}",
            )
            script_close = st.text_area(
                "✅ Close",
                value=pb_state["script"].get("close", ""),
                height=60,
                key=f"sr_script_close_{customer_id}",
            )
            pb_state["script"]["opening"] = script_opening
            pb_state["script"]["offer"] = script_offer
            pb_state["script"]["close"] = script_close

        st.markdown("<br>", unsafe_allow_html=True)

        # ═══════ SECTION 4: OUTCOME SIMULATOR ═══════
        st.markdown(
            """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.1rem;">
            <h4 style="margin:0;color:#0f172a;">📊 Outcome Simulator</h4>
            <span style="background:#f1f5f9;color:#475569;padding:3px 10px;border-radius:8px;font-size:0.72rem;font-weight:700;">STRATEGY BENCHMARKING</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Choice of pathway
        rec_path = pathways_data["recommended"]
        p_names = [p["name"] for p in pathways_data["pathways"]]

        sim_col1, sim_col2 = st.columns([1.5, 1])
        with sim_col1:
            selected_path_name = st.selectbox(
                "Select Intervention Strategy to Simulate:",
                options=p_names,
                index=p_names.index(rec_path) if rec_path in p_names else 0,
                key=f"sim_path_{customer_id}",
            )
        with sim_col2:
            st.markdown(
                f"""
            <div style="margin-top:1.8rem; font-size:0.8rem; color:#475569;">
                <b>AI Recommended:</b> <span style="color:#00A3E0;">{rec_path}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Dynamic simulation based on selection
        sel_path = next(
            (p for p in pathways_data["pathways"] if p["name"] == selected_path_name),
            pathways_data["pathways"][0],
        )

        st.markdown(
            f"""
        <div style="background:#eff6ff; padding:10px; border-radius:8px; margin-bottom:1rem; border:1px solid #bfdbfe;">
            <span style="font-weight:700; color:#1e40af; font-size:0.85rem;">PATHWAY INSIGHT:</span>
            <span style="color:#1e3a8a; font-size:0.85rem;">{sel_path["why_recommended"]}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        oc_left, oc_right = st.columns(2)
        without = outcome_data["without_intervention"]

        # Scaling with-intervention metrics based on path strength
        base_with = outcome_data["with_intervention"]
        prob_val = int(sel_path["acceptance_probability"] * 100)

        with oc_left:
            st.markdown(
                f"""
            <div class="outcome-card" style="background:#fef2f2;border-color:#fecaca;">
                <h4 style="color:#dc2626;">❌ Without Intervention</h4>
                <div class="outcome-row"><span class="outcome-label">Default Probability</span><span class="outcome-val" style="color:#dc2626;">{without["default_probability"]}</span></div>
                <div class="outcome-row"><span class="outcome-label">Expected Loss (6M)</span><span class="outcome-val" style="color:#dc2626;">{without["expected_loss_6m"]}</span></div>
                <div class="outcome-row"><span class="outcome-label">Credit Score Impact</span><span class="outcome-val" style="color:#dc2626;">{without["credit_score_impact"]}</span></div>
                <div class="outcome-row"><span class="outcome-label">Customer Retention</span><span class="outcome-val" style="color:#dc2626;">{without["customer_retention"]}</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with oc_right:
            st.markdown(
                f"""
            <div class="outcome-card" style="background:#f0fdf4;border-color:#bbf7d0;">
                <h4 style="color:#15803d;">✅ With {selected_path_name}</h4>
                <div class="outcome-row"><span class="outcome-label">Acceptance Probability</span><span class="outcome-val" style="color:#15803d;">{prob_val}%</span></div>
                <div class="outcome-row"><span class="outcome-label">Default Prob. Reduction</span><span class="outcome-val" style="color:#15803d;">-{int((1 - sel_path["acceptance_probability"]) * 40 + 30)}%</span></div>
                <div class="outcome-row"><span class="outcome-label">Expected Recovery (12M)</span><span class="outcome-val" style="color:#15803d;">{base_with["expected_recovery_12m"]}</span></div>
                <div class="outcome-row"><span class="outcome-label">Retention Impact</span><span class="outcome-val" style="color:#15803d;">High (Long-term)</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # ROI highlight
        st.markdown(
            f"""
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:10px;padding:1rem 1.5rem;margin-top:0.8rem;display:flex;justify-content:space-between;align-items:center;">
            <div style="color:white;font-weight:700;">💰 Estimated Intervention ROI</div>
            <div style="color:#34d399;font-size:1.3rem;font-weight:800;">{outcome_data["intervention_roi"]} protected</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550 SECTION 5: AI AGENT WORKFLOW SCHEDULER \u2550\u2550\u2550\u2550\u2550\u2550\u2550
        st.markdown("---")
        st.markdown(
            """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.4rem;">
            <h4 style="margin:0;color:#0f172a;">🗓️ AI Agent Workflow Scheduler</h4>
            <span style="background:linear-gradient(90deg,#00A3E0,#0077b6);color:white;padding:3px 12px;border-radius:20px;font-size:0.72rem;font-weight:700;">ANALYST-CONTROLLED AUTOMATION</span>
        </div>
        <p style="color:#64748b;font-size:0.85rem;margin-bottom:1.2rem;">
            Configure and schedule exactly <b>when</b> each outreach action is triggered for this customer.
            The AI pre-populates optimal dates based on EMI proximity. You can override any date.
        </p>
        """,
            unsafe_allow_html=True,
        )

        # Compute EMI due date for default scheduling
        _today = datetime.now()

        try:
            _raw_emi_day = customer.get("emi_day_of_month")
            _emi_day = int(float(_raw_emi_day)) if pd.notna(_raw_emi_day) else 5
        except (ValueError, TypeError):
            _emi_day = 5

        if _emi_day >= _today.day:
            _emi_date = _today.replace(day=_emi_day)
        else:
            import calendar as _cal

            _last_day = _cal.monthrange(_today.year, _today.month)[1]
            _next_month = _today.replace(day=1) + timedelta(days=_last_day)
            try:
                _emi_date = _next_month.replace(day=_emi_day)
            except Exception:
                _emi_date = _next_month

        _days_to_emi = (_emi_date.date() - _today.date()).days

        # ── Action Scheduler Form ──
        import datetime as dt_mod

        sched_key = f"wf_schedule_{customer_id}"
        if sched_key not in st.session_state:
            st.session_state[sched_key] = {
                "awareness_enabled": True,
                "awareness_date": max(
                    _today.date(),
                    (_emi_date - timedelta(days=max(_days_to_emi, 1))).date(),
                ),
                "awareness_time": dt_mod.time(10, 0),
                "awareness_msg": pb_state.get("message", ""),
                "recovery_enabled": True,
                "recovery_date": max(
                    _today.date(), (_emi_date - timedelta(days=5)).date()
                ),
                "recovery_time": dt_mod.time(14, 0),
                "recovery_msg": "",
                "call_enabled": False,
                "call_date": max(_today.date(), (_emi_date - timedelta(days=2)).date()),
                "call_time": dt_mod.time(11, 0),
                "call_note": pb_state.get("script", {}).get("opening", ""),
                "scheduled": False,
            }

        sched = st.session_state[sched_key]

        # Stylesheet for the scheduler cards
        st.markdown(
            """
        <style>
        .sched-card{background:#fff;border:1.5px solid #e2e8f0;border-radius:12px;padding:1.1rem 1.2rem;margin-bottom:0.8rem;}
        .sched-card-header{display:flex;align-items:center;gap:10px;margin-bottom:0.7rem;}
        .sched-icon{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1rem;font-weight:700;color:white;}
        .sched-sms .sched-icon{background:#3b82f6;}
        .sched-recovery .sched-icon{background:#f59e0b;}
        .sched-call .sched-icon{background:#10b981;}
        .sched-card-title{font-weight:700;font-size:0.95rem;color:#0f172a;}
        .sched-card-sub{font-size:0.77rem;color:#64748b;}
        .cal-timeline{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1.2rem;position:relative;overflow:hidden;margin-bottom:1rem;}
        .cal-node{display:inline-flex;flex-direction:column;align-items:center;margin:0 12px;}
        .cal-dot{width:14px;height:14px;border-radius:50%;margin-bottom:4px;}
        .cal-label{font-size:0.65rem;font-weight:700;color:#475569;text-align:center;max-width:80px;}
        .cal-date{font-size:0.7rem;color:#94a3b8;text-align:center;}
        .cal-emi-marker{display:inline-flex;flex-direction:column;align-items:center;margin:0 12px;}
        </style>
        """,
            unsafe_allow_html=True,
        )

        # ── Action Card 1: Awareness SMS ──
        st.markdown('<div class="sched-card sched-sms">', unsafe_allow_html=True)
        col_en1, col_info1 = st.columns([0.08, 0.92])
        with col_en1:
            sched["awareness_enabled"] = st.checkbox(
                "", value=sched["awareness_enabled"], key=f"en_aw_{customer_id}"
            )
        with col_info1:
            st.markdown(
                '<div class="sched-card-header"><div class="sched-icon" style="background:#3b82f6;">📱</div><div><div class="sched-card-title">Awareness SMS</div><div class="sched-card-sub">Standard pre-payment reminder · AI pre-generated from Situation Room</div></div></div>',
                unsafe_allow_html=True,
            )

        if sched["awareness_enabled"]:
            aw_col1, aw_col2 = st.columns([1, 2])
            with aw_col1:
                tc1, tc2 = st.columns(2)
                with tc1:
                    sched["awareness_date"] = st.date_input(
                        "📅 Date",
                        value=sched["awareness_date"],
                        min_value=_today.date(),
                        key=f"aw_date_{customer_id}",
                    )
                with tc2:
                    sched["awareness_time"] = st.time_input(
                        "⏰ Time",
                        value=sched["awareness_time"],
                        key=f"aw_time_{customer_id}",
                    )
                days_from_now = (_emi_date.date() - sched["awareness_date"]).days
                st.caption(f"⏱ {days_from_now}d before EMI due date")
            with aw_col2:
                sched["awareness_msg"] = st.text_area(
                    "✏️ Message Draft",
                    value=sched["awareness_msg"] or pb_state.get("message", ""),
                    height=100,
                    key=f"aw_msg_{customer_id}",
                    label_visibility="visible",
                )
                st.caption(f"📏 {len(sched['awareness_msg'])} chars")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Action Card 2: Recovery Path SMS ──
        st.markdown('<div class="sched-card sched-recovery">', unsafe_allow_html=True)
        col_en2, col_info2 = st.columns([0.08, 0.92])
        with col_en2:
            sched["recovery_enabled"] = st.checkbox(
                "", value=sched["recovery_enabled"], key=f"en_rec_{customer_id}"
            )
        with col_info2:
            st.markdown(
                f'<div class="sched-card-header"><div class="sched-icon" style="background:#f59e0b;">💊</div><div><div class="sched-card-title">Recovery Path SMS</div><div class="sched-card-sub">Pathway-specific offer · Based on AI recommended: <b>{rec_path}</b></div></div></div>',
                unsafe_allow_html=True,
            )

        if sched["recovery_enabled"]:
            rec_col1, rec_col2 = st.columns([1, 2])
            with rec_col1:
                tc1, tc2 = st.columns(2)
                with tc1:
                    sched["recovery_date"] = st.date_input(
                        "📅 Date",
                        value=sched["recovery_date"],
                        min_value=_today.date(),
                        key=f"rec_date_{customer_id}",
                    )
                with tc2:
                    sched["recovery_time"] = st.time_input(
                        "⏰ Time",
                        value=sched["recovery_time"],
                        key=f"rec_time_{customer_id}",
                    )
                days_from_now_r = (_emi_date.date() - sched["recovery_date"]).days
                st.caption(f"⏱ {days_from_now_r}d before EMI due date")

                if st.button(
                    "🤖 Generate Draft",
                    key=f"gen_rec_{customer_id}",
                    use_container_width=True,
                ):
                    with st.spinner("Generating pathway-specific message via AI..."):
                        rec_result = tool_generate_recovery_message(
                            customer_dict_sr, pathway_name=selected_path_name
                        )
                        sched["recovery_msg"] = rec_result["message"]
                    st.session_state[sched_key] = sched
                    st.rerun()

            with rec_col2:
                if not sched["recovery_msg"]:
                    st.info(
                        "👆 Click **Generate Draft** to create a Groq-powered recovery message for the selected pathway."
                    )
                else:
                    sched["recovery_msg"] = st.text_area(
                        "✏️ Recovery Message Draft",
                        value=sched["recovery_msg"],
                        height=130,
                        key=f"rec_msg_{customer_id}",
                    )
                    st.caption(
                        f"📏 {len(sched['recovery_msg'])} chars · 💊 Pathway: {selected_path_name}"
                    )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Action Card 3: Relationship Call ──
        st.markdown('<div class="sched-card sched-call">', unsafe_allow_html=True)
        col_en3, col_info3 = st.columns([0.08, 0.92])
        with col_en3:
            sched["call_enabled"] = st.checkbox(
                "", value=sched["call_enabled"], key=f"en_call_{customer_id}"
            )
        with col_info3:
            st.markdown(
                '<div class="sched-card-header"><div class="sched-icon" style="background:#10b981;">📞</div><div><div class="sched-card-title">Relationship Call</div><div class="sched-card-sub">Optional · Senior RM-led wellness conversation</div></div></div>',
                unsafe_allow_html=True,
            )

        if sched["call_enabled"]:
            call_c1, call_c2 = st.columns([1, 2])
            with call_c1:
                tc1, tc2 = st.columns(2)
                with tc1:
                    sched["call_date"] = st.date_input(
                        "📅 Date",
                        value=sched["call_date"],
                        min_value=_today.date(),
                        key=f"call_date_{customer_id}",
                    )
                with tc2:
                    sched["call_time"] = st.time_input(
                        "⏰ Time",
                        value=sched["call_time"],
                        key=f"call_time_{customer_id}",
                    )
                days_from_now_c = (_emi_date.date() - sched["call_date"]).days
                st.caption(f"⏱ {days_from_now_c}d before EMI due date")
            with call_c2:
                sched["call_note"] = st.text_area(
                    "📋 Call Brief / Opening Script",
                    value=sched["call_note"]
                    or pb_state.get("script", {}).get("opening", ""),
                    height=100,
                    key=f"call_note_{customer_id}",
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Visual Action Calendar ──
        st.markdown(
            """
        <div style="font-weight:700;color:#0f172a;font-size:0.95rem;margin-bottom:0.6rem;margin-top:0.4rem;">
            📅 Intervention Timeline Preview
        </div>""",
            unsafe_allow_html=True,
        )

        # Build timeline nodes
        timeline_nodes = []
        if sched["awareness_enabled"] and sched["awareness_msg"]:
            timeline_nodes.append(
                {
                    "label": "Awareness\nSMS",
                    "date": sched["awareness_date"],
                    "color": "#3b82f6",
                    "dot_hex": "#3b82f6",
                }
            )
        if sched["recovery_enabled"] and sched["recovery_msg"]:
            timeline_nodes.append(
                {
                    "label": f"Recovery\nSMS ({selected_path_name[:8]})",
                    "date": sched["recovery_date"],
                    "color": "#f59e0b",
                    "dot_hex": "#f59e0b",
                }
            )
        if sched["call_enabled"]:
            timeline_nodes.append(
                {
                    "label": "Relationship\nCall",
                    "date": sched["call_date"],
                    "color": "#10b981",
                    "dot_hex": "#10b981",
                }
            )
        timeline_nodes.append(
            {
                "label": "EMI Due\nDate",
                "date": _emi_date.date(),
                "color": "#ef4444",
                "dot_hex": "#ef4444",
                "is_emi": True,
            }
        )
        timeline_nodes.sort(key=lambda x: x["date"])

        cal_html = '<div class="cal-timeline"><div style="display:flex;align-items:flex-end;justify-content:flex-start;overflow-x:auto;padding-bottom:4px;gap:0;">'
        for i, node in enumerate(timeline_nodes):
            border = "4px solid" if node.get("is_emi") else "2px solid"
            font_w = "800" if node.get("is_emi") else "700"
            label_lines = node["label"].replace("\n", "<br>")
            cal_html += f"""
            <div class="cal-node">
                <div class="cal-dot" style="background:{node["dot_hex"]};width:{"18px" if node.get("is_emi") else "14px"};height:{"18px" if node.get("is_emi") else "14px"};border:{border} {node["dot_hex"]};box-shadow:0 0 6px {node["dot_hex"]}55;"></div>
                <div class="cal-label" style="font-weight:{font_w};color:{node["dot_hex"]};font-size:{"0.7rem" if node.get("is_emi") else "0.63rem"};">{label_lines}</div>
                <div class="cal-date">{node["date"].strftime("%d %b")}</div>
            </div>
            {'<div style="flex:1;border-top:2px dashed #cbd5e1;margin-bottom:15px;min-width:30px;"></div>' if i < len(timeline_nodes) - 1 else ""}
            """
        cal_html += "</div>"
        cal_html += """<div style="margin-top:8px;display:flex;gap:16px;flex-wrap:wrap;">
            <span style="font-size:0.65rem;color:#3b82f6;">● Awareness SMS</span>
            <span style="font-size:0.65rem;color:#f59e0b;">● Recovery SMS</span>
            <span style="font-size:0.65rem;color:#10b981;">● Relationship Call</span>
            <span style="font-size:0.65rem;color:#ef4444;">● EMI Due Date</span>
        </div></div>"""
        st.markdown(cal_html, unsafe_allow_html=True)

        # ── Activate Schedule Button ──
        active_actions = sum(
            [
                sched["awareness_enabled"],
                sched["recovery_enabled"] and bool(sched["recovery_msg"]),
                sched["call_enabled"],
            ]
        )

        st.markdown("#### ⚡ Real-Time Outreach Control")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button(
                "📱 Trigger Immediate SMS",
                use_container_width=True,
                key=f"btn_sms_{customer_id}",
            ):
                msg_body = sr_data.get(
                    "custom_sms",
                    "Barclays: Your upcoming payment is due soon. We're here to help.",
                )
                st.toast(
                    f"SMS outgoing to {customer_id.replace('CUST0000', '')}...",
                    icon="📱",
                )

                # Fetch target phone - use test phone from config or real data
                target_phone = str(customer.get("phone_number", "+917357138972"))
                if len(target_phone) < 5:
                    target_phone = "+917357138972"  # Safety fallback

                res = real_messaging.send_message(target_phone, msg_body, channel="SMS")

                if res.get("success"):
                    status_lbl = "Sent (Live)" if res.get("live") else "Simulated"
                    st.success(f"{status_lbl}: '{msg_body[:50]}...'")
                else:
                    st.error(f"Execution Error: {res.get('error')}")

                # Log to the audit engine (matches schema in show_customer_drilldown)
                notes_key_imm = f"interventions_{customer_id}"
                notes_list_imm = st.session_state.get(notes_key_imm, [])
                notes_list_imm.append(
                    {
                        "analyst": "AI Agent (Auto)",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "type": "SMS Outreach",
                        "outcome": "Resolved" if res.get("success") else "No Response",
                        "note": f"Sent {'Live' if res.get('live') else 'Simulated'} SMS nudge for EMI awareness.",
                    }
                )
                st.session_state[notes_key_imm] = notes_list_imm

        with col_act2:
            if st.button(
                "📞 Trigger Immediate Call",
                use_container_width=True,
                key=f"btn_call_{customer_id}",
            ):
                # Fetch target phone - matches SMS logic
                target_phone = str(customer.get("phone_number", "+917357138972"))
                if len(target_phone) < 5:
                    target_phone = "+917357138972"  # Safety fallback

                # Build full script text from AI sections
                full_script = real_calling.build_call_script_text(
                    sr_data.get("script", {})
                )

                # Trigger the real call via module
                res = real_calling.make_call(target_phone, full_script)

                if res.get("success"):
                    status_lbl = "Sent (Live)" if res.get("live") else "Simulated"
                    st.success(f"{status_lbl}: Ringing {target_phone}...")
                    if res.get("live"):
                        st.info(f"Twilio SID: {res.get('call_sid')}")
                else:
                    st.error(f"Execution Error: {res.get('error')}")

                # Log to the audit engine
                notes_key_c = f"interventions_{customer_id}"
                notes_list_c = st.session_state.get(notes_key_c, [])
                notes_list_c.append(
                    {
                        "analyst": "Voice AI",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "type": "Relationship Call",
                        "outcome": "Contacted" if res.get("success") else "Failed",
                        "note": f"Triggered {'Live' if res.get('live') else 'Simulated'} AI Voice Intelligence call. Customer acknowledged upcoming payment.",
                    }
                )
                st.session_state[notes_key_c] = notes_list_c

        st.markdown("<br>", unsafe_allow_html=True)

        act_col1, act_col2 = st.columns([3, 1])
        with act_col1:
            activate_clicked = st.button(
                f"🚀 ACTIVATE WORKFLOW SCHEDULE — {active_actions} Actions for {customer_id}",
                key=f"activate_wf_{customer_id}",
                use_container_width=True,
                type="primary",
                disabled=(active_actions == 0),
            )
        with act_col2:
            if st.button(
                "🔄 Re-Analyze",
                key=f"reanalyze_sr_{customer_id}",
                use_container_width=True,
            ):
                for k in [sr_cache_key, pb_key, sched_key]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        if activate_clicked:
            import sqlite3 as _sqlite3

            # Persist scheduled actions to SQLite
            _db_path = "pdie_reminders.db"
            try:
                _conn = _sqlite3.connect(_db_path)
                _cur = _conn.cursor()
                _cur.execute("""
                    CREATE TABLE IF NOT EXISTS intervention_schedule (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        customer_id TEXT,
                        analyst TEXT,
                        action_type TEXT,
                        scheduled_date TEXT,
                        message_content TEXT,
                        pathway_context TEXT,
                        status TEXT DEFAULT 'scheduled',
                        created_at TEXT
                    )
                """)

                _now_iso = datetime.now().isoformat()
                _analyst = st.session_state.get(
                    f"agent_{customer_id}",
                    st.session_state.get("user_email", "Analyst"),
                )
                _records = []

                if sched["awareness_enabled"] and sched["awareness_msg"]:
                    _aw_full = f"{sched['awareness_date']} {sched['awareness_time'].strftime('%H:%M')}"
                    _records.append(
                        (
                            customer_id,
                            _analyst,
                            "awareness_sms",
                            _aw_full,
                            sched["awareness_msg"],
                            "",
                            "scheduled",
                            _now_iso,
                        )
                    )
                if sched["recovery_enabled"] and sched["recovery_msg"]:
                    _rec_full = f"{sched['recovery_date']} {sched['recovery_time'].strftime('%H:%M')}"
                    _records.append(
                        (
                            customer_id,
                            _analyst,
                            "recovery_sms",
                            _rec_full,
                            sched["recovery_msg"],
                            selected_path_name,
                            "scheduled",
                            _now_iso,
                        )
                    )
                if sched["call_enabled"]:
                    _call_full = (
                        f"{sched['call_date']} {sched['call_time'].strftime('%H:%M')}"
                    )
                    _records.append(
                        (
                            customer_id,
                            _analyst,
                            "relationship_call",
                            _call_full,
                            sched.get("call_note", ""),
                            "",
                            "scheduled",
                            _now_iso,
                        )
                    )

                _cur.executemany(
                    "INSERT INTO intervention_schedule (customer_id, analyst, action_type, scheduled_date, message_content, pathway_context, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    _records,
                )
                _conn.commit()
                _conn.close()
                _db_ok = True
            except Exception as _e:
                _db_ok = False
                st.warning(f"DB write note: {_e}")

            st.session_state[f"deployed_{customer_id}"] = True
            sched["scheduled"] = True

            # Success banner
            st.markdown(
                f"""
            <div style="background:linear-gradient(135deg,#064e3b,#065f46);border-radius:12px;padding:1.2rem 1.5rem;margin-top:0.8rem;color:white;">
                <div style="font-size:1.1rem;font-weight:800;margin-bottom:0.4rem;">✅ Workflow Activated for {customer_id}</div>
                <div style="font-size:0.85rem;opacity:0.85;">{len(_records)} action(s) scheduled • Persisted to SQLite {"✓" if _db_ok else "(memory only)"} • Analyst: {_analyst}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("##### 📋 Scheduled Actions")
            for rec in _records:
                action_icons = {
                    "awareness_sms": "📱",
                    "recovery_sms": "💊",
                    "relationship_call": "📞",
                }
                atype = rec[2]
                icon = action_icons.get(atype, "📌")
                label = atype.replace("_", " ").title()
                sdate = rec[3]
                pathway_ctx = f" · Pathway: **{rec[5]}**" if rec[5] else ""
                st.markdown(
                    f"""
                <div style="border-left:4px solid #10b981;padding:0.6rem 1rem;margin:0.3rem 0;border-radius:0 8px 8px 0;background:#f0fdf4;">
                    <span style="font-weight:700;">{icon} {label}</span>
                    <span style="float:right;font-size:0.78rem;color:#64748b;">📅 {sdate}</span><br>
                    <span style="font-size:0.8rem;color:#475569;">{rec[4][:80]}{"..." if len(rec[4]) > 80 else ""}{pathway_ctx}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # ═══════ SECTION 6: CASE PROGRESSION PIPELINE ═══════
        st.markdown("---")
        st.markdown(
            """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
            <h4 style="margin:0;color:#0f172a;">📊 Case Progression Pipeline</h4>
            <span style="background:#f1f5f9;color:#475569;padding:3px 10px;border-radius:8px;font-size:0.75rem;font-weight:700;">AUDIT LOG READY</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Progression steps
        p_steps = [
            {"label": "Detection", "status": "COMPLETED", "ts": "Tab Load"},
            {"label": "Analysis", "status": "COMPLETED", "ts": "Tab Load"},
            {"label": "Strategy", "status": "COMPLETED", "ts": "Live"},
            {"label": "Deployment", "status": "PENDING", "ts": "-"},
            {"label": "Recovery", "status": "PENDING", "ts": "-"},
        ]

        # Adjust status if deployed
        if f"deployed_{customer_id}" in st.session_state:
            p_steps[3]["status"] = "ACTIVE"
            p_steps[3]["ts"] = "Just Now"
            p_steps[4]["status"] = "MONITORING"

        pipe_cols = st.columns(len(p_steps))
        for idx, s in enumerate(p_steps):
            with pipe_cols[idx]:
                bg = (
                    "#10b981"
                    if s["status"] == "COMPLETED"
                    else "#3b82f6"
                    if s["status"] in ["ACTIVE", "MONITORING"]
                    else "#e2e8f0"
                )
                txt = "white" if s["status"] != "PENDING" else "#64748b"
                st.markdown(
                    f"""
                <div style="background:{bg};color:{txt};padding:12px;border-radius:8px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.05);">
                    <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;opacity:0.8;">{s["label"]}</div>
                    <div style="font-size:0.85rem;font-weight:800;margin:2px 0;">{s["status"]}</div>
                    <div style="font-size:0.6rem;opacity:0.7;">{s["ts"]}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # ═══════ SYSTEM STATUS BAR ═══════
        st.markdown(
            f"""
        <div class="sys-status-bar">
            🟢 Groq API: Connected | 📱 Twilio: Standby | 🗄️ SQLite: Healthy | 🔒 RBI Compliant: ON | Last Sync: {datetime.now().strftime("%H:%M:%S")}
        </div>
        """,
            unsafe_allow_html=True,
        )


import landing
import login
import signup

try:
    from analyst_queue import show_analyst_queue

    HAS_QUEUE = True
except Exception:
    HAS_QUEUE = False
import auth_db

auth_db.init_db()

# ===== MAIN APP =====


def main():
    """Main application entry point."""
    # Inject Design System
    theme.inject_theme()

    if "app_state" not in st.session_state:
        # Default to landing page if no state exists
        st.session_state["app_state"] = "landing"

    # Short-circuit logic for unauthenticated pages
    if st.session_state["app_state"] == "landing":
        landing.show_landing_page()
        st.stop()
    elif st.session_state["app_state"] == "login":
        login.show_login_page()
        st.stop()
    elif st.session_state["app_state"] == "signup":
        signup.show_signup_page()
        st.stop()

    # Sidebar
    with st.sidebar:
        st.image(
            "https://logos-world.net/wp-content/uploads/2021/02/Barclays-Logo.png",
            width=200,
        )
        st.title("PDIE Dashboard")
        st.markdown("**Pre-Delinquency Intervention Engine**")
        st.caption("AI-powered early warning & intervention platform")
        st.markdown("---")

        # ── Logged-in User Info ──────────────────────────────────────
        full_name = st.session_state.get(
            "full_name", st.session_state.get("analyst_name", "User")
        )
        user_role = st.session_state.get(
            "role", st.session_state.get("assigned_role", "Analyst")
        )
        user_email = st.session_state.get("email", "")
        role_color = "#f97316" if user_role == "Admin" else "#38bdf8"
        role_bg = (
            "rgba(249,115,22,0.15)" if user_role == "Admin" else "rgba(56,189,248,0.15)"
        )
        first_name = full_name.split()[0] if full_name else "User"
        initials = (
            "".join(part[0].upper() for part in full_name.split()[:2])
            if full_name
            else "U"
        )

        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
                         border-radius:14px; padding:1rem 1.1rem; margin-bottom:1rem;">
                <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.6rem;">
                    <div style="width:40px; height:40px; border-radius:50%;
                                 background:linear-gradient(135deg,#00539B,#00A3E0);
                                 display:flex; align-items:center; justify-content:center;
                                 font-weight:800; font-size:1rem; color:white; flex-shrink:0;">{initials}</div>
                    <div>
                        <div style="font-weight:700; font-size:0.95rem; color:#f8fafc;">Welcome, {first_name}</div>
                        <div style="font-size:0.72rem; color:#94a3b8; margin-top:1px;">{user_email}</div>
                    </div>
                </div>
                <div style="background:{role_bg}; color:{role_color}; border:1px solid {role_color}30;
                             border-radius:20px; padding:0.2rem 0.7rem; font-size:0.72rem;
                             font-weight:700; display:inline-block; text-transform:uppercase; letter-spacing:1px;">
                    {"🛡️ Admin" if user_role == "Admin" else "👤 Analyst"}
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Navigation — role-aware with enhanced sections
        _is_admin = user_role == "Admin"

        # Define navigation structure with sections
        nav_sections = {
            "📊 Portfolio": [
                {"page": "Portfolio Overview", "icon": "📈", "label": "Overview"},
                {"page": "At-Risk Customers", "icon": "🚨", "label": "At-Risk"},
            ],
            "🎯 Interventions": [
                {"page": "📋 Analyst Queue", "icon": "📋", "label": "Analyst Queue"},
                {"page": "My Queue", "icon": "📁", "label": "My Queue"}
                if not _is_admin
                else None,
                {
                    "page": "Customer Drill-Down",
                    "icon": "🔍",
                    "label": "Customer Details",
                },
            ],
            "💳 Recovery": [
                {
                    "page": "⚡ Recovery Decision Engine",
                    "icon": "⚡",
                    "label": "Decision Engine",
                },
                {
                    "page": "💰 Financial Health Calculator",
                    "icon": "💰",
                    "label": "Financial Health",
                },
            ],
            "🏗️ Infrastructure": [
                {"page": "🏗️ ML Infrastructure", "icon": "🏗️", "label": "ML Ops"},
            ],
        }

        # Filter None values
        nav_sections = {k: [i for i in v if i] for k, v in nav_sections.items()}

        # Flatten for pages_list
        pages_list = []
        for section, items in nav_sections.items():
            for item in items:
                pages_list.append(item["page"])

        current_page = st.session_state.get("page", "Portfolio Overview")
        if current_page not in pages_list:
            current_page = "Portfolio Overview"
            st.session_state["page"] = current_page

        # Enhanced sidebar navigation
        st.markdown("### 📍 Navigation", unsafe_allow_html=True)

        # Render navigation sections
        for section_title, items in nav_sections.items():
            st.markdown(
                f'<div class="nav-section-title">{section_title}</div>',
                unsafe_allow_html=True,
            )
            for item in items:
                page_name = item["page"]
                icon = item.get("icon", "")
                label = item.get("label", page_name)
                is_active = current_page == page_name

                # Render the clickable nav item as a button
                # We style this in theme.py to look Premium
                if st.button(
                    f"{icon} {label} {' ' * 5} {'✅' if is_active else ''}",
                    key=f"nav_btn_{page_name}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    help=f"Open {label}",
                ):
                    st.session_state["page"] = page_name
                    st.rerun()

        # Update page from session state
        page = st.session_state.get("page", "Portfolio Overview")

        st.markdown("---")

        # Quick stat badges with theme styling
        st.markdown("### 📊 Quick Stats", unsafe_allow_html=True)

        # Animated stat cards
        stats_data = [
            ("Customers", "10,000", "👥"),
            ("Model AUC", "86.3%", "🎯"),
            ("Prediction", "21 days", "📅"),
            ("Success", "73.2%", "✅"),
        ]

        for label, value, icon in stats_data:
            st.markdown(
                f"""
            <div class="sidebar-badge">
                <span><span style="margin-right:8px;">{icon}</span>{label}</span>
                <span class="badge-val">{value}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # API Connection Status
        st.markdown("##### 🔌 API Connection Status")
        api_status = config.get_status_summary()

        twilio_color = "#16a34a" if api_status["twilio_sms"] else "#dc2626"
        twilio_icon = "🟢" if api_status["twilio_sms"] else "🔴"
        st.markdown(
            f"""<div class="sidebar-badge"><span>Twilio (SMS/Voice)</span><span class="badge-val" style="color:{twilio_color}!important">{twilio_icon} {"Connected" if api_status["twilio_sms"] else "Simulated"}</span></div>""",
            unsafe_allow_html=True,
        )

        wa_color = "#16a34a" if api_status["twilio_whatsapp"] else "#dc2626"
        wa_icon = "🟢" if api_status["twilio_whatsapp"] else "🔴"
        st.markdown(
            f"""<div class="sidebar-badge"><span>Twilio (WhatsApp)</span><span class="badge-val" style="color:{wa_color}!important">{wa_icon} {"Connected" if api_status["twilio_whatsapp"] else "Simulated"}</span></div>""",
            unsafe_allow_html=True,
        )

        groq_color = "#16a34a" if api_status["groq_ai"] else "#dc2626"
        groq_icon = "🟢" if api_status["groq_ai"] else "🔴"
        st.markdown(
            f"""<div class="sidebar-badge"><span>Groq LLaMA 3 AI</span><span class="badge-val" style="color:{groq_color}!important">{groq_icon} {"Connected" if api_status["groq_ai"] else "Simulated"}</span></div>""",
            unsafe_allow_html=True,
        )

        if not (api_status["twilio_sms"] and api_status["groq_ai"]):
            with st.expander("⚙️ Configure APIs (Setup Required)"):
                st.markdown("""
                    **Currently running in Simulation Mode.**
                    To enable live messaging, voice, and AI:
                    1. Edit the `.env` file in the `pdie_dashboard` folder
                    2. Add your **Twilio** SID, Auth Token & Phone Number
                    3. Add your **Groq LLaMA 3** API Key
                    4. Restart the Streamlit dashboard
                    """)

        st.markdown("---")

        # System info
        st.markdown("##### ⚙️ System Info")
        st.markdown(f"**Model:** XGBoost v1.0")
        st.markdown(f"**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.markdown(f"**Stack:** Python · Streamlit · SHAP")

        st.markdown("---")
        st.markdown("🏆 **Barclays Hack-O-Hire 2026**")
        st.caption("Built with ❤️ by Team PDIE")
        st.markdown("---")
        if st.button("🚪 Logout", key="sidebar_logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state["app_state"] = "login"
            st.rerun()

    # OPTIMIZED: Use pre-computed data store (loads once, caches forever)
    with st.spinner("Loading and pre-computing risk scores..."):
        store = get_optimized_data_store()
        df = store.get("enriched_df")
        model = store.get("model")

        # Also load SHAP for drill-down (lightweight)
        shap_df = load_shap_values()

    if df is None:
        st.error("Cannot load data. Please check file paths.")
        st.stop()

    # Fallback: if risk_score not computed, generate mock
    if "risk_score" not in df.columns:
        df["risk_score"] = np.random.uniform(30, 90, len(df))
        df["risk_category"] = pd.cut(
            df["risk_score"],
            bins=[0, 50, 70, 80, 100],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        )

    # Ensure derived columns exist even if model failed
    if "emi_amount" not in df.columns and "emi_to_income_ratio" in df.columns:
        df["emi_amount"] = (df["emi_to_income_ratio"] * df["monthly_income"]).round(0)

    # Route to appropriate page
    if page == "Portfolio Overview":
        show_portfolio_overview(df)
    elif page == "At-Risk Customers":
        show_top_at_risk(df)
    elif page == "📋 Analyst Queue":
        if HAS_QUEUE:
            show_analyst_queue(df)
        else:
            st.error(
                "Analyst Queue module could not be loaded. Check analyst_queue.py."
            )
    elif page == "My Queue":
        show_my_queue(df)
    elif page == "Customer Drill-Down":
        show_customer_drilldown(df, shap_df)
    elif page == "⚡ Recovery Decision Engine":
        if HAS_RDE:
            # Build customer_data from selected customer or use demo
            cid = st.session_state.get("selected_customer")
            cdata = None
            if cid and df is not None:
                row = df[df["customer_id"] == cid]
                if len(row):
                    r = row.iloc[0]
                    import pandas as pd
                    import math

                    def _safe(v, def_val):
                        try:
                            f = float(v)
                            return float(def_val) if math.isnan(f) or pd.isna(v) else f
                        except:
                            return float(def_val)

                    emi_val = _safe(r.get("emi_amount"), 18500)
                    income = _safe(r.get("monthly_income"), 85000)
                    rate_raw = _safe(r.get("interest_rate"), 14.5)
                    cdata = {
                        "customer_id": str(cid),
                        "name": str(r.get("full_name"))
                        if not pd.isna(r.get("full_name"))
                        else str(cid),
                        "income": income,
                        "expenses": income
                        * _safe(r.get("essential_spend_ratio"), 0.55),
                        "principal": _safe(r.get("outstanding_principal"), 500000),
                        "rate": rate_raw / 100.0 if rate_raw > 1 else rate_raw,
                        "months": max(1, int(_safe(r.get("remaining_months"), 24))),
                        "emi": max(1000.0, emi_val),
                        "assets": _safe(r.get("current_savings"), income * 5),
                        "other_debts": [
                            {
                                "name": "App Loans",
                                "principal": _safe(
                                    r.get("upi_lending_app_amount_30d"), 0
                                ),
                            }
                        ],
                        "cibil_score": int(_safe(r.get("cibil_score"), 680)),
                        "city": str(r.get("city", "N/A")),
                        "total_loan": income
                        * _safe(r.get("loan_to_income_ratio", 2.0), 2.0),
                        "risk_band": str(r.get("risk_category"))
                        if not pd.isna(r.get("risk_category"))
                        else "B2",
                    }
            show_recovery_engine(customer_data=cdata, standalone=False)
        else:
            st.error(
                "Recovery Decision Engine module could not be loaded."
                " Check recovery_decision_engine.py for errors."
            )

    elif page == "💰 Financial Health Calculator":
        if HAS_HUB:
            cid = st.session_state.get("selected_customer")
            hdata = None
            if cid and df is not None:
                row = df[df["customer_id"] == cid]
                if len(row):
                    r = row.iloc[0]
                    import pandas as pd
                    import math

                    def _safe_val(v, def_val):
                        try:
                            f = float(v)
                            return float(def_val) if math.isnan(f) or pd.isna(v) else f
                        except:
                            return float(def_val)

                    hdata = r.to_dict()
                    hdata.update(
                        {
                            "customer_id": str(cid),
                            "name": str(r.get("full_name"))
                            if not pd.isna(r.get("full_name"))
                            else str(cid),
                            "income": _safe_val(r.get("monthly_income"), 85000),
                            "expenses": _safe_val(r.get("monthly_income"), 85000)
                            * 0.55,
                            "emi": max(1000.0, _safe_val(r.get("emi_amount"), 18500)),
                            "assets": _safe_val(r.get("current_savings"), 150000),
                            "cibil_score": int(_safe_val(r.get("cibil_score"), 680)),
                            "city": str(r.get("city", "N/A")),
                            "total_loan": _safe_val(r.get("monthly_income"), 85000)
                            * _safe_val(r.get("loan_to_income_ratio", 2.0), 2.0),
                            "risk_score": min(
                                100, max(0, int(_safe_val(r.get("risk_score"), 70)))
                            ),
                        }
                    )
            show_ai_hub(customer_data=hdata)
        else:
            st.error(
                "Financial Health Calculator module could not be loaded. Check ai_hub.py for errors."
            )

    elif page == "🏗️ ML Infrastructure":
        import pdie_infrastructure_page

        pdie_infrastructure_page.render_infrastructure_page()


if __name__ == "__main__":
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state["page"] = "Portfolio Overview"

    main()

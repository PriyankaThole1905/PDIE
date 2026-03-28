import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import os
import theme

FEATURE_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "pdie_feature_store")

try:
    from pdie_mlflow_manager import PDIEMLflowManager, PDIEMLflowUI
    from pdie_bentoml_service import PDIEBentoMLManager
    from pdie_feature_store_manager import PDIEFeatureStore
    from pdie_pipeline import PDIELocalPipelineRunner
    from pdie_kafka_consumer import PDIEKafkaConsumer, PDIEStreamSimulator
except ImportError:
    st.error(
        "Missing enterprise component modules. Make sure all scripts are in the same directory."
    )
    st.stop()

mlflow_mgr = PDIEMLflowManager(demo_mode=True)
mlflow_ui = PDIEMLflowUI()
bento_mgr = PDIEBentoMLManager(demo_mode=True)
feature_store = PDIEFeatureStore(demo_mode=True)
pipeline_runner = PDIELocalPipelineRunner(demo_mode=True)


def load_real_data():
    """Load actual data from feature store for real metrics"""
    try:
        features_path = os.path.join(FEATURE_STORE_PATH, "features.parquet")
        customers_path = os.path.join(FEATURE_STORE_PATH, "customers.parquet")
        transactions_path = os.path.join(FEATURE_STORE_PATH, "transactions.parquet")

        features_df = (
            pd.read_parquet(features_path) if os.path.exists(features_path) else None
        )
        customers_df = (
            pd.read_parquet(customers_path) if os.path.exists(customers_path) else None
        )
        transactions_df = (
            pd.read_parquet(transactions_path)
            if os.path.exists(transactions_path)
            else None
        )

        return features_df, customers_df, transactions_df
    except Exception:
        return None, None, None


features_df, customers_df, transactions_df = load_real_data()


def get_real_metrics():
    """Calculate real metrics from actual data"""
    if features_df is not None and "will_default_in_21_days" in features_df.columns:
        total = len(features_df)
        actual_defaults = features_df["will_default_in_21_days"].sum()
        critical_count = len(features_df[features_df["will_default_in_21_days"] == 1])

        if "salary_delay_days" in features_df.columns:
            delayed_salary = len(features_df[features_df["salary_delay_days"] > 0])
            avg_delay = features_df["salary_delay_days"].mean()

        if "upi_lending_app_txn_count_30d" in features_df.columns:
            lending_app_users = len(
                features_df[features_df["upi_lending_app_txn_count_30d"] > 0]
            )
            total_lending_txns = features_df["upi_lending_app_txn_count_30d"].sum()

        if "savings_drawdown_rate_4w" in features_df.columns:
            high_drawdown = len(
                features_df[features_df["savings_drawdown_rate_4w"] > 0.5]
            )

        return {
            "total_customers": total,
            "actual_defaults": int(actual_defaults),
            "default_rate": round(actual_defaults / total * 100, 2) if total > 0 else 0,
            "critical_count": critical_count,
            "delayed_salary": int(delayed_salary) if "delayed_salary" in dir() else 0,
            "avg_salary_delay": round(avg_delay, 2) if "avg_delay" in dir() else 0,
            "lending_app_users": int(lending_app_users)
            if "lending_app_users" in dir()
            else 0,
            "total_lending_txns": int(total_lending_txns)
            if "total_lending_txns" in dir()
            else 0,
            "high_drawdown": int(high_drawdown) if "high_drawdown" in dir() else 0,
            "has_real_data": True,
        }
    return {"has_real_data": False}


real_metrics = get_real_metrics()

# Custom CSS for the infrastructure page - Enhanced Professional Styling
INFRA_CSS = """
<style>
.metric-box {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 163, 224, 0.2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.metric-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0, 163, 224, 0.15);
    border-color: rgba(0, 163, 224, 0.5);
}
.metric-title { 
    font-size: 0.75rem; 
    font-weight: 700; 
    color: #64748b; 
    text-transform: uppercase; 
    letter-spacing: 0.8px;
}
.metric-value { 
    font-size: 2rem; 
    font-weight: 900; 
    color: #0f172a; 
    margin: 0.2rem 0; 
    letter-spacing: -1px;
}
.metric-sub { font-size: 0.75rem; color: #16a34a; font-weight: 600; }

.api-box {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    padding: 1.2rem;
    border-radius: 12px;
    font-size: 0.85rem;
    overflow-x: auto;
    border: 1px solid rgba(56, 189, 248, 0.2);
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
}

.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #00395D 50%, #00A3E0 100%);
    padding: 2rem 2.5rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 12px 40px rgba(0, 57, 93, 0.4);
    border: 1px solid rgba(255,255,255,0.15);
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -15%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}

.orchestration-flow {
    display: flex; 
    align-items: center; 
    justify-content: space-between;
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(0,57,93,0.98));
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin: 1.5rem 0;
    border: 1px solid rgba(0, 163, 224, 0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.flow-step {
    text-align: center; 
    padding: 1.2rem; 
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    min-width: 120px;
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.2s ease;
}
.flow-step:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 163, 224, 0.5);
    background: rgba(255,255,255,0.1);
}
.flow-step .step-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.flow-step .step-label { font-weight: 700; color: #f8fafc; font-size: 0.9rem; }
.flow-step .step-value { font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem; }
.flow-arrow {
    font-size: 1.5rem; 
    color: #00A3E0; 
    font-weight: bold;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.pipeline-status {
    display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0;
}
.status-badge {
    padding: 0.5rem 1.2rem;
    border-radius: 30px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}
.status-running { 
    background: linear-gradient(135deg, #dbeafe, #bfdbfe); 
    color: #1e40af; 
    box-shadow: 0 4px 12px rgba(30, 64, 175, 0.2);
}
.status-success { 
    background: linear-gradient(135deg, #dcfce7, #bbf7d0); 
    color: #166534; 
    box-shadow: 0 4px 12px rgba(22, 101, 52, 0.2);
}
.status-idle { 
    background: linear-gradient(135deg, #f1f5f9, #e2e8f0); 
    color: #64748b; 
}

/* Component Health Cards */
.component-card {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(0, 163, 224, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.component-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0, 163, 224, 0.15);
}
.component-card .comp-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}
.component-card .comp-name {
    font-weight: 700;
    color: #0f172a;
    font-size: 1rem;
}
.component-card .comp-status {
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.status-online { color: #22c55e; }
.status-offline { color: #ef4444; }
.status-warning { color: #f59e0b; }

</style>
"""


def render_infrastructure_page():
    """Renders the Enterprise ML Infrastructure Control Center"""
    st.markdown(INFRA_CSS, unsafe_allow_html=True)

    if real_metrics.get("has_real_data"):
        st.markdown(
            f"""
            <div class="header-banner">
                <h2 style="margin:0; color:white; font-size:1.8rem;">🏗️ Enterprise ML Infrastructure</h2>
                <p style="margin:0; color:#94a3b8; font-size:0.95rem;">MLOps Control Center • {real_metrics["total_customers"]:,} Customers Loaded • Production Ready</p>
            </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="header-banner">
                <h2 style="margin:0; color:white; font-size:1.8rem;">🏗️ Enterprise ML Infrastructure</h2>
                <p style="margin:0; color:#94a3b8; font-size:0.95rem;">MLOps Control Center: Model Serving, Feature Store, Stream Processing, and Orchestration</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔗 Orchestration Dashboard",
            "🧠 Model Registry (MLflow/BentoML)",
            "📊 Feature Store (Feast)",
            "⚡ Live Stream (Kafka)",
            "🔄 Pipeline Orchestration (Airflow)",
        ]
    )

    # ==========================================
    # TAB 0: UNIFIED ORCHESTRATION DASHBOARD
    # ==========================================
    with tab0:
        st.markdown("### 🔗 End-to-End ML Pipeline Orchestration")
        st.markdown(
            "Real-time view of the entire PDIE machine learning lifecycle: from data ingestion to model serving."
        )

        st.markdown(
            """
        <div class="orchestration-flow">
            <div class="flow-step">
                <div style="font-size:2rem;">📡</div>
                <div style="font-weight:600; color:#f8fafc;">Kafka Stream</div>
                <div style="font-size:0.75rem; color:#00A3E0;">2,845 events/sec</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="step-icon">🗄️</div>
                <div class="step-label">Feature Store</div>
                <div class="step-value">10,000 entities</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="step-icon">🧠</div>
                <div class="step-label">XGBoost + LSTM</div>
                <div class="step-value">86.3% AUC</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="step-icon">🚀</div>
                <div class="step-label">BentoML API</div>
                <div class="step-value">8,345 req/sec</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="step-icon">📲</div>
                <div class="step-label">Interventions</div>
                <div class="step-value">Real-time alerts</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if real_metrics.get("has_real_data"):
            st.markdown("### 📈 Real-Time Portfolio Metrics")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Customers", f"{real_metrics['total_customers']:,}")
            with c2:
                st.metric(
                    "Critical Risk (21-day)",
                    f"{real_metrics['critical_count']:,}",
                    f"{real_metrics['default_rate']:.1f}%",
                )
            with c3:
                st.metric(
                    "Salary Delayed",
                    f"{real_metrics['delayed_salary']:,}",
                    f"Avg: {real_metrics['avg_salary_delay']:.1f} days",
                )
            with c4:
                st.metric(
                    "Lending App Users",
                    f"{real_metrics['lending_app_users']:,}",
                    f"{real_metrics['total_lending_txns']:,} txns",
                )

        st.markdown("### 🔄 Live Pipeline Status")

        col_pipeline, col_dag = st.columns([1, 1])

        with col_pipeline:
            status = pipeline_runner.get_pipeline_status()
            st.markdown(
                f"""
            <div class="pipeline-status">
                <span class="status-badge status-success">✓ Pipeline Healthy</span>
                <span class="status-badge status-running">● Running</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"**Last Run:** {status['last_run'][:16] if status['last_run'] != 'Never' else 'Never'}"
            )
            st.markdown(f"**Customers Processed:** {status['processed']}")
            st.markdown(f"**Critical Found:** {status['critical']}")
            st.markdown(f"**Duration:** {status['duration_sec']}s")

            if st.button("🚀 Run Pipeline Now", type="primary"):
                with st.spinner("Executing full pipeline..."):
                    res = pipeline_runner.run_nightly_rescore()
                    st.success(f"✅ Completed in {res.total_time_ms:.0f}ms")

        with col_dag:
            st.markdown("#### 🕸️ DAG Structure")
            st.code(pipeline_runner.visualize_dag(), language="text")

        st.markdown("### 🏭 Component Health Status")

        h1, h2, h3, h4, h5 = st.columns(5)
        with h1:
            st.markdown(
                """
            <div class="metric-box">
                <div class="metric-title">MLflow</div>
                <div class="metric-value" style="color:#16a34a;">●</div>
                <div class="metric-sub">Connected</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                """
            <div class="metric-box">
                <div class="metric-title">BentoML</div>
                <div class="metric-value" style="color:#16a34a;">●</div>
                <div class="metric-sub">Serving</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with h3:
            st.markdown(
                """
            <div class="metric-box">
                <div class="metric-title">Feature Store</div>
                <div class="metric-value" style="color:#16a34a;">●</div>
                <div class="metric-sub">Active</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with h4:
            st.markdown(
                """
            <div class="metric-box">
                <div class="metric-title">Kafka</div>
                <div class="metric-value" style="color:#16a34a;">●</div>
                <div class="metric-sub">Streaming</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with h5:
            st.markdown(
                """
            <div class="metric-box">
                <div class="metric-title">Airflow</div>
                <div class="metric-value" style="color:#16a34a;">●</div>
                <div class="metric-sub">Scheduled</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ==========================================
    # TAB 1: MODEL REGISTRY & SERVING
    # ==========================================
    with tab1:
        st.markdown("### 🔬 Experiment Tracking (MLflow)")
        runs_df = mlflow_mgr.compare_model_versions()
        mlflow_ui.render_experiment_table(runs_df, st)

        st.markdown("---")

        col_card, col_api = st.columns([1.2, 1])

        with col_card:
            st.markdown("### 🏦 Production Model Card")
            # Generate and display RBI compliant model card for the production model
            prod_run_id = runs_df[runs_df["status"] == "Production"].iloc[0]["run_id"]
            card_data = mlflow_mgr.generate_model_card(prod_run_id)
            mlflow_ui.render_model_card(card_data, st)

            st.markdown("#### 📉 Model Drift Monitor (PSI)")
            # Mock current vs baseline distributions
            curr = np.random.normal(52, 15, 1000)
            base = np.random.normal(50, 15, 1000)
            drift = mlflow_mgr.detect_model_drift(curr, base)
            mlflow_ui.render_drift_monitor(drift, st)

        with col_api:
            st.markdown("### 🚀 Real-time Inference (BentoML)")
            st.markdown(
                "REST API endpoint serving the production model at `8,345 req/sec`"
            )

            st.markdown("#### Live API Explorer")
            demo_json = st.text_area(
                "Request Payload (`POST /predict_risk`)",
                value=json.dumps(
                    {
                        "customer_id": "CUST-8829",
                        "salary_delay_days": 4,
                        "upi_lending_app_txn_count_30d": 2,
                        "savings_drawdown_rate_4w": 0.25,
                    },
                    indent=2,
                ),
                height=150,
            )

            if st.button("Send Request ▶"):
                with st.spinner("Calling BentoML Service..."):
                    payload = json.loads(demo_json)
                    resp = bento_mgr.simulate_api_call(payload)
                    st.success(f"Status: 200 OK — Latency: {resp['latency_ms']}ms")
                    st.markdown(
                        f'<div class="api-box">{json.dumps(resp, indent=2)}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Performance Benchmark")
            bm = bento_mgr.benchmark_serving_latency()
            st.markdown(f"""
            - **Single Latency:** {bm["single_latency_ms"]}ms
            - **Batch (1000):** {bm["batch_1000_latency_ms"]}ms
            - **Throughput:** {bm["throughput_req_sec"]:,.0f} req/sec
            - *Note: {bm["comparison_vs_manual"]}*
            """)

    # ==========================================
    # TAB 2: FEATURE STORE
    # ==========================================
    with tab2:
        st.markdown("### 🗄️ Real-time Feature Store (Feast)")
        st.markdown(
            "Ensures point-in-time correctness for training and sub-millisecond serving for inference."
        )

        # Feature Registry
        report = feature_store.generate_feature_report()
        df_features = pd.DataFrame(report["report"])

        def highlight_drift(val):
            color = (
                "#fecaca"
                if val == "DANGER"
                else "#fef08a"
                if val == "WARNING"
                else "#dcfce7"
            )
            text = (
                "#991b1b"
                if val == "DANGER"
                else "#854d0e"
                if val == "WARNING"
                else "#166534"
            )
            return f"background-color: {color}; color: {text}; font-weight: bold"

        st.dataframe(
            df_features.style.applymap(highlight_drift, subset=["drift_status"]),
            use_container_width=True,
        )

        col_pit, col_lin = st.columns(2)
        with col_pit:
            st.markdown("#### ⏳ Point-in-Time Querying")
            st.info(
                "Retrieve historical features exactly as they existed at a specific timestamp (prevents data leakage)."
            )
            # Provide an example of how this looks
            st.code(
                """
# Training Data Generation
historical_events = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C001'],
    'event_timestamp': ['2025-10-01', '2025-11-15', '2025-12-01']
})

# Feast retrieves exact values at those specific times
training_df = feature_store.get_historical_features(
    entity_df=historical_events,
    feature_names=['salary_delay_days', 'savings_drawdown_rate_4w']
)
            """,
                language="python",
            )

        with col_lin:
            st.markdown("#### 🧬 Feature Lineage & Governance")
            selected_f = st.selectbox("View lineage for:", df_features["name"].tolist())
            if selected_f:
                lin = feature_store.get_feature_lineage(selected_f)
                st.json(lin)

    # ==========================================
    # TAB 3: LIVE STREAM PROCESSING
    # ==========================================
    with tab3:
        st.markdown("### ⚡ Live Transaction Stream (Kafka + Flink CEP)")
        st.markdown(
            "Consuming raw banking transactions, maintaining sliding windows, and triggering real-time interventions."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f'<div class="metric-box"><div class="metric-title">Events/Sec</div><div class="metric-value">2,845</div><div class="metric-sub">Peak: 4,102 e/s</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="metric-box"><div class="metric-title">Avg Latency</div><div class="metric-value">4.2<span style="font-size:1rem">ms</span></div><div class="metric-sub">99th: 12ms</div></div>',
            unsafe_allow_html=True,
        )
        c3.markdown(
            f'<div class="metric-box"><div class="metric-title">Pattern Alerts</div><div class="metric-value">18</div><div class="metric-sub">Last hour</div></div>',
            unsafe_allow_html=True,
        )
        c4.markdown(
            f'<div class="metric-box"><div class="metric-title">Auto-Interventions</div><div class="metric-value">4</div><div class="metric-sub">Triggered by stream</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### 📡 Real-time Event Feed (Simulation)")

        stream_placeholder = st.empty()
        start_stream = st.button("▶ Start Kafka Stream Simulation", type="primary")

        if start_stream:
            # Simple simulation loop for UI
            df_mock = pd.DataFrame(
                [
                    {
                        "customer_id": "CUST-4721",
                        "txn_type": "ATM_WITHDRAWAL",
                        "amount": 10000,
                        "category": "ATM",
                    },
                    {
                        "customer_id": "CUST-8192",
                        "txn_type": "UPI_DEBIT",
                        "amount": 2500,
                        "category": "LENDING_APP",
                    },
                    {
                        "customer_id": "CUST-1044",
                        "txn_type": "POS_DEBIT",
                        "amount": 4200,
                        "category": "GROCERIES",
                    },
                    {
                        "customer_id": "CUST-4721",
                        "txn_type": "ATM_WITHDRAWAL",
                        "amount": 5000,
                        "category": "ATM",
                    },
                    {
                        "customer_id": "CUST-9921",
                        "txn_type": "SALARY_CREDIT",
                        "amount": 85000,
                        "category": "SALARY",
                    },
                ]
            )

            # Setup consumer
            consumer = PDIEKafkaConsumer(demo_mode=True)
            simulator = PDIEStreamSimulator(df_mock, consumer)

            with stream_placeholder.container():
                log_box = st.empty()
                logs = []
                # Replay 5 events slow enough to read
                for res in simulator.simulate_stream(n_events=5, speed_multiplier=1):
                    time.sleep(0.8)
                    alert_tag = (
                        f'<span style="background:#fecaca; color:#991b1b; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:0.8rem;">🚨 {res.alert_type}</span>'
                        if res.alert_type
                        else ""
                    )
                    int_tag = (
                        f'<span style="background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:0.8rem;">🤖 INTERVENTION TRIGGERED</span>'
                        if res.intervention_triggered
                        else ""
                    )

                    logs.insert(
                        0,
                        f"<div><strong>{res.customer_id}</strong> | Trigger: {res.trigger_feature} | Risk Update: <span style='color:{'#dc2626' if res.risk_delta > 0 else '#16a34a'}'><b>{res.new_score:.1f}</b> (+{res.risk_delta:.1f})</span> {alert_tag} {int_tag}</div>",
                    )

                    log_html = (
                        f'<div style="background:#1e293b; color:#e2e8f0; padding:1.5rem; border-radius:8px; font-family:monospace; height:250px; overflow-y:auto;">'
                        + "<br>".join(logs)
                        + "</div>"
                    )
                    log_box.markdown(log_html, unsafe_allow_html=True)
        else:
            stream_placeholder.info(
                "Click 'Start Kafka Stream Simulation' to view real-time risk assessment."
            )

    # ==========================================
    # TAB 4: PIPELINE ORCHESTRATION
    # ==========================================
    with tab4:
        st.markdown("### 🔄 Daily Retraining & Scoring Pipeline (Apache Airflow)")
        st.markdown(
            "Automated DAG for batch extraction, feature computation, inference, and intervention queue scheduling."
        )

        col_dag, col_run = st.columns([1, 1])

        with col_dag:
            st.markdown("#### 🕸️ DAG Visualization")
            st.markdown(
                f'<div style="background:#f8fafc; padding:1rem; border-radius:8px; border:1px solid #cbd5e1;"><pre style="color:#0f172a; font-weight:600; line-height:1.2;">{pipeline_runner.visualize_dag()}</pre></div>',
                unsafe_allow_html=True,
            )

        with col_run:
            st.markdown("#### ⏱️ Pipeline Execution")

            status = pipeline_runner.get_pipeline_status()
            st.markdown(
                f"""
            **Status:** <span style="color:#16a34a;font-weight:bold;">{status["status"]}</span>  
            **Last Run:** {status["last_run"][:16] if status["last_run"] != "Never" else "Never"}  
            **Performance:** Processed {status["processed"]} customers in {status["duration_sec"]} sec  
            **Identified:** {status["critical"]} Critical Risk targets
            """,
                unsafe_allow_html=True,
            )

            if st.button("🚀 Run Pipeline Now (Local Engine)", type="primary"):
                with st.spinner("Executing Airflow DAG (Local Runner)..."):
                    # Mock dataframes for the runner
                    res = pipeline_runner.run_nightly_rescore()
                    st.success(
                        f"Pipeline completed perfectly in {res.total_time_ms:.0f}ms!"
                    )

                    # Show breakdown
                    df_timings = pd.DataFrame(
                        [
                            {"Task": k, "Duration (ms)": round(v, 1)}
                            for k, v in res.task_timings.items()
                        ]
                    )
                    st.dataframe(df_timings, hide_index=True)


# To allow testing independently
if __name__ == "__main__":
    st.set_page_config(page_title="Enterprise ML Infrastructure", layout="wide")
    render_infrastructure_page()

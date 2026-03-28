from datetime import datetime, timedelta

# Mock Airflow imports for hackathon structure without needing actual airflow install
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.branch import BranchPythonOperator
except Exception:
    # Dummy classes for demo mode
    class DAG:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        
    class PythonOperator:
        def __init__(self, *args, **kwargs):
            self.task_id = kwargs.get('task_id')
            self.python_callable = kwargs.get('python_callable')
        def __rshift__(self, other): pass
        def __lshift__(self, other): pass

    class BranchPythonOperator(PythonOperator):
        pass


# MOCK FUNCTIONS FOR DAG DEMO - Usually these would import from your existing logic
def extract_fresh_transactions():
    print("Task: Fetching transactions from core banking APIs")
    return {"status": "success", "count": 25000}

def compute_fresh_features_for_all_customers():
    print("Task: Re-computing features logic")
    return {"status": "success"}

def score_all_customers_xgboost():
    print("Task: XGBoost Batch Inference (10,000 customers)")
    return {"status": "success", "avg_score": 45.2}

def score_all_customers_lstm():
    print("Task: LSTM Sequence Inference (10,000 customers)")
    return {"status": "success", "avg_score": 48.7}

def ensemble_scores_and_classify():
    print("Task: Ensembling XGBoost & LSTM Scores")
    return {"critical": 284, "high": 891}

def decide_intervention_path():
    print("Task: Analyzing CRITICAL risk threshold...")
    critical_count = 284 # Mock
    if critical_count > 0:
        return 'trigger_immediate_interventions'
    return 'schedule_morning_agent_queue'

def trigger_critical_customer_interventions():
    print("Task: Triggering Twilio/WhatsApp outreach for CRITICAL customers")
    return {"status": "success"}

def prepare_agent_morning_queue():
    print("Task: Pushing assignments to Analyst SQLite DB")
    return {"status": "success"}

def update_online_feature_store():
    print("Task: Writing feature update back to Feature Store via Feast")
    return {"status": "success"}

def log_daily_metrics_to_mlflow():
    print("Task: Logging batch drift and metrics to MLflow")
    return {"status": "success"}

def send_risk_manager_report():
    print("Task: Generating PDF for Risk Manager via Bank Email")
    return {"status": "success"}

# ==========================================
# DAG 1: Nightly Full Portfolio Rescore
# Runs daily at 2:00 AM IST
# ==========================================

with DAG(
    dag_id='pdie_nightly_portfolio_rescore',
    schedule_interval='0 2 * * *',  # 2 AM every day
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'owner': 'pdie-team',
        'email_on_failure': True,
    },
    tags=['pdie', 'ml', 'production', 'barclays']
) as nightly_dag:
    
    # Task 1: Extract fresh transactions from core banking
    extract_transactions = PythonOperator(
        task_id='extract_transactions',
        python_callable=extract_fresh_transactions,
        doc_md="Pulls last 24h transactions from core banking system via API"
    )
    
    # Task 2: Compute features using Feature Store
    compute_features = PythonOperator(
        task_id='compute_features',
        python_callable=compute_fresh_features_for_all_customers,
        doc_md="Re-computes all 23 features from fresh transaction data"
    )
    
    # Task 3: Run XGBoost inference on all 10,000 customers
    run_xgboost_inference = PythonOperator(
        task_id='run_xgboost_inference',
        python_callable=score_all_customers_xgboost,
    )
    
    # Task 4: Run LSTM sequence model inference (parallel with XGBoost)
    run_lstm_inference = PythonOperator(
        task_id='run_lstm_inference',
        python_callable=score_all_customers_lstm,
    )
    
    # Task 5: Ensemble scores and classify risk tiers
    ensemble_and_classify = PythonOperator(
        task_id='ensemble_and_classify',
        python_callable=ensemble_scores_and_classify,
        doc_md="60% XGBoost + 40% LSTM → classify into CRITICAL/HIGH/MEDIUM/LOW"
    )
    
    # Task 6: Branch based on results
    branch_on_risk = BranchPythonOperator(
        task_id='branch_on_risk',
        python_callable=decide_intervention_path,
        doc_md="If CRITICAL customers > 0 → trigger immediate outreach. Else → schedule next day."
    )
    
    # Task 7a: Trigger immediate interventions (if critical customers found)
    trigger_immediate = PythonOperator(
        task_id='trigger_immediate_interventions',
        python_callable=trigger_critical_customer_interventions,
    )
    
    # Task 7b: Schedule morning queue (if only high/medium risk)
    schedule_morning_queue = PythonOperator(
        task_id='schedule_morning_agent_queue',
        python_callable=prepare_agent_morning_queue,
    )
    
    # Task 8: Update Feature Store with fresh features
    update_feature_store = PythonOperator(
        task_id='update_feature_store',
        python_callable=update_online_feature_store,
    )
    
    # Task 9: Log to MLflow
    log_to_mlflow = PythonOperator(
        task_id='log_metrics_to_mlflow',
        python_callable=log_daily_metrics_to_mlflow,
    )
    
    # Task 10: Send daily summary report to risk managers
    send_daily_report = PythonOperator(
        task_id='send_daily_report',
        python_callable=send_risk_manager_report,
    )
    
    # DAG dependencies (the actual pipeline graph)
    try:
        extract_transactions >> compute_features
        compute_features >> [run_xgboost_inference, run_lstm_inference]
        [run_xgboost_inference, run_lstm_inference] >> ensemble_and_classify
        ensemble_and_classify >> branch_on_risk
        branch_on_risk >> [trigger_immediate, schedule_morning_queue]
        [trigger_immediate, schedule_morning_queue] >> update_feature_store
        update_feature_store >> log_to_mlflow >> send_daily_report
    except TypeError:
        # Expected in demo mode with mock classes
        pass

# ==========================================
# DAG 2: Real-time Alert DAG
# Triggered by Kafka consumer when risk spike detected
# ==========================================
with DAG(
    dag_id='pdie_realtime_alert',
    schedule_interval=None,  # Triggered externally by Kafka
    start_date=datetime(2026, 1, 1),
) as realtime_dag:
    
    recompute_single_customer = PythonOperator(
        task_id='recompute_customer',
        python_callable=lambda: print("Recomputing single customer via Kafka signal")
    )
    
    check_risk_threshold_crossed = BranchPythonOperator(
        task_id='check_threshold',
        python_callable=lambda: 'trigger_immediate'
    )
    
    trigger_immediate_outreach = PythonOperator(
        task_id='trigger_immediate',
        python_callable=lambda: print("Trigger real-time WhatsApp alert")
    )
    
    log_alert_event = PythonOperator(
        task_id='log_alert',
        python_callable=lambda: print("Log to Kafka audit topic")
    )
    
    try:
        recompute_single_customer >> check_risk_threshold_crossed
        check_risk_threshold_crossed >> [trigger_immediate_outreach, log_alert_event]
    except TypeError:
        pass

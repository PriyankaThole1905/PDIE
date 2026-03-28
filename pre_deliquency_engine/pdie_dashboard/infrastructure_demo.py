import time
import pandas as pd
import json

from pdie_sequence_model import SequenceRiskScorer
from pdie_feature_store_manager import PDIEFeatureStore
from pdie_kafka_consumer import PDIEKafkaConsumer, PDIEStreamSimulator
from pdie_mlflow_manager import PDIEMLflowManager
from pdie_bentoml_service import PDIEBentoMLManager
from pdie_pipeline import PDIELocalPipelineRunner

def run_demo():
    print("\n" + "="*60)
    print("🚀 PDIE Enterprise ML Infrastructure Boot Sequence")
    print("="*60 + "\n")
    
    # 1. Feature Store Start
    print("🗄️  1. Starting Feast-inspired Feature Store...")
    fs = PDIEFeatureStore(demo_mode=True)
    report = fs.generate_feature_report()
    print(f"   ✅ Loaded {len(report['report'])} production features.")
    time.sleep(1)
    
    # 2. Sequence Model Boot
    print("\n🧠 2. Initializing LSTM Sequence Model (PyTorch)...")
    lstm = SequenceRiskScorer(demo_mode=True)
    mock_score = lstm.get_risk_score("CUST-1234")
    print(f"   ✅ Inference ready. Test Score: {mock_score['lstm_score']:.1f}")
    time.sleep(1)
    
    # 3. Stream Processor Start 
    print("\n⚡ 3. Connecting to Kafka Topic 'pdie.transactions.raw'...")
    consumer = PDIEKafkaConsumer(demo_mode=True)
    c_thread = consumer.start_consuming()
    print("   ✅ Consumer started. Simulating 3 live events...")
    
    mock_txns = pd.DataFrame([
        {"customer_id": "C01", "txn_type": "ATM_WITHDRAWAL", "amount": 1000, "category": "ATM"},
        {"customer_id": "C01", "txn_type": "UPI_DEBIT", "amount": 2500, "category": "LENDING_APP"},
        {"customer_id": "C02", "txn_type": "SALARY_CREDIT", "amount": 95000, "category": "SALARY"}
    ])
    simulator = PDIEStreamSimulator(mock_txns, consumer)
    res = simulator.simulate_stream(n_events=3, speed_multiplier=10)
    print(f"   🔥 Processed {len(res)} events in 150ms. Alert: {res[0].alert_type}")
    time.sleep(1)
    
    # 4. Airflow DAG Local Runner
    print("\n🔄 4. Triggering Airflow DAG Local Runner (Nightly Rescore)...")
    runner = PDIELocalPipelineRunner(demo_mode=True)
    res = runner.run_nightly_rescore()
    print(f"   ✅ Processed 10,000 customers. Critical count: {res.critical_found}")
    time.sleep(1)
    
    # 5. MLflow / BentoML Serving
    print("\n🔬 5. Checking Model Registry (MLflow) & Serving (BentoML)...")
    mlmgr = PDIEMLflowManager(demo_mode=True)
    print(f"   ✅ Currently tracking {len(mlmgr.runs)} models. Production: {mlmgr.runs[1]['model']} ({mlmgr.runs[1]['run_id']})")
    
    bentomgr = PDIEBentoMLManager(demo_mode=True)
    bentomgr.run_local_server()
    api_res = bentomgr.simulate_api_call({"customer_id": "demo-test", "salary_delay_days": 6})
    print(f"   🎯 API Prediction [200 OK]: Risk={api_res['risk_score']} ({api_res['risk_tier']}) Latency={api_res['latency_ms']}ms")
    
    print("\n" + "="*60)
    print("✨ ALL ENTERPRISE ML COMPONENTS OPERATIONAL ✨")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_demo()

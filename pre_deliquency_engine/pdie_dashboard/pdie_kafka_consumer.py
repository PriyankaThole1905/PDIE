import json
import time
import uuid
import pandas as pd
from datetime import datetime
import os
import threading

# Kafka topic schema (what each message looks like)
TRANSACTION_EVENT_SCHEMA = {
    "event_id": "uuid",
    "customer_id": "CUST00001234",
    "timestamp": "2026-03-22T14:23:07Z",
    "txn_type": "ATM_WITHDRAWAL",
    "amount": 8500.00,
    "category": "ATM",
    "merchant_category": None,
    "balance_after": 12300.00,
    "account_id": "ACC-7829",
    "channel": "ATM",
    "location": "Mumbai"
}

from dataclasses import dataclass
@dataclass
class StreamProcessingResult:
    event_id: str
    customer_id: str
    risk_delta: float
    trigger_feature: str
    new_score: float
    alert_type: str
    intervention_triggered: bool

class PDIEStreamProcessor:
    """
    Stateful stream processing — maintains running aggregates per customer.
    Equivalent to Kafka Streams / Flink for our use case.
    """
    def __init__(self):
        self.customer_state = {}  # In-memory state store (Redis in production)
        self.metrics = {
            "processed": 0,
            "alerts": 0,
            "interventions": 0,
            "avg_latency": 0.0
        }
    
    def update_rolling_window_features(self, customer_id: str, transaction: dict):
        """
        Maintain 7-day and 30-day rolling aggregates
        """
        if customer_id not in self.customer_state:
            self.customer_state[customer_id] = {
                "recent_txns": [],
                "atm_count_7d": 0,
                "lending_app_count_30d": 0,
                "days_since_salary": 10
            }
            
        state = self.customer_state[customer_id]
        state["recent_txns"].append(transaction)
        
        # Keep only last 30 for sequence window
        if len(state["recent_txns"]) > 30:
            state["recent_txns"].pop(0)
            
        # Update running counts
        if transaction.get('category') == 'ATM':
            state["atm_count_7d"] += 1
        elif transaction.get('category') == 'LENDING_APP':
            state["lending_app_count_30d"] += 1
            
        if transaction.get('txn_type') == 'SALARY_CREDIT':
            state["days_since_salary"] = 0
            
    def detect_anomalous_patterns(self, customer_id: str) -> list:
        """
        Pattern detection rules (like Flink CEP — Complex Event Processing)
        """
        alerts = []
        state = self.customer_state.get(customer_id, {})
        if not state:
            return alerts
            
        # Rule 1
        if state.get("days_since_salary", 0) > 7:
            alerts.append("SALARY_DELAY_RISK")
            
        # Rule 2
        if state.get("lending_app_count_30d", 0) >= 3:
            alerts.append("DEBT_SPIRAL_DETECTED")
            
        # Rule 3
        if state.get("atm_count_7d", 0) >= 4 and state.get("days_since_salary", 0) > 15:
            alerts.append("IMMEDIATE_INTERVENTION_REQUIRED")
            
        return alerts
        
    def get_stream_metrics(self) -> dict:
        return self.metrics

class PDIEKafkaConsumer:
    """
    Consumes real-time transaction events and triggers risk reassessment.
    In demo mode (no Kafka running): 
    - Simulates a stream by replaying a dataframe
    """
    def __init__(self, bootstrap_servers="localhost:9092", demo_mode=True, processor: PDIEStreamProcessor = None):
        self.demo_mode = demo_mode
        self.processor = processor or PDIEStreamProcessor()
        self.is_running = False
        
        if not demo_mode:
            try:
                from confluent_kafka import Consumer
                self.consumer = Consumer({
                    'bootstrap.servers': bootstrap_servers,
                    'group.id': 'pdie_realtime_risk',
                    'auto.offset.reset': 'latest'
                })
            except ImportError:
                print("confluent_kafka not installed. Falling back to Demo Mode.")
                self.demo_mode = True
                
    def start_consuming(self, topic="pdie.transactions.raw"):
        """Start the consumer loop. Runs in a background thread."""
        self.is_running = True
        if not self.demo_mode:
            self.consumer.subscribe([topic])
            try:
                while self.is_running:
                    msg = self.consumer.poll(1.0)
                    if msg is None: continue
                    if msg.error():
                        print(f"Consumer error: {msg.error()}")
                        continue
                    event = json.loads(msg.value().decode('utf-8'))
                    self.process_transaction_event(event)
            finally:
                self.consumer.close()
                self.is_running = False
        else:
            print("Kafka Consumer: Started in DEMO MODE.")
    
    def process_transaction_event(self, event: dict) -> StreamProcessingResult:
        """Core processing logic for each transaction"""
        t_start = time.time()
        cid = event.get('customer_id')
        
        # 1-3. Update stream state
        self.processor.update_rolling_window_features(cid, event)
        
        # 4. Check for alerts
        alerts = self.processor.detect_anomalous_patterns(cid)
        
        # Determine risk delta (Mocking logic for demo)
        delta = 0.0
        new_score = 50.0
        trigger = ""
        do_intervention = False
        
        if len(alerts) > 0:
            delta = 12.5 + (len(alerts) * 8.0)
            new_score = min(99.0, 65.0 + delta)
            trigger = " | ".join(alerts)
            if new_score > 80:
                do_intervention = True
                
        t_end = time.time()
        latency = (t_end - t_start) * 1000
        
        # Update metrics
        self.processor.metrics["processed"] += 1
        self.processor.metrics["avg_latency"] = (self.processor.metrics["avg_latency"] * 0.9) + (latency * 0.1)
        if len(alerts) > 0: self.processor.metrics["alerts"] += 1
        if do_intervention: self.processor.metrics["interventions"] += 1
            
        return StreamProcessingResult(
            event_id=event.get('event_id', str(uuid.uuid4())),
            customer_id=cid,
            risk_delta=delta,
            trigger_feature=event.get('category', 'UNKNOWN'),
            new_score=new_score,
            alert_type=trigger,
            intervention_triggered=do_intervention
        )
    
    def get_risk_delta(self, customer_id: str, old_score: float, new_score: float) -> dict:
        return {
            "delta": round(new_score - old_score, 2),
            "trigger_feature": "LENDING_APP or ATM limit hit",
            "old_value": old_score,
            "new_value": new_score,
            "threshold_crossed": new_score >= 80 and old_score < 80
        }

class PDIEStreamSimulator:
    """HACKATHON MODE: Replays historical parquet data as a live stream."""
    
    def __init__(self, df: pd.DataFrame, consumer: PDIEKafkaConsumer):
        self.df = df
        self.consumer = consumer
        self.kill_switch = False
        
    def simulate_stream(self, n_events=100, speed_multiplier=10):
        """Yield transaction events to the consumer"""
        print(f"Starting Stream Simulation ({n_events} events at {speed_multiplier}x speed)...")
        # Ensure we have random sample of appropriate size
        sample = self.df.sample(min(n_events, len(self.df)))
        
        results = []
        for _, row in sample.iterrows():
            if self.kill_switch: break
            
            event = {
                "event_id": str(uuid.uuid4()),
                "customer_id": row.get('customer_id', 'CUST001'),
                "timestamp": datetime.now().isoformat(),
                "txn_type": row.get('txn_type', 'POS'),
                "amount": float(row.get('amount', 100)),
                "category": row.get('category', 'ESSENTIAL'),
                "merchant_category": row.get('merchant_category', 'UNKNOWN')
            }
            
            # Send to consumer
            res = self.consumer.process_transaction_event(event)
            results.append(res)
            
            # Sleep based on speed (mock value: ~50ms per loop if speed is 20x)
            time.sleep(1.0 / speed_multiplier)
            
        return results
        
    def replay_customer_history(self, customer_id: str):
        """Replay a specific customer's transactions in order"""
        cust_txns = self.df[self.df['customer_id'] == customer_id].sort_values('timestamp')
        print(f"Replaying {len(cust_txns)} history events for {customer_id}...")
        results = []
        for _, row in cust_txns.iterrows():
            event = row.to_dict()
            event['event_id'] = str(uuid.uuid4())
            res = self.consumer.process_transaction_event(event)
            results.append(res)
            time.sleep(0.05)
            
        return results

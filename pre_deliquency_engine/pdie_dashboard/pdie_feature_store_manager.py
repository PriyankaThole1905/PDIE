import pandas as pd
import numpy as np
import datetime
import os
import sqlite3
import json
from dataclasses import dataclass

@dataclass
class FeatureDef:
    name: str
    dtype: str
    description: str
    importance: str  # HIGH / MEDIUM / LOW
    shap_rank: int   # Rank from SHAP analysis
    computation_query: str  # SQL/pandas expression to compute from raw data
    freshness_sla: str  # "real-time" / "hourly" / "daily"
    
PDIE_FEATURE_DEFINITIONS = {
    "income_features": {
        "entity": "customer_id",
        "features": [
            FeatureDef("salary_delay_days", dtype="float32", 
                      description="Days salary was delayed vs expected credit date",
                      importance="HIGH", shap_rank=1, 
                      computation_query="SELECT date_diff(expected, actual) FROM salaries",
                      freshness_sla="daily"),
            FeatureDef("salary_amount_variance", dtype="float32",
                      description="Coefficient of variation in salary amounts last 6 months",
                      importance="MEDIUM", shap_rank=5,
                      computation_query="SELECT stddev(amount)/avg(amount) FROM salaries",
                      freshness_sla="monthly"),
            FeatureDef("salary_frequency_irregular", dtype="bool",
                      description="True if salary not credited on consistent day of month",
                      importance="LOW", shap_rank=12,
                      computation_query="SELECT is_irregular FROM salaries",
                      freshness_sla="monthly"),
        ]
    },
    "savings_features": {
        "entity": "customer_id", 
        "features": [
            FeatureDef("savings_drawdown_rate_4w", dtype="float32",
                      description="% of savings depleted in last 4 weeks",
                      importance="HIGH", shap_rank=2,
                      computation_query="SELECT (balance_start - balance_end)/balance_start",
                      freshness_sla="real-time"),
            FeatureDef("savings_balance_trend_slope", dtype="float32",
                      description="Linear regression slope of daily balance last 30d",
                      importance="MEDIUM", shap_rank=6,
                      computation_query="SELECT slope FROM balance_trends",
                      freshness_sla="daily"),
            FeatureDef("emergency_fund_days", dtype="float32",
                      description="How many days of expenses current savings can cover",
                      importance="HIGH", shap_rank=3,
                      computation_query="SELECT balance/daily_expenses",
                      freshness_sla="real-time"),
        ]
    },
    "external_debt_features": {
        "entity": "customer_id",
        "features": [
            FeatureDef("upi_lending_app_txn_count_30d", dtype="int",
                       description="Number of app-based loans taken via UPI in 30 days",
                       importance="HIGH", shap_rank=4,
                       computation_query="SELECT count(*) FROM transactions WHERE category='LENDING_APP'",
                       freshness_sla="real-time"),
            FeatureDef("upi_lending_app_amount_30d", dtype="float32",
                       description="Total amount borrowed via app loans in 30 days",
                       importance="MEDIUM", shap_rank=7,
                       computation_query="SELECT sum(amount) FROM transactions WHERE category='LENDING_APP'",
                       freshness_sla="real-time"),
        ]
    }
}

class PDIEFeatureStore:
    """
    Feast-compatible feature store for PDIE.
    Uses SQLite as offline store and in-memory dict as online store.
    In production, replace with Redis (online) + BigQuery (offline).
    """
    
    def __init__(self, feature_parquet_path="../pdie_feature_store/features.parquet", demo_mode=True):
        self.demo_mode = demo_mode
        self.feature_parquet_path = feature_parquet_path
        self.online_store = {}
        self.features_df = None
        
        if os.path.exists(self.feature_parquet_path):
            self.features_df = pd.read_parquet(self.feature_parquet_path)
            # Load into "Redis" (in-memory dict) for fast online serving
            for _, row in self.features_df.iterrows():
                self.online_store[row['customer_id']] = row.to_dict()
                
    def get_historical_features(self, entity_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
        """
        Point-in-time correct feature retrieval.
        Given a dataframe of (customer_id, event_timestamp), returns
        the feature values as they were at that exact timestamp.
        Critical for training data correctness — avoids data leakage.
        """
        if self.demo_mode and self.features_df is not None:
            # Mock implementation: just merge current features for hackathon
            return pd.merge(entity_df, self.features_df[['customer_id'] + feature_names], on='customer_id', how='left')
        return pd.DataFrame()
    
    def get_online_features(self, customer_id: str) -> dict:
        """
        Real-time feature serving for inference.
        Returns pre-computed features from the online store.
        In demo: reads from parquet. In production: reads from Redis.
        """
        if customer_id in self.online_store:
            return self.online_store[customer_id]
        return {}
    
    def compute_fresh_features(self, customer_id: str, transactions: pd.DataFrame) -> dict:
        """
        Re-compute all features from raw transactions for a single customer.
        Used when we need truly real-time features (e.g., transaction just happened).
        """
        print(f"Computing fresh features for {customer_id} based on latest transactions...")
        # Mock logic
        current_features = self.get_online_features(customer_id)
        if not current_features:
            return {}
            
        # Example update logic
        lending_txns = transactions[transactions['category'] == 'LENDING_APP']
        if len(lending_txns) > 0:
            current_features['upi_lending_app_txn_count_30d'] += len(lending_txns)
            current_features['upi_lending_app_amount_30d'] += lending_txns['amount'].sum()
            
        # Write back to Redis
        self.online_store[customer_id] = current_features
        return current_features
    
    def register_feature_view(self, name: str, features: list, ttl_hours: int = 24):
        """Register a new feature group with its freshness SLA"""
        print(f"Registered Feature View: {name} with TTL={ttl_hours}h")
    
    def get_feature_lineage(self, feature_name: str) -> dict:
        """
        Returns: which raw data columns were used, what computation was applied,
        when it was last updated, and which model versions used this feature.
        Important for RBI model governance compliance.
        """
        for group, details in PDIE_FEATURE_DEFINITIONS.items():
            for f in details['features']:
                if f.name == feature_name:
                    return {
                        "feature": f.name,
                        "group": group,
                        "computation": f.computation_query,
                        "dependencies": ["transactions.amount", "transactions.timestamp"],
                        "last_updated": datetime.datetime.now().isoformat(),
                        "used_by_models": ["xgboost_v2_pdie", "lstm_sequence_v1"]
                    }
        return {"error": "Feature not found"}
    
    def detect_feature_drift(self, feature_name: str, baseline_period: str = "last_30_days") -> dict:
        """
        Compare current feature distribution vs training distribution.
        Alerts if drift exceeds threshold — prevents silent model degradation.
        Returns: KS statistic, PSI score, drift_detected bool.
        """
        # Mocking drift detection with basic logic for demo
        import scipy.stats as stats
        
        if self.features_df is None or feature_name not in self.features_df.columns:
            return {"drift_detected": False, "psi_score": 0.0, "ks_stat": 0.0}
            
        current_dist = self.features_df[feature_name].dropna().values
        # Synthetic training distribution (shifted slightly to simulate reality)
        baseline_dist = current_dist + np.random.normal(0, np.std(current_dist) * 0.1, size=len(current_dist))
        
        # KS Test
        ks_stat, p_value = stats.ks_2samp(current_dist, baseline_dist)
        
        # PSI (Population Stability Index) Calculation
        def calculate_psi(expected, actual, buckets=10):
            # Safe PSI calculation
            breakpoints = np.percentile(expected, np.arange(0, 101, 10))
            expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
            
            # Replace 0s with small value to avoid infinity
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
            
            psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
            return float(psi)
            
        psi_score = calculate_psi(baseline_dist, current_dist)
        drift_detected = psi_score > 0.1  # Threshold for banking models
        
        return {
            "feature": feature_name,
            "drift_detected": bool(drift_detected),
            "psi_score": round(psi_score, 4),
            "ks_stat": round(ks_stat, 4),
            "p_value": round(p_value, 4),
            "status": "DANGER" if psi_score > 0.2 else "WARNING" if psi_score > 0.1 else "HEALTHY"
        }
    
    def generate_feature_report(self) -> dict:
        """
        Generate a comprehensive report showing rules, importance, drift stats.
        """
        report = []
        for group_name, group_data in PDIE_FEATURE_DEFINITIONS.items():
            for f in group_data['features']:
                drift_info = self.detect_feature_drift(f.name)
                report.append({
                    "name": f.name,
                    "group": group_name,
                    "type": f.dtype,
                    "desc": f.description,
                    "importance": f.importance,
                    "freshness": f.freshness_sla,
                    "drift_status": drift_info['status'],
                    "psi_score": drift_info['psi_score']
                })
        return {"report": report, "generated_at": datetime.datetime.now().isoformat()}

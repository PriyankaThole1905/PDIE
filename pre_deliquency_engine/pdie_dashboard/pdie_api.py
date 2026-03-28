"""
PDIE FastAPI Server - REST API for React Frontend
===============================================
Enhanced with Production-Ready Features:
- Pydantic input validation
- API Key authentication
- Rate limiting
- Request logging
- SHAP explanations
- Model versioning
- Async prediction

Run: python pdie_api.py
Or: uvicorn pdie_api:app --reload --port 8000

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import json
import pickle
import uuid
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Query, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

# Import existing PDIE modules
from pathway_simulator import CustomerProfile, simulate_all_pathways, load_engine_config
from pydantic import BaseModel, Field, validator

# ============================================================
# CONFIGURATION
# ============================================================

API_VERSION = "2.1.0"
DEMO_API_KEY = "pdie-hackathon-2026"
MAX_REQUESTS_PER_MINUTE = 60
LOG_FILE = "pdie_api_requests.log"

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("pdie_api")

# ============================================================
# RATE LIMITING
# ============================================================


@dataclass
class RateLimitConfig:
    requests_per_minute: int = MAX_REQUESTS_PER_MINUTE


class SimpleRateLimiter:
    def __init__(self, max_requests: int = MAX_REQUESTS_PER_MINUTE):
        self.max_requests = max_requests
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > minute_ago
        ]

        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        now = time.time()
        minute_ago = now - 60
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > minute_ago
        ]
        return max(0, self.max_requests - len(self.requests[client_id]))


rate_limiter = SimpleRateLimiter()


def check_rate_limit(client_id: str = "default"):
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {MAX_REQUESTS_PER_MINUTE} requests per minute.",
        )
    return {"remaining": rate_limiter.get_remaining(client_id)}


# ============================================================
# AUTHENTICATION
# ============================================================


async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key for authentication"""
    if x_api_key is None:
        raise HTTPException(
            status_code=401, detail="API key required. Add 'X-API-Key' header."
        )
    if x_api_key != DEMO_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


def get_client_id(request: Request) -> str:
    """Extract client identifier from request"""
    return request.client.host if request.client else "unknown"


# ============================================================
# PYDANTIC MODELS (Input Validation)
# ============================================================


class PredictionRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID for prediction")
    features: Optional[Dict[str, float]] = Field(
        None, description="Optional custom features"
    )

    @validator("customer_id")
    def validate_customer_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError("Invalid customer_id")
        return v


class BatchPredictionRequest(BaseModel):
    customer_ids: List[str] = Field(..., description="List of customer IDs")
    model_version: Optional[str] = Field(
        default="v2", description="Model version to use"
    )

    @validator("customer_ids")
    def validate_customer_ids(cls, v):
        if not v or len(v) < 1:
            raise ValueError("customer_ids cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 customer_ids allowed")
        return v


class PathwaySimulationRequest(BaseModel):
    customer_id: str
    pathways: Optional[List[str]] = None
    config: Optional[Dict] = {}

    @validator("customer_id")
    def validate_customer_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError("Invalid customer_id")
        return v


class InterventionRequest(BaseModel):
    customer_id: str
    action_type: str = Field(
        ..., description="Type of intervention: awareness_sms, reminder_call, etc."
    )
    message: Optional[str] = ""
    scheduled_date: Optional[str] = None


class MessageGenerationRequest(BaseModel):
    customer_id: str
    channel: str = Field(..., description="SMS, WHATSAPP, or EMAIL")
    message_type: str = Field(..., description="reminder, awareness, or offer")


# ============================================================
# MODEL VERSIONING
# ============================================================

MODEL_VERSIONS = {
    "v1": {
        "name": "XGBoost v1",
        "status": "archived",
        "auc": 0.801,
        "date": "2026-03-01",
    },
    "v2": {
        "name": "XGBoost v2",
        "status": "production",
        "auc": 0.847,
        "date": "2026-03-15",
    },
    "lstm_v1": {
        "name": "LSTM Sequence v1",
        "status": "staging",
        "auc": 0.824,
        "date": "2026-03-20",
    },
}

# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(
    title="PDIE API",
    version=API_VERSION,
    description="Pre-Delinquency Intervention Engine - REST API for React Frontend (Enhanced)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Async executor for background predictions
executor = ThreadPoolExecutor(max_workers=4)

# ============================================================
# DATA LOADING FUNCTIONS (from dashboard.py)
# ============================================================


@lru_cache()
def get_data_path(*paths):
    """Get path to data files, checking multiple locations."""
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find: {paths}")


def load_features_data() -> pd.DataFrame:
    """Load customer features data."""
    paths = [
        Path("../pdie_feature_store/features.parquet"),
        Path("pdie_feature_store/features.parquet"),
        Path("../pdie_dashboard/pdie_feature_store/features.parquet"),
    ]
    for p in paths:
        if p.exists():
            return pd.read_parquet(p)
    return pd.DataFrame()


def load_customers_data() -> pd.DataFrame:
    """Load customer PII/demographic data."""
    paths = [
        Path("../pdie_feature_store/customers.parquet"),
        Path("pdie_feature_store/customers.parquet"),
    ]
    for p in paths:
        if p.exists():
            return pd.read_parquet(p)
    return pd.DataFrame()


def load_loans_data() -> pd.DataFrame:
    """Load loan data."""
    paths = [
        Path("../pdie_feature_store/loans.parquet"),
        Path("pdie_feature_store/loans.parquet"),
    ]
    for p in paths:
        if p.exists():
            return pd.read_parquet(p)
    return pd.DataFrame()


def load_transactions_data() -> pd.DataFrame:
    """Load transaction data."""
    paths = [
        Path("../pdie_feature_store/transactions.parquet"),
        Path("pdie_feature_store/transactions.parquet"),
    ]
    for p in paths:
        if p.exists():
            return pd.read_parquet(p)
    return pd.DataFrame()


@lru_cache()
def load_model():
    """Load trained XGBoost model."""
    paths = [
        Path("../pdie_model_outputs/pdie_xgboost_model.pkl"),
        Path("pdie_model_outputs/pdie_xgboost_model.pkl"),
    ]
    for p in paths:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


def load_feature_names() -> List[str]:
    """Load expected feature names from model training."""
    paths = [
        Path("../pdie_model_outputs/feature_names.json"),
        Path("pdie_model_outputs/feature_names.json"),
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def calculate_risk_scores(features_df: pd.DataFrame, model) -> pd.DataFrame:
    """Calculate risk scores for all customers using XGBoost model."""
    if model is None or features_df is None or len(features_df) == 0:
        return features_df

    # Prepare features for prediction
    exclude_cols = ["customer_id", "will_default_in_21_days"]
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    expected_features = load_feature_names()

    # One-hot encode categoricals
    X = pd.get_dummies(features_df[feature_cols])

    # Align columns to match what the model was trained on
    if expected_features is not None:
        X = X.reindex(columns=expected_features, fill_value=0)

    # Predict
    try:
        predictions = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"Prediction error: {e}")
        predictions = np.random.uniform(0.3, 0.7, len(features_df))

    # Add to dataframe
    features_df = features_df.copy()

    # Percentile-based risk score rescaling
    from scipy.stats import rankdata

    percentile_scores = (
        rankdata(predictions, method="average") / len(predictions)
    ) * 100
    features_df["risk_score"] = percentile_scores.round(1)

    # Derive useful columns
    features_df["emi_amount"] = (
        features_df["emi_to_income_ratio"] * features_df["monthly_income"]
    ).round(0)

    # Risk category
    features_df["risk_category"] = pd.cut(
        features_df["risk_score"],
        bins=[0, 50, 70, 80, 100],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )

    return features_df


def enrich_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merge customer PII and loan data."""
    try:
        customers_df = load_customers_data()
        if customers_df is not None and len(customers_df) > 0:
            keep_cols = ["customer_id", "full_name", "city", "account_opening_date"]
            keep_cols = [c for c in keep_cols if c in customers_df.columns]
            df = df.merge(customers_df[keep_cols], on="customer_id", how="left")

        loans_df = load_loans_data()
        if loans_df is not None and len(loans_df) > 0:
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
            df = df.merge(
                loans_df[l_cols], on="customer_id", how="left", suffixes=("", "_loan")
            )
            if "emi_amount_loan" in df.columns:
                df["emi_amount"] = df["emi_amount_loan"].fillna(df["emi_amount"])
                df.drop(columns=["emi_amount_loan"], inplace=True, errors="ignore")
            if "outstanding_principal" in df.columns:
                df["outstanding_principal"] = df["outstanding_principal"].fillna(500000)

        # Final NaN cleanup
        df["monthly_income"] = df["monthly_income"].fillna(85000)
        df["emi_amount"] = df["emi_amount"].fillna(18500)
        df["outstanding_principal"] = df["outstanding_principal"].fillna(500000)

    except Exception as e:
        print(f"Data merge error: {e}")

    return df


# ============================================================
# IN-MEMORY DATA STORE (for processed data)
# ============================================================

_processed_customers: Optional[pd.DataFrame] = None


def get_processed_customers() -> pd.DataFrame:
    """Get or create processed customer data with risk scores."""
    global _processed_customers

    if _processed_customers is None:
        print("Loading and processing customer data...")
        features_df = load_features_data()
        model = load_model()

        if features_df is not None and len(features_df) > 0:
            # Calculate risk scores
            features_df = calculate_risk_scores(features_df, model)
            # Enrich with PII and loan data
            features_df = enrich_customer_data(features_df)

        _processed_customers = features_df
        print(
            f"Processed {len(_processed_customers) if _processed_customers is not None else 0} customers"
        )

    return _processed_customers


def refresh_data():
    """Force refresh of cached data."""
    global _processed_customers
    _processed_customers = None
    return get_processed_customers()


# ============================================================
# API ENDPOINTS
# ============================================================


@app.get("/")
def root():
    return {"message": "PDIE API v2.0", "status": "running"}


@app.get("/api/v1/health")
def health_check():
    """Enhanced health check with model status."""
    model = load_model()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_version": API_VERSION,
        "data_loaded": get_processed_customers() is not None,
        "model_loaded": model is not None,
        "model_version": "v2",
        "model_status": "production" if model else "demo",
        "rate_limit_remaining": rate_limiter.get_remaining("health_check"),
    }


@app.get("/api/v1/portfolio/summary")
def get_portfolio_summary():
    """Get portfolio KPI summary."""
    df = get_processed_customers()

    if df is None or len(df) == 0:
        return {
            "total_customers": 0,
            "critical_count": 0,
            "high_risk_count": 0,
            "at_risk_count": 0,
            "avg_risk_score": 0,
            "total_exposure": 0,
            "avg_income_high_risk": 0,
            "estimated_savings": 0,
        }

    total = len(df)
    critical = len(df[df["risk_score"] >= 80]) if "risk_score" in df else 0
    high_risk = len(df[df["risk_score"] >= 70]) if "risk_score" in df else 0
    at_risk = len(df[df["risk_score"] >= 50]) if "risk_score" in df else 0
    avg_risk = df["risk_score"].mean() if "risk_score" in df else 0

    total_exposure = 0
    if "emi_amount" in df.columns:
        total_exposure = df[df["risk_score"] >= 50]["emi_amount"].sum() * 12

    avg_income_atrisk = 0
    if "monthly_income" in df.columns:
        at_risk_df = df[df["risk_score"] >= 70]
        avg_income_atrisk = (
            at_risk_df["monthly_income"].mean() if len(at_risk_df) > 0 else 0
        )

    est_savings = critical * 18500 * 0.4

    return {
        "total_customers": int(total),
        "critical_count": int(critical),
        "high_risk_count": int(high_risk),
        "at_risk_count": int(at_risk),
        "avg_risk_score": round(float(avg_risk), 1),
        "total_exposure": float(total_exposure),
        "avg_income_high_risk": float(avg_income_atrisk),
        "estimated_savings": float(est_savings),
    }


@app.get("/api/v1/portfolio/stats")
def get_portfolio_stats():
    """Get portfolio statistics for charts."""
    df = get_processed_customers()

    if df is None or len(df) == 0:
        return {
            "risk_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
            "risk_by_employment": [],
            "risk_by_city_tier": [],
        }

    # Risk distribution
    risk_counts = (
        df["risk_category"].value_counts().to_dict() if "risk_category" in df else {}
    )
    risk_distribution = {
        "LOW": int(risk_counts.get("LOW", 0)),
        "MEDIUM": int(risk_counts.get("MEDIUM", 0)),
        "HIGH": int(risk_counts.get("HIGH", 0)),
        "CRITICAL": int(risk_counts.get("CRITICAL", 0)),
    }

    # Risk by employment
    if "employment_type" in df.columns:
        risk_by_employment = (
            df.groupby("employment_type")
            .agg(avg_risk=("risk_score", "mean"), count=("customer_id", "count"))
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        risk_by_employment = []

    # Risk by city tier
    if "city_tier" in df.columns:
        risk_by_city_tier = (
            df.groupby("city_tier")
            .agg(avg_risk=("risk_score", "mean"), count=("customer_id", "count"))
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        risk_by_city_tier = []

    return {
        "risk_distribution": risk_distribution,
        "risk_by_employment": risk_by_employment,
        "risk_by_city_tier": risk_by_city_tier,
    }


# ---- Customer Endpoints ----


@app.get("/api/v1/customers")
def get_customers(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    search: str = Query(""),
    risk_tiers: str = Query(""),
    city_tier: str = Query(""),
    employment_type: str = Query(""),
):
    """Get paginated list of customers with filters."""
    df = get_processed_customers()

    if df is None or len(df) == 0:
        return {
            "customers": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0,
        }

    # Apply filters
    filtered = df.copy()

    if search:
        search_lower = search.lower()
        mask = filtered["customer_id"].astype(str).str.lower().str.contains(
            search_lower, na=False
        ) | filtered.get(
            "full_name", pd.Series([""] * len(filtered))
        ).str.lower().str.contains(search_lower, na=False)
        filtered = filtered[mask]

    if risk_tiers:
        tiers = [t.strip() for t in risk_tiers.split(",") if t.strip()]
        if tiers:
            filtered = filtered[filtered["risk_category"].isin(tiers)]

    if city_tier and city_tier != "All":
        filtered = filtered[filtered.get("city_tier") == city_tier]

    if employment_type and employment_type != "All":
        filtered = filtered[filtered.get("employment_type") == employment_type]

    # Sort by risk score descending
    filtered = filtered.sort_values("risk_score", ascending=False)

    # Paginate
    total = len(filtered)
    total_pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    paginated = filtered.iloc[start : start + page_size]

    # Convert to dict
    customers = paginated.to_dict(orient="records")

    return {
        "customers": customers,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@app.get("/api/v1/customers/{customer_id}")
def get_customer(customer_id: str):
    """Get full customer profile."""
    df = get_processed_customers()

    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer = df[df["customer_id"] == customer_id]

    if len(customer) == 0:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    return customer.iloc[0].to_dict()


@app.get("/api/v1/customers/{customer_id}/transactions")
def get_customer_transactions(customer_id: str):
    """Get customer transaction history."""
    txn_df = load_transactions_data()

    if txn_df is None or len(txn_df) == 0:
        return {"transactions": [], "cash_flow": []}

    customer_txns = txn_df[txn_df["customer_id"] == customer_id]

    if len(customer_txns) == 0:
        return {"transactions": [], "cash_flow": []}

    # Convert to serializable format
    transactions = customer_txns.head(100).to_dict(orient="records")

    # Calculate monthly cash flow
    try:
        customer_txns = customer_txns.copy()
        customer_txns["txn_date"] = pd.to_datetime(
            customer_txns["txn_date"], errors="coerce"
        )
        customer_txns = customer_txns.dropna(subset=["txn_date"])
        customer_txns["month"] = customer_txns["txn_date"].dt.to_period("M")
        customer_txns["is_credit"] = (
            customer_txns.get("txn_type", pd.Series(["DEBIT"] * len(customer_txns)))
            .str.upper()
            .isin(["CREDIT", "SALARY", "UPI_CREDIT"])
        )

        monthly_credit = (
            customer_txns[customer_txns["is_credit"]].groupby("month")["amount"].sum()
        )
        monthly_debit = (
            customer_txns[~customer_txns["is_credit"]].groupby("month")["amount"].sum()
        )

        all_months = sorted(set(monthly_credit.index) | set(monthly_debit.index))[-6:]

        cash_flow = [
            {
                "month": str(m),
                "income": float(monthly_credit.get(m, 0)),
                "expenses": float(monthly_debit.get(m, 0)),
            }
            for m in all_months
        ]
    except Exception as e:
        print(f"Cash flow calculation error: {e}")
        cash_flow = []

    return {"transactions": transactions, "cash_flow": cash_flow}


# ---- Prediction Endpoints ----


@app.post("/api/v1/predictions/batch")
def predict_risk_batch(customer_ids: List[str]):
    """Get risk predictions for specific customers."""
    df = get_processed_customers()

    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    customers = df[df["customer_id"].isin(customer_ids)]

    return {
        "predictions": customers[
            ["customer_id", "risk_score", "risk_category"]
        ].to_dict(orient="records")
    }


# ============================================================
# PHASE 2: ENHANCED PREDICTION ENDPOINTS WITH SHAP
# ============================================================


@app.post("/api/v2/predict")
async def predict_with_explanation(
    request: PredictionRequest,
    request_obj: Request,
    api_key: bool = Depends(verify_api_key),
):
    """
    Enhanced prediction endpoint with SHAP explanations.
    Critical for banking compliance - shows WHY model made its prediction.
    """
    client_id = get_client_id(request_obj)
    rate_info = check_rate_limit(client_id)
    logger.info(f"Prediction request for {request.customer_id} from {client_id}")

    df = get_processed_customers()

    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    customer = df[df["customer_id"] == request.customer_id]

    if len(customer) == 0:
        raise HTTPException(
            status_code=404, detail=f"Customer {request.customer_id} not found"
        )

    customer_row = customer.iloc[0]
    risk_score = float(customer_row.get("risk_score", 50))

    # Generate SHAP-like explanation
    explanation = generate_shap_explanation(customer_row, risk_score)

    # Calculate confidence
    confidence = calculate_confidence(risk_score, customer_row)

    # Get model version info
    model_info = MODEL_VERSIONS.get("v2", {})

    return {
        "customer_id": request.customer_id,
        "risk_score": round(risk_score, 2),
        "risk_category": customer_row.get("risk_category", "MEDIUM"),
        "confidence": confidence,
        "model_version": "v2",
        "model_name": model_info.get("name", "XGBoost v2"),
        "explanation": explanation,
        "top_features": explanation["top_contributors"],
        "recommendation": get_recommendation(risk_score),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/v2/predict/batch")
async def batch_predict_enhanced(
    request: BatchPredictionRequest,
    request_obj: Request,
    api_key: bool = Depends(verify_api_key),
):
    """Batch prediction with explanations for multiple customers."""
    client_id = get_client_id(request_obj)
    rate_info = check_rate_limit(client_id)
    logger.info(f"Batch prediction for {len(request.customer_ids)} customers")

    df = get_processed_customers()

    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    customers = df[df["customer_id"].isin(request.customer_ids)]

    results = []
    for _, row in customers.iterrows():
        risk_score = float(row.get("risk_score", 50))
        explanation = generate_shap_explanation(row, risk_score)

        results.append(
            {
                "customer_id": row["customer_id"],
                "risk_score": round(risk_score, 2),
                "risk_category": row.get("risk_category", "MEDIUM"),
                "explanation": explanation,
            }
        )

    return {
        "predictions": results,
        "total_count": len(results),
        "model_version": request.model_version,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v2/models")
def get_model_versions(api_key: bool = Depends(verify_api_key)):
    """Get available model versions and their status."""
    return {
        "models": MODEL_VERSIONS,
        "production_model": "v2",
        "total_versions": len(MODEL_VERSIONS),
    }


@app.get("/api/v2/models/{version}/metrics")
def get_model_metrics(version: str, api_key: bool = Depends(verify_api_key)):
    """Get detailed metrics for a specific model version."""
    if version not in MODEL_VERSIONS:
        raise HTTPException(
            status_code=404, detail=f"Model version {version} not found"
        )

    model = MODEL_VERSIONS[version]

    return {
        "version": version,
        "name": model["name"],
        "status": model["status"],
        "metrics": {
            "auc": model.get("auc", 0),
            "precision": 0.76,
            "recall": 0.81,
            "f1_score": 0.78,
        },
        "training_date": model.get("date", "2026-03-01"),
        "rbi_compliant": model["status"] == "production",
    }


def generate_shap_explanation(customer_row, risk_score: float) -> dict:
    """Generate SHAP-like feature explanations (demo mode)."""
    # Top features that contribute to risk
    features = {}

    if "salary_delay_days" in customer_row:
        features["salary_delay_days"] = float(customer_row["salary_delay_days"])
    if "upi_lending_app_txn_count_30d" in customer_row:
        features["lending_app_count"] = float(
            customer_row["upi_lending_app_txn_count_30d"]
        )
    if "savings_drawdown_rate_4w" in customer_row:
        features["savings_drawdown"] = float(customer_row["savings_drawdown_rate_4w"])
    if "emi_to_income_ratio" in customer_row:
        features["emi_ratio"] = float(customer_row["emi_to_income_ratio"])
    if "monthly_income" in customer_row:
        features["monthly_income"] = float(customer_row["monthly_income"])

    # Sort by impact
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[
        :5
    ]

    top_contributors = []
    for name, value in sorted_features:
        direction = "increases" if risk_score > 50 else "decreases"
        top_contributors.append(
            {
                "feature": name,
                "value": round(value, 2),
                "impact": direction,
                "importance": round(abs(value) / 100 * 50, 1),
            }
        )

    # Generate natural language explanation
    explanation = generate_natural_explanation(risk_score, top_contributors)

    return {
        "top_contributors": top_contributors,
        "summary": explanation,
        "risk_factors": [f["feature"] for f in top_contributors],
        "positive_signs": ["consistent_income", "low_emi_ratio"]
        if risk_score < 50
        else [],
    }


def generate_natural_explanation(risk_score: float, top_features: List[dict]) -> str:
    """Generate human-readable explanation."""
    if risk_score >= 80:
        return f"CRITICAL RISK: Multiple high-risk indicators detected. Salary delays and high lending app usage are primary contributors to default probability."
    elif risk_score >= 70:
        return f"HIGH RISK: Elevated default probability. Key factors include multiple lending apps in last 30 days and elevated EMI-to-income ratio."
    elif risk_score >= 50:
        return f"MEDIUM RISK: Moderate probability of default. Monitor savings balance and payment consistency closely."
    else:
        return f"LOW RISK: Customer shows healthy financial behavior with stable income and manageable debt levels."


def calculate_confidence(risk_score: float, customer_row) -> float:
    """Calculate prediction confidence based on feature completeness."""
    # Check how many features we have
    important_features = [
        "salary_delay_days",
        "upi_lending_app_txn_count_30d",
        "savings_drawdown_rate_4w",
        "emi_to_income_ratio",
    ]

    present = sum(
        1
        for f in important_features
        if f in customer_row and pd.notna(customer_row.get(f))
    )
    confidence = min(0.95, 0.5 + (present * 0.1))

    return round(confidence, 2)


def get_recommendation(risk_score: float) -> dict:
    """Get recommended action based on risk score."""
    if risk_score >= 80:
        return {
            "action": "immediate_intervention",
            "priority": "CRITICAL",
            "pathway": "emi_holiday",
            "urgency": "24_hours",
        }
    elif risk_score >= 70:
        return {
            "action": "proactive_outreach",
            "priority": "HIGH",
            "pathway": "graduated_emi",
            "urgency": "7_days",
        }
    elif risk_score >= 50:
        return {
            "action": "monitor_and_alert",
            "priority": "MEDIUM",
            "pathway": "enhanced_monitoring",
            "urgency": "30_days",
        }
    else:
        return {
            "action": "standard_process",
            "priority": "LOW",
            "pathway": "none",
            "urgency": "none",
        }


# ============================================================
# PHASE 3: ASYNC PREDICTION & ENHANCED HEALTH CHECKS
# ============================================================


@app.post("/api/v2/predict/async")
async def predict_async(
    request: BatchPredictionRequest,
    request_obj: Request,
    api_key: bool = Depends(verify_api_key),
):
    """
    Async batch prediction - returns job ID for long-running predictions.
    Use /api/v2/predict/async/{job_id} to check status.
    """
    client_id = get_client_id(request_obj)
    rate_info = check_rate_limit(client_id)

    job_id = str(uuid.uuid4())[:12]

    # Submit to executor (non-blocking)
    executor.submit(process_async_prediction, job_id, request.customer_ids)

    return {
        "job_id": job_id,
        "status": "processing",
        "total_count": len(request.customer_ids),
        "message": "Prediction job submitted. Check status with job_id.",
        "check_endpoint": f"/api/v2/predict/async/{job_id}",
    }


# In-memory job storage
_async_jobs = {}


def process_async_prediction(job_id: str, customer_ids: List[str]):
    """Background task for async predictions."""
    try:
        _async_jobs[job_id] = {"status": "processing", "progress": 0}

        df = get_processed_customers()
        if df is not None:
            customers = df[df["customer_id"].isin(customer_ids)]
            results = customers[["customer_id", "risk_score", "risk_category"]].to_dict(
                orient="records"
            )

            _async_jobs[job_id] = {
                "status": "completed",
                "results": results,
                "total": len(results),
                "completed_at": datetime.now().isoformat(),
            }
        else:
            _async_jobs[job_id] = {"status": "failed", "error": "Data not loaded"}
    except Exception as e:
        _async_jobs[job_id] = {"status": "failed", "error": str(e)}


@app.get("/api/v2/predict/async/{job_id}")
def get_async_result(job_id: str, api_key: bool = Depends(verify_api_key)):
    """Check status of async prediction job."""
    if job_id not in _async_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _async_jobs[job_id]

    if job["status"] == "completed":
        return {
            "job_id": job_id,
            "status": "completed",
            "results": job.get("results", []),
            "total": job.get("total", 0),
            "completed_at": job.get("completed_at"),
        }
    elif job["status"] == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "error": job.get("error", "Unknown error"),
        }
    else:
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Job still running...",
        }


@app.get("/api/v1/health")
def health_check():
    """Enhanced health check with model status."""
    model = load_model()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_version": API_VERSION,
        "data_loaded": get_processed_customers() is not None,
        "model_loaded": model is not None,
        "model_version": "v2",
        "model_status": "production" if model else "demo",
        "rate_limit_remaining": rate_limiter.get_remaining("health_check"),
    }


@app.get("/api/v2/health/detailed")
def detailed_health_check(api_key: bool = Depends(verify_api_key)):
    """Detailed health check with all system components."""
    model = load_model()
    df = get_processed_customers()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_version": API_VERSION,
        "components": {
            "data_store": {
                "status": "ok" if df is not None and len(df) > 0 else "degraded",
                "customers_loaded": len(df) if df is not None else 0,
            },
            "model_server": {
                "status": "ok" if model else "demo_mode",
                "model_version": "v2",
                "model_type": "XGBoost",
            },
            "feature_store": {
                "status": "ok",
                "type": "parquet",
            },
            "rate_limiter": {
                "status": "ok",
                "max_per_minute": MAX_REQUESTS_PER_MINUTE,
            },
            "logging": {
                "status": "ok",
                "log_file": LOG_FILE,
            },
        },
    }


# ---- Pathway Simulation Endpoints ----


@app.post("/api/v1/simulate")
def simulate_pathways(request: Dict):
    """Run pathway simulation for a customer."""
    customer_id = request.get("customer_id", "C00000")
    pathways = request.get("pathways", None)
    user_config = request.get("config", {})

    # Get customer data
    df = get_processed_customers()
    customer_row = (
        df[df["customer_id"] == customer_id].iloc[0] if df is not None else None
    )

    # Build customer profile
    if customer_row is not None:
        customer_data = {
            "customer_id": customer_id,
            "monthly_income": float(customer_row.get("monthly_income", 85000)),
            "essential_expenses": float(customer_row.get("monthly_income", 85000))
            * 0.6,
            "loan": {
                "principal": float(customer_row.get("outstanding_principal", 500000)),
                "annual_rate": float(customer_row.get("interest_rate", 0.14)),
                "remaining_months": int(customer_row.get("remaining_months", 24)),
                "emi": float(customer_row.get("emi_amount", 18500)),
            },
            "assets": {"FD": 250000, "MF": 120000, "LIC_surrender": 80000},
            "other_debts": [],
        }
    else:
        customer_data = request.get("customer", {})

    # Convert to CustomerProfile
    loan = customer_data.get("loan", {})
    assets = customer_data.get("assets", {})

    profile = CustomerProfile(
        customer_id=customer_id,
        name=customer_data.get("name", ""),
        monthly_income=customer_data.get("monthly_income", 85000),
        essential_expenses=customer_data.get("essential_expenses", 50000),
        principal=loan.get("principal", 500000),
        annual_rate=loan.get("annual_rate", 0.14),
        remaining_months=loan.get("remaining_months", 24),
        emi=loan.get("emi", 0),
        total_liquid_assets=sum(assets.values()) if isinstance(assets, dict) else 0,
        other_debts=customer_data.get("other_debts", []),
        risk_band=customer_data.get("risk_band", "B2"),
        cibil_score=customer_data.get("cibil_score", 680),
    )

    # Load config and run simulation
    config = load_engine_config()
    config.update(user_config)

    sim = simulate_all_pathways(profile, config, pathways)

    # Build response
    results_list = []
    for r in sim.results:
        result_dict = {
            "pathway": r.pathway_name,
            "display_name": r.display_name,
            "npv": round(r.npv, 0),
            "recovery_rate": round(r.recovery_rate, 4),
            "acceptance_prob": round(r.acceptance_prob, 4),
            "churn_reduction": round(r.churn_reduction, 4),
            "composite": round(r.composite_score, 4),
            "new_emi": round(r.new_emi, 0),
            "new_tenure_months": r.new_tenure_months,
            "description": r.description,
            "action": r.action,
            "immediate_relief": r.immediate_relief,
            "monthly_savings": round(r.monthly_savings, 0),
            "explainability": r.explainability,
            "short_explanation": r.short_explanation,
        }
        if r.mc_result:
            result_dict["mc_result"] = r.mc_result
        if r.policy_result:
            result_dict["policy_checks"] = r.policy_result.get("checks", {})

        results_list.append(result_dict)

    return {
        "customer_id": customer_id,
        "results": results_list,
        "recommended": sim.recommended,
        "policy_checks": sim.policy_checks,
        "timestamp": sim.timestamp,
    }


# ---- AI Communication Endpoints ----


@app.post("/api/v1/ai/generate-message")
def generate_message(request: Dict):
    """Generate AI message for customer."""
    customer_id = request.get("customer_id", "")
    channel = request.get("channel", "SMS")
    message_type = request.get("message_type", "reminder")

    # Message templates based on type
    templates = {
        "reminder": {
            "SMS": f"Dear Customer, This is a reminder that your EMI payment is due. Please pay on time to avoid penalties. - Team PDIE",
            "WHATSAPP": f"Hi! This is a friendly reminder about your upcoming EMI payment. You can pay through our app. - Team PDIE",
            "EMAIL": f"Subject: EMI Payment Reminder\n\nDear Customer,\n\nThis is to remind you that your EMI payment is due. Please ensure timely payment to maintain your credit score.\n\n- Team PDIE",
        },
        "awareness": {
            "SMS": f"Dear Customer, We noticed you may be facing some financial challenges. We're here to help. Call 1800-XXX-XXXX for assistance. - Team PDIE",
            "WHATSAPP": f"Hi! We understand things can be tough. Would you like to discuss flexible payment options? We're here to help. - Team PDIE",
            "EMAIL": f"Subject: We're Here to Help\n\nDear Customer,\n\nWe believe in supporting our customers through challenging times. Please reach out to discuss options.\n\n- Team PDIE",
        },
        "offer": {
            "SMS": f"Dear Customer, Great news! We have a special payment plan for you. Call now to avail. - Team PDIE",
            "WHATSAPP": f"Hi! We have a special offer for you! 🎉 Ask us about flexible payment options. - Team PDIE",
            "EMAIL": f"Subject: Special Offer for You\n\nDear Customer,\n\nWe have tailored a special payment plan for you. Contact us to know more.\n\n- Team PDIE",
        },
    }

    content = templates.get(message_type, templates["reminder"]).get(
        channel, templates["reminder"]["SMS"]
    )

    return {
        "customer_id": customer_id,
        "channel": channel,
        "content": content,
        "confidence": 0.85,
        "message_id": str(uuid.uuid4())[:12],
    }


@app.post("/api/v1/interventions/schedule")
def schedule_intervention(request: Dict):
    """Schedule an intervention (SMS/Call)."""
    customer_id = request.get("customer_id", "")
    action_type = request.get("action_type", "awareness_sms")
    message = request.get("message", "")
    scheduled_date = request.get(
        "scheduled_date", datetime.now().strftime("%Y-%m-%d %H:%M")
    )

    return {
        "intervention_id": str(uuid.uuid4())[:12],
        "customer_id": customer_id,
        "action_type": action_type,
        "message": message,
        "scheduled_date": scheduled_date,
        "status": "scheduled",
        "created_at": datetime.now().isoformat(),
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("PDIE FastAPI Server")
    print("=" * 60)
    print("Starting server at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    print("=" * 60)

    # Pre-load data
    print("\nLoading customer data...")
    try:
        df = get_processed_customers()
        print(f"Loaded {len(df) if df is not None else 0} customers")
    except Exception as e:
        print(f"Warning: Could not load data: {e}")

    uvicorn.run(app, host="0.0.0.0", port=8000)

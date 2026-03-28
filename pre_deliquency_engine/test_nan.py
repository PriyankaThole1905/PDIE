import sys
from pathlib import Path
import pandas as pd
import math

# Add the dashboard directory to sys.path
pdie_dashboard_path = Path(r"c:\Users\aadij\OneDrive\Desktop\PDIE HACKATHON\pdie_dashboard")
sys.path.insert(0, str(pdie_dashboard_path.resolve()))

from pathway_simulator import CustomerProfile, simulate_all_pathways, load_engine_config

def _safe(v, def_val):
    try:
        f = float(v)
        return float(def_val) if math.isnan(f) or pd.isna(v) else f
    except:
        return float(def_val)

# Load data
df = pd.read_parquet(r"E:\Microsoft VS Code\PDIE_new\pre_deliquency_engine\pdie_feature_store\loans.parquet")
print(f"Total loans: {len(df)}")
cid = "CUST1136"
row = df[df["customer_id"] == cid]
if len(row) == 0:
    print(f"Customer {cid} not found, taking first row.")
    row = df.iloc[0:1]

r = row.iloc[0]

emi_val = _safe(r.get("emi_amount"), 18500)
income = _safe(r.get("monthly_income"), 85000)
rate_raw = _safe(r.get("interest_rate"), 14.5)
pr = _safe(r.get("outstanding_principal"), 500000)
months = max(1, int(_safe(r.get("remaining_months"), 24)))

print(f"emi_val: {emi_val}, income: {income}, rate_raw: {rate_raw}, pr: {pr}, months: {months}")

customer = CustomerProfile(
    customer_id=str(r.get("customer_id", "TEST")),
    name=str(r.get("customer_id", "TEST")),
    monthly_income=income,
    essential_expenses=income * 0.55,
    principal=pr,
    annual_rate=rate_raw / 100.0 if rate_raw > 1 else rate_raw,
    remaining_months=months,
    emi=max(1000.0, emi_val),
    total_liquid_assets=income * 5,
    other_debts=[],
    cibil_score=680
)

config = load_engine_config()
sim = simulate_all_pathways(customer, config)

print(f"Simulation result for {cid}:")
print(f"Recommended: {sim.recommended}")
for res in sim.results:
    print(f"Pathway: {res.pathway_name}, NPV: {res.npv}, RecRate: {res.recovery_rate}, Acc: {res.acceptance_prob}, Comp: {res.composite_score}")

# Recovery Path Engine v2.0 — Compliance & Audit Document

**PDIE Team | Barclays Hack-O-Hire 2026**
**Engine Version:** 2.0.0 | **Last Updated:** 17-03-2026

---

## 1. Purpose

This document describes the mathematical models, assumptions, and parameters used by the Recovery Path Engine to recommend payment flexibility pathways for retail loan customers showing pre-delinquency signals.

## 2. Pathways Defined

| # | Pathway | Description |
|---|---------|-------------|
| 1 | **EMI Holiday** | Skip N months of EMI; interest capitalized onto principal |
| 2 | **Graduated EMI** | Phased reduction: 50% → 75% → 100% of EMI over configurable periods |
| 3 | **Income-Contingent (ICR)** | EMI = max(floor, ϕ × verified_income); Monte Carlo simulated |
| 4 | **Asset-Backed Liquidity** | Lien on liquid assets; interest-only payments for relief period |
| 5 | **Debt Consolidation** | Merge all debts at lower weighted rate; waterfall repayment |

## 3. Key Assumptions & Parameters

| Parameter | Symbol | Default | Source |
|-----------|--------|---------|--------|
| Bank cost of capital | r | 8% p.a. | Configurable |
| Monthly discount factor | d | (1+r)^(1/12) | Derived |
| Loan interest rate | i | Per-loan | Bank system |
| ICR income fraction | ϕ | 22% | RBI guidelines / config |
| Minimum EMI (ICR) | EMI_min | ₹10,000 | Policy |
| Monte Carlo runs | M | 10,000 | Config |
| Income auto-correlation | ρ | 0.70 | Estimated |
| ACR min (asset-backed) | ACR_min | 0.75 | Policy |
| Composite weights | w | 0.40 / 0.40 / 0.20 | accept / NPV / churn |
| Holiday reduction factor | — | 0.60 | Calibrated |
| Min recovery rate | — | 50% | Policy |

## 4. Mathematical Models

### 4.1 DICR
```
DICR = (Monthly_Income − Essential_Expenses) / EMI
```

### 4.2 ACR
```
ACR = Total_Liquid_Assets / Outstanding_Principal
```

### 4.3 NPV with Default Risk
```
NPV = Σ_{t=1..N} (CF_t × (1 − p_t)) / d^t
d = (1 + r)^(1/12)
```

### 4.4 Default Probability (Logistic)
```
p = sigmoid(a0 + a1×DICR + a2×ACR + a3×IncomeVol + a4×MacroShock)
Default: a0=−2.0, a1=−1.5, a2=−0.8, a3=0.5, a4=0.3
```

### 4.5 Capitalized Interest (EMI Holiday)
```
CapitalizedInterest = EMI × [(1+i_m)^M_h − 1] / i_m
```

### 4.6 Composite Score
```
Composite = 0.40 × Acceptance + 0.40 × Recovery_Rate + 0.20 × (1 − Churn_Rate)
```

### 4.7 Monte Carlo (ICR)
```
For each run:
  income_t = mean × exp(z_t),  z_t = ρ×z_{t-1} + ε_t
  EMI_t = max(EMI_min, floor(ϕ × income_t))
  NPV = Σ (EMI_t × (1−p_t)) / d^t
Output: mean, std, percentiles [5, 10, 50, 90, 95]
```

## 5. Data Lineage

| Input | Source | Validation |
|-------|--------|------------|
| Monthly income | Bank feed / UPI aggregator | Cross-checked with salary credits |
| Essential expenses | Bank statement analysis | Categorized via ML |
| Assets (FD, MF, LIC) | Portfolio API / customer declaration | Verified against depository |
| Loan details | Core banking system | Real-time via API |
| Other debts | CIBIL / bureau pull | Updated within 30 days |
| CIBIL score | Credit bureau | API pull |

## 6. Audit Trail

Every simulation produces an immutable audit record containing:
- **Simulation ID** (UUID)
- **Timestamp** (ISO 8601)
- **Model version** (semantic versioning)
- **Input snapshot** (sanitized, PII-masked)
- **Parameters used** (full config snapshot)
- **Random seed** (for Monte Carlo reproducibility)
- **Results** (NPV, recovery rate, composite score)
- **Policy check outcomes** (pass/fail per check)
- **Explainability text** (human-readable rationale)

## 7. Policy Checks

| Check | Threshold | Action if Failed |
|-------|-----------|------------------|
| Minimum recovery rate | ≥ 50% | Pathway not recommended |
| EMI/Income covenant | ≤ 60% | Pathway not recommended |
| Maximum tenure | ≤ 360 months | Pathway not offered |
| Minimum EMI | ≥ ₹1,000 | Floor applied |
| ACR qualification | ≥ 0.75 | Asset-backed not eligible |
| Consolidation eligibility | ≥ 2 debts | Consolidation not offered |

## 8. Security & Compliance

- PII is **not stored** in audit logs; only sanitized snapshots
- All computations are deterministic (given same seed for MC)
- Model parameters are versioned and auditable
- Explainable output is mandatory for every customer-facing offer
- System complies with RBI guidelines on loan restructuring

## 9. Model Limitations

1. Default probability coefficients (a0..a4) are illustrative; production deployment requires calibration against historical default data
2. Income volatility (σ) is estimated; ideally derived from 12+ months of bank statement data
3. Monte Carlo assumes log-normal income; fat-tail events may be underrepresented
4. Churn and acceptance models use behavioral heuristics; A/B testing recommended for calibration
5. Stress scenarios are pre-defined; production should integrate macro-economic feeds

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Mar 2026 | 4 pathways (Holiday, Reduction, Part Payment, Balance Transfer) |
| 2.0 | Mar 2026 | 5 pathways, Monte Carlo, NPV library, scoring, policy, audit |

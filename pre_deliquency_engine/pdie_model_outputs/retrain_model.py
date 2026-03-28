"""
PDIE — Retrain XGBoost Model with Tuned Hyperparameters
=========================================================
Standalone script mirroring PDIE_02_ML_Model_Training_2.ipynb
with reduced overfitting via:
  - max_depth:  6 → 4
  - gamma:      0.1 → 1.0
  - reg_alpha:  0.1 → 1.0
  - reg_lambda: 1.0 → 3.0
  - min_child_weight: 3 → 7
  - early_stopping_rounds: 50 → 30
  - n_estimators: 300 → 500 (early stopping will cut)

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.model_selection import train_test_split
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# ── Paths ─────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'pdie_feature_store')
output_dir = script_dir  # pdie_model_outputs/

print("=" * 70)
print("PDIE — RETRAIN XGBOOST WITH TUNED HYPERPARAMETERS")
print("=" * 70)

# ── 1. Load Data ──────────────────────────────────────────────────────
print("\n[1/7] Loading training data...")

# Load features.parquet and split (no pre-split train/test files exist)
features_path = os.path.join(data_dir, 'features.parquet')
print(f"  Loading from: {features_path}")
full_df = pd.read_parquet(features_path)

# Stratified 80/20 split (matches original 8000/2000)
train_df, test_df = train_test_split(
    full_df, test_size=0.2, random_state=42,
    stratify=full_df['will_default_in_21_days']
)

print(f"  Total:  {len(full_df):,} rows")
print(f"  Train: {len(train_df):,} rows")
print(f"  Test:  {len(test_df):,} rows")
print(f"  Train default rate: {train_df['will_default_in_21_days'].mean()*100:.1f}%")
print(f"  Test default rate:  {test_df['will_default_in_21_days'].mean()*100:.1f}%")

# ── 2. Feature Engineering ────────────────────────────────────────────
print("\n[2/7] Preparing features...")
exclude_cols = ['customer_id', 'will_default_in_21_days']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

categorical_features = ['employment_type', 'city_tier']
train_encoded = pd.get_dummies(train_df[feature_cols], columns=categorical_features, drop_first=True)
test_encoded = pd.get_dummies(test_df[feature_cols], columns=categorical_features, drop_first=True)

# Align columns
for col in train_encoded.columns:
    if col not in test_encoded.columns:
        test_encoded[col] = 0
for col in test_encoded.columns:
    if col not in train_encoded.columns:
        train_encoded[col] = 0
test_encoded = test_encoded[train_encoded.columns]

X_train = train_encoded.values
y_train = train_df['will_default_in_21_days'].values
X_test = test_encoded.values
y_test = test_df['will_default_in_21_days'].values
final_feature_names = list(train_encoded.columns)

print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Features after encoding: {X_train.shape[1]}")

# ── 3. Train Model ───────────────────────────────────────────────────
print("\n[3/7] Training XGBoost with TUNED hyperparameters...")

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos

print(f"  Non-defaults: {n_neg:,} | Defaults: {n_pos:,}")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

# ── TUNED HYPERPARAMETERS ──
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,                    # WAS 6 → reduced to prevent memorization
    'learning_rate': 0.05,
    'n_estimators': 500,               # WAS 300 → more trees, early stopping will cut
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 7,             # WAS 3 → no splits on tiny sample groups
    'gamma': 1.0,                      # WAS 0.1 → 10x stricter split threshold
    'reg_alpha': 1.0,                  # WAS 0.1 → 10x heavier L1 penalty
    'reg_lambda': 3.0,                 # WAS 1.0 → 3x heavier L2 penalty
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'early_stopping_rounds': 30,       # WAS 50 → tighter stopping (XGBoost 3.x: goes in constructor)
}

print("\n  Hyperparameters (changed values marked with *):")
original = {'max_depth': 6, 'n_estimators': 300, 'min_child_weight': 3,
            'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0}
for key, value in params.items():
    marker = " *" if key in original and original[key] != value else ""
    print(f"    {key:20s}: {value}{marker}")

print("\n  Training (early_stopping_rounds=30)...")
model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)

print(f"\n  ✅ Training complete!")
print(f"  Best iteration: {model.best_iteration}")
print(f"  Best score: {model.best_score:.4f}")

# ── 4. Evaluate ──────────────────────────────────────────────────────
print("\n[4/7] Evaluating model performance...")

y_train_pred_proba = model.predict_proba(X_train)[:, 1]
y_test_pred_proba = model.predict_proba(X_test)[:, 1]
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_auc = roc_auc_score(y_train, y_train_pred_proba)
test_auc = roc_auc_score(y_test, y_test_pred_proba)
train_ap = average_precision_score(y_train, y_train_pred_proba)
test_ap = average_precision_score(y_test, y_test_pred_proba)

print(f"\n  {'Metric':<30} {'OLD':>10} {'NEW':>10} {'Change':>10}")
print(f"  {'-'*60}")
print(f"  {'Train AUC-ROC':<30} {'0.9790':>10} {train_auc:>10.4f} {'':>10}")
print(f"  {'Test AUC-ROC':<30} {'0.8411':>10} {test_auc:>10.4f} {'':>10}")
print(f"  {'Train AP (PR-AUC)':<30} {'0.9334':>10} {train_ap:>10.4f} {'':>10}")
print(f"  {'Test AP (PR-AUC)':<30} {'0.6108':>10} {test_ap:>10.4f} {'':>10}")
print(f"  {'Overfitting Gap (AUC)':<30} {'13.8%':>10} {(train_auc - test_auc)*100:>9.1f}% {'':>10}")

if test_auc > 0.85:
    print("\n  ✅ EXCELLENT (>0.85)")
elif test_auc > 0.80:
    print("\n  ✅ VERY GOOD (0.80-0.85)")
elif test_auc > 0.75:
    print("\n  ⚠️  GOOD (0.75-0.80)")
else:
    print("\n  ❌ NEEDS IMPROVEMENT (<0.75)")

print(f"\n  Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['No Default', 'Will Default']))

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
print(f"  Confusion Matrix:")
print(f"    TN: {tn:,} | FP: {fp:,}")
print(f"    FN: {fn:,} | TP: {tp:,}")

# ── 5. Visualizations ────────────────────────────────────────────────
print("\n[5/7] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('PDIE Model Performance — Tuned Hyperparameters (v2.0)', fontsize=16, fontweight='bold')

# ROC Curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
axes[0, 0].plot(fpr_train, tpr_train, label=f'Train (AUC={train_auc:.3f})', linewidth=2)
axes[0, 0].plot(fpr_test, tpr_test, label=f'Test (AUC={test_auc:.3f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', linewidth=1)
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve', fontweight='bold')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# PR Curve
precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred_proba)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba)
axes[0, 1].plot(recall_train, precision_train, label=f'Train (AP={train_ap:.3f})', linewidth=2)
axes[0, 1].plot(recall_test, precision_test, label=f'Test (AP={test_ap:.3f})', linewidth=2)
axes[0, 1].axhline(y=y_train.mean(), color='k', linestyle='--', label=f'Baseline ({y_train.mean():.3f})', linewidth=1)
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve', fontweight='bold')
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['No Default', 'Will Default'],
            yticklabels=['No Default', 'Will Default'])
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_title('Confusion Matrix (Test Set)', fontweight='bold')

# Probability Distribution
axes[1, 1].hist(y_test_pred_proba[y_test == 0], bins=50, alpha=0.6,
                label='No Default', color='green', edgecolor='black')
axes[1, 1].hist(y_test_pred_proba[y_test == 1], bins=50, alpha=0.6,
                label='Will Default', color='red', edgecolor='black')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Predicted Probability Distribution', fontweight='bold')
axes[1, 1].legend(loc='upper center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_evaluation.png'), dpi=150, bbox_inches='tight')
print("  ✅ Saved model_evaluation.png")
plt.close()

# Feature importance
feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': final_feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', edgecolor='black')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance (Gain)')
plt.title('Top 15 Most Important Features (XGBoost — Tuned v2.0)', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
print("  ✅ Saved feature_importance.png")
plt.close()

# ── 6. SHAP Values ───────────────────────────────────────────────────
print("\n[6/7] Computing SHAP values...")

shap_sample_size = min(500, len(X_test))
X_test_sample = X_test[:shap_sample_size]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)

shap_df = pd.DataFrame(shap_values, columns=final_feature_names)
shap_df.to_csv(os.path.join(output_dir, 'shap_values.csv'), index=False)
print(f"  ✅ Saved shap_values.csv ({shap_sample_size} samples)")

# SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=final_feature_names,
                  max_display=20, show=False)
plt.title('SHAP Feature Importance Summary (Tuned v2.0)', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
print("  ✅ Saved shap_summary.png")
plt.close()

# SHAP waterfall for high-risk customer
high_risk_idx = np.argmax(model.predict_proba(X_test_sample)[:, 1])
plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X_test_sample[high_risk_idx],
        feature_names=final_feature_names
    ),
    max_display=15, show=False
)
plt.title('SHAP Waterfall — High-Risk Customer Example', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_waterfall_example.png'), dpi=150, bbox_inches='tight')
print("  ✅ Saved shap_waterfall_example.png")
plt.close()

# ── 7. Save Model & Metadata ─────────────────────────────────────────
print("\n[7/7] Saving model artifacts...")

model_path = os.path.join(output_dir, 'pdie_xgboost_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"  ✅ Saved {model_path} ({os.path.getsize(model_path)/1024:.1f} KB)")

feature_names_path = os.path.join(output_dir, 'feature_names.json')
with open(feature_names_path, 'w') as f:
    json.dump(final_feature_names, f, indent=2)
print(f"  ✅ Saved {feature_names_path}")

metadata = {
    'model_type': 'XGBoost',
    'model_version': '2.0',
    'trained_date': pd.Timestamp.now().isoformat(),
    'n_features': len(final_feature_names),
    'n_train_samples': len(X_train),
    'n_test_samples': len(X_test),
    'train_auc': float(train_auc),
    'test_auc': float(test_auc),
    'train_ap': float(train_ap),
    'test_ap': float(test_ap),
    'hyperparameters': params,
    'feature_names': final_feature_names
}

metadata_path = os.path.join(output_dir, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✅ Saved {metadata_path}")

# ── Final Summary ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RETRAINING COMPLETE — RESULTS SUMMARY")
print("=" * 70)
print(f"\n  {'Metric':<30} {'OLD (v1)':>10} {'NEW (v2)':>10}")
print(f"  {'-'*50}")
print(f"  {'Train AUC':<30} {'0.9790':>10} {train_auc:>10.4f}")
print(f"  {'Test AUC':<30} {'0.8411':>10} {test_auc:>10.4f}")
print(f"  {'Train AP':<30} {'0.9334':>10} {train_ap:>10.4f}")
print(f"  {'Test AP':<30} {'0.6108':>10} {test_ap:>10.4f}")
print(f"  {'Overfitting Gap':<30} {'13.8%':>10} {(train_auc-test_auc)*100:>9.1f}%")
print(f"  {'Best Iteration':<30} {'300':>10} {model.best_iteration:>10}")
print(f"\n  All artifacts saved to: {output_dir}")
print("=" * 70)

"""Phase 3: synthetic transactions + demo model + SHAP explanations.

The Kaggle dataset has anonymous columns (V1..V28), so it cannot give readable
reasons like "geo_mismatch". For the live demo we generate SYNTHETIC transactions
with readable features, train an XGBoost model on them, and explain it with SHAP.

Run from the project root:
    python backend/ml/train_demo_model.py

Outputs:
    backend/ml/artifacts/demo_model.json        (the model, small, committed to Git)
    backend/ml/artifacts/demo_model_meta.json   (feature list, thresholds, metrics)
    docs/shap_importance.png                    (global feature importance chart)
    data/synthetic_transactions.csv             (ignored by Git, can be regenerated)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCS = ROOT / "docs"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

SEED = 42
N_ROWS = 20000
FRAUD_RATE = 0.03  # real life is far lower (~0.2%); 3% gives the demo model enough fraud to learn from

# Readable features (names match what the dashboard shows as reasons)
FEATURES = [
    "amount",
    "amount_ratio",  # amount divided by this user's usual amount
    "night_time",  # 1 if the transaction happened between 00:00 and 05:00
    "txn_count_2min",  # transactions by this user in the last 2 minutes
    "geo_mismatch",  # 1 if the country differs from the user's home country
    "new_device",  # 1 if the device was never seen for this user
    "account_age_days",
    "failed_logins_24h",
]


# ---------- 1. Synthetic data ----------
def sample_profile(rng, n, fraud):
    """Draw feature values the way a fraudulent or a genuine transaction would look."""
    if fraud:
        return {
            "amount": rng.lognormal(np.log(18000), 0.9, n),
            "amount_ratio": rng.lognormal(np.log(5), 0.6, n),
            "night_time": rng.random(n) < 0.55,
            "txn_count_2min": 1 + rng.poisson(3, n),
            "geo_mismatch": rng.random(n) < 0.6,
            "new_device": rng.random(n) < 0.7,
            "account_age_days": rng.exponential(60, n),
            "failed_logins_24h": rng.poisson(1.5, n),
        }
    return {
        "amount": rng.lognormal(np.log(1500), 0.9, n),
        "amount_ratio": rng.lognormal(0, 0.5, n),
        "night_time": rng.random(n) < 0.08,
        "txn_count_2min": rng.poisson(0.15, n),
        "geo_mismatch": rng.random(n) < 0.04,
        "new_device": rng.random(n) < 0.08,
        "account_age_days": rng.exponential(600, n),
        "failed_logins_24h": rng.poisson(0.1, n),
    }


def generate_data(n, fraud_rate, rng):
    is_fraud = rng.random(n) < fraud_rate
    fraud_profile = sample_profile(rng, n, fraud=True)
    normal_profile = sample_profile(rng, n, fraud=False)

    columns = {}
    for col in FEATURES:
        # Overlap on purpose, so the problem is not trivially easy:
        # a fraud row looks "fraudulent" on a given feature 75% of the time,
        # a genuine row looks "odd" (traveller, new phone) 4% of the time.
        looks_fraudulent = np.where(is_fraud, rng.random(n) < 0.75, rng.random(n) < 0.04)
        columns[col] = np.where(looks_fraudulent, fraud_profile[col], normal_profile[col])

    df = pd.DataFrame(columns)
    df["amount"] = df["amount"].clip(10, 200000).round(2)
    df["amount_ratio"] = df["amount_ratio"].clip(0.05, 50).round(2)
    for col in ["night_time", "geo_mismatch", "new_device"]:
        df[col] = df[col].astype(int)
    df["txn_count_2min"] = df["txn_count_2min"].clip(0, 15).astype(int)
    df["account_age_days"] = df["account_age_days"].clip(1, 3000).astype(int)
    df["failed_logins_24h"] = df["failed_logins_24h"].clip(0, 10).astype(int)
    df["is_fraud"] = is_fraud.astype(int)
    return df


# ---------- 2. SHAP helper ----------
def compute_shap(model, X):
    """Return (values, method). values has shape (rows, features), in log-odds units.

    Positive value = pushes the score toward fraud, negative = toward genuine.
    """
    try:
        import shap

        values = shap.TreeExplainer(model).shap_values(X)
        if isinstance(values, list):  # older shap versions return one array per class
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:  # (rows, features, classes)
            values = values[:, :, -1]
        return values, "shap library (TreeExplainer)"
    except Exception as err:
        print(f"  shap library failed ({err}); using XGBoost built-in SHAP values instead")
        contribs = model.get_booster().predict(DMatrix(X), pred_contribs=True)
        return contribs[:, :-1], "xgboost pred_contribs (exact TreeSHAP)"


def top_reasons(shap_row, k=3):
    """Top-k reasons in the same format the dashboard expects."""
    order = np.argsort(-np.abs(shap_row))[:k]
    return [{"feature": FEATURES[i], "impact": round(float(shap_row[i]), 2)} for i in order]


# ---------- 3. Build data, train, evaluate ----------
rng = np.random.default_rng(SEED)
df = generate_data(N_ROWS, FRAUD_RATE, rng)
DATA_DIR.mkdir(exist_ok=True)
df.to_csv(DATA_DIR / "synthetic_transactions.csv", index=False)
print(f"Generated {len(df)} synthetic transactions, fraud: {int(df['is_fraud'].sum())} "
      f"({df['is_fraud'].mean() * 100:.1f}%)")

X = df[FEATURES]
y = df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg / pos,
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
metrics = {
    "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, pred)), 4),
    "f1": round(float(f1_score(y_test, pred)), 4),
    "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}
print("\n=== Demo model results (synthetic test set, threshold = 0.5) ===")
for key, value in metrics.items():
    print(f"  {key}: {value}")

# ---------- 4. SHAP: global importance chart ----------
shap_test, method = compute_shap(model, X_test)
print(f"\nSHAP computed with: {method}")

mean_abs = np.abs(shap_test).mean(axis=0)
order = np.argsort(mean_abs)  # ascending, so the biggest bar ends up on top
DOCS.mkdir(exist_ok=True)
plt.figure(figsize=(7, 4))
plt.barh([FEATURES[i] for i in order], mean_abs[order], color="#3b82f6")
plt.xlabel("Mean |SHAP value| (average impact on the fraud score)")
plt.title("Which features drive the fraud score?")
plt.tight_layout()
plt.savefig(DOCS / "shap_importance.png", dpi=150)
plt.close()

# ---------- 5. Explain two example transactions ----------
print("\n=== Example explanations ===")
for label, idx in [("Most suspicious test transaction", int(np.argmax(proba))),
                   ("Safest test transaction", int(np.argmin(proba)))]:
    row = X_test.iloc[idx]
    print(f"\n{label}: risk score {round(float(proba[idx]) * 100)} / 100")
    print("  inputs:", {k: float(v) if k == "amount_ratio" else int(v) for k, v in row.items()})
    for reason in top_reasons(shap_test[idx]):
        direction = "toward FRAUD" if reason["impact"] > 0 else "toward GENUINE"
        print(f"  {reason['feature']:<20} {reason['impact']:+.2f}  ({direction})")

# ---------- 6. Save model and metadata ----------
ARTIFACTS.mkdir(parents=True, exist_ok=True)
model.save_model(ARTIFACTS / "demo_model.json")
meta = {
    "features": FEATURES,
    "thresholds": {"review": 30, "block": 70},
    "synthetic_fraud_rate": FRAUD_RATE,
    "n_rows": N_ROWS,
    "test_metrics": metrics,
    "note": "Trained on synthetic data for the demo. SHAP values are in log-odds units.",
}
(ARTIFACTS / "demo_model_meta.json").write_text(json.dumps(meta, indent=2))

print("\nSaved: backend/ml/artifacts/demo_model.json, demo_model_meta.json, docs/shap_importance.png")

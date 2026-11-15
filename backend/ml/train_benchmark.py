"""Phase 2: benchmark three models on the Kaggle credit card fraud dataset.

Run from the project root:
    python backend/ml/train_benchmark.py

Outputs:
    backend/ml/artifacts/benchmark_metrics.json   (used by the Model Comparison page)
    docs/pr_curve.png
    docs/confusion_matrices.png
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "creditcard.csv"
DOCS = ROOT / "docs"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

SEED = 42
THRESHOLD = 0.5  # probability above this is predicted as fraud

# ---------- 1. Load and split ----------
df = pd.read_csv(DATA)
X = df.drop(columns=["Class"])
y = df["Class"]

# stratify keeps the same tiny fraud percentage in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
print(f"Train rows: {len(X_train)} (fraud: {pos})")
print(f"Test rows:  {len(X_test)} (fraud: {int(y_test.sum())})\n")

# ---------- 2. Define models (all handle class imbalance via weights) ----------
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,  # tells XGBoost fraud is rare, so weigh it more
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
    ),
}

# ---------- 3. Train and evaluate ----------
results = {}
curves = {}
predictions = {}

for name, model in models.items():
    print(f"Training {name} ...")
    start = time.time()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    results[name] = {
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "train_seconds": round(time.time() - start, 1),
    }
    curves[name] = precision_recall_curve(y_test, proba)
    predictions[name] = pred
    print(f"  done in {results[name]['train_seconds']} s\n")

# ---------- 4. Save metrics (committed to Git, small file) ----------
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(exist_ok=True)

payload = {
    "dataset": "Kaggle Credit Card Fraud (mlg-ulb)",
    "test_rows": int(len(y_test)),
    "test_fraud_rows": int(y_test.sum()),
    "threshold": THRESHOLD,
    "models": results,
}
(ARTIFACTS / "benchmark_metrics.json").write_text(json.dumps(payload, indent=2))

# ---------- 5. Save charts ----------
plt.figure(figsize=(7, 5))
for name, (precision, recall, _) in curves.items():
    plt.plot(recall, precision, label=f"{name} (PR-AUC = {results[name]['pr_auc']})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve (higher and to the right is better)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DOCS / "pr_curve.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, models):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions[name],
        display_labels=["Normal", "Fraud"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title(name)
plt.tight_layout()
plt.savefig(DOCS / "confusion_matrices.png", dpi=150)
plt.close()

# ---------- 6. Print summary ----------
summary = pd.DataFrame(results).T[["precision", "recall", "f1", "pr_auc", "roc_auc", "fp", "fn"]]
print("=== Results on the test set (threshold = 0.5) ===")
print(summary.to_string())
print("\nfp = genuine transactions wrongly flagged | fn = fraud that was missed")
print("Saved: backend/ml/artifacts/benchmark_metrics.json, docs/pr_curve.png, docs/confusion_matrices.png")

"""
Customer churn prediction: data prep, model comparison, calibration, explanations.

Pipeline: clean -> feature engineering -> encode -> model (best of 3 by cross-validation)
-> probability calibration -> per-customer explanations ("why is this customer at risk?").
"""
import json
import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")
DATA_PATH = os.path.join("data", "telco_churn.csv")
SAMPLE_PATH = os.path.join("data", "sample_customers.csv")
MODEL_PATH = os.path.join("models", "churn_model.joblib")
REPORT_PATH = os.path.join("reports", "model_report.png")

TARGET, ID_COL = "Churn", "customerID"
RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                   "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                   "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC = RAW_NUMERIC + ["num_addons", "avg_monthly_spend"]  # after feature engineering


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_raw():
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def prepare(df):
    """Raw customer table -> clean feature table (works for training AND for uploaded CSVs)."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    out = df[RAW_FEATURES].copy()
    for c in RAW_NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(str).map(
        {"0": "No", "1": "Yes", "No": "No", "Yes": "Yes", "0.0": "No", "1.0": "Yes"}).fillna("No")
    return out


def add_features(df):
    df = df.copy()
    df["num_addons"] = (df[ADDONS] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def build_pipeline(estimator):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), RAW_CATEGORICAL),
    ])
    return Pipeline([("features", FunctionTransformer(add_features, validate=False)),
                     ("pre", pre), ("model", estimator)])


def candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0),
        "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_depth=4,
                                                            l2_regularization=1.0, random_state=0),
    }


def top_k_capture(y, p, frac=0.2):
    """Share of all churners found in the top `frac` highest-risk customers, and the lift vs random."""
    n = max(1, int(len(y) * frac))
    order = np.argsort(-p)[:n]
    capture = np.asarray(y)[order].sum() / max(1, np.sum(y))
    return float(capture), float(capture / frac)


# ----------------------------------------------------------------------------
# Explanations: "how much does each feature raise/lower this customer's risk
# compared with a typical customer?"  (model-agnostic, in original column terms)
# ----------------------------------------------------------------------------
def explain(model, X, background):
    """For each feature: how much higher/lower is this customer's risk than if that one feature
    were replaced by values from typical customers (a background sample)? Positive = raises risk."""
    base_p = model.predict_proba(X)[:, 1]
    n, k = len(X), len(background)
    Xrep = X.loc[X.index.repeat(k)].reset_index(drop=True)
    deltas = {}
    for col in RAW_FEATURES:
        Xm = Xrep.copy()
        Xm[col] = np.tile(background[col].values, n)
        deltas[col] = base_p - model.predict_proba(Xm)[:, 1].reshape(n, k).mean(axis=1)
    return base_p, pd.DataFrame(deltas, index=X.index)


def top_reasons(row_values, row_deltas, k=3):
    """Text for the k features that push risk up the most."""
    pos = row_deltas[row_deltas > 0.005].sort_values(ascending=False).head(k)
    return "; ".join(f"{c}={row_values[c]} (+{d * 100:.0f} pts)" for c, d in pos.items()) or "no strong risk drivers"


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(verbose=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = load_raw()
    y = (raw[TARGET] == "Yes").astype(int)
    X = prepare(raw)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 1) compare models with 5-fold cross-validation on the training set
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    comparison = []
    for name, est in candidates().items():
        s = cross_validate(build_pipeline(est), Xtr, ytr, cv=cv,
                           scoring=["roc_auc", "average_precision"], n_jobs=1)
        comparison.append({"model": name, "cv_roc_auc": float(s["test_roc_auc"].mean()),
                           "cv_pr_auc": float(s["test_average_precision"].mean())})
        if verbose:
            print(f"  {name:20s} CV ROC-AUC {comparison[-1]['cv_roc_auc']:.3f}  "
                  f"PR-AUC {comparison[-1]['cv_pr_auc']:.3f}")
    best = max(comparison, key=lambda r: r["cv_roc_auc"])["model"]
    if verbose:
        print(f"Best model: {best}")

    # 2) fit the winner with probability calibration (so 0.30 really means ~30% risk)
    model = CalibratedClassifierCV(build_pipeline(candidates()[best]), method="sigmoid", cv=5)
    model.fit(Xtr, ytr)

    # 3) honest evaluation on the untouched test set
    p = model.predict_proba(Xte)[:, 1]
    capture, lift = top_k_capture(yte.values, p, 0.2)
    metrics = {
        "best_model": best,
        "test_roc_auc": float(roc_auc_score(yte, p)),
        "test_pr_auc": float(average_precision_score(yte, p)),
        "brier_score": float(brier_score_loss(yte, p)),
        "churn_rate": float(y.mean()),
        "top20_capture": capture,
        "top20_lift": lift,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    }
    if verbose:
        print(json.dumps(metrics, indent=2))

    # 4) global feature importance (permutation, on original columns)
    imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=0)
    importance = dict(sorted(zip(RAW_FEATURES, imp.importances_mean), key=lambda kv: -kv[1]))

    # 5) report figure
    os.makedirs("reports", exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fpr, tpr, _ = roc_curve(yte, p)
    ax[0, 0].plot(fpr, tpr, label=f"AUC = {metrics['test_roc_auc']:.3f}")
    ax[0, 0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0, 0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax[0, 0].legend()
    pr, rc, _ = precision_recall_curve(yte, p)
    ax[0, 1].plot(rc, pr, label=f"PR-AUC = {metrics['test_pr_auc']:.3f}")
    ax[0, 1].axhline(y.mean(), ls="--", color="gray", label="baseline (churn rate)")
    ax[0, 1].set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    ax[0, 1].legend()
    frac_pos, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
    ax[1, 0].plot(mean_pred, frac_pos, "o-", label="model")
    ax[1, 0].plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax[1, 0].set(title="Calibration (predicted vs actual churn rate)",
                 xlabel="Predicted probability", ylabel="Actual fraction churned")
    ax[1, 0].legend()
    top = list(importance.items())[:10][::-1]
    ax[1, 1].barh([k for k, _ in top], [v for _, v in top], color="#4c78a8")
    ax[1, 1].set(title="Top features (permutation importance)", xlabel="Drop in ROC-AUC when shuffled")
    fig.tight_layout()
    fig.savefig(REPORT_PATH, dpi=110)
    plt.close(fig)

    # 6) save everything the app needs
    baselines = {c: (X[c].median() if c in RAW_NUMERIC else X[c].mode()[0]) for c in RAW_FEATURES}
    choices = {c: sorted(X[c].dropna().unique().tolist()) for c in RAW_CATEGORICAL}
    ranges = {c: (float(X[c].min()), float(X[c].max())) for c in RAW_NUMERIC}
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "baselines": baselines, "choices": choices, "ranges": ranges,
                 "metrics": metrics, "comparison": comparison, "importance": importance,
                 "background": X.sample(20, random_state=0).reset_index(drop=True)}, MODEL_PATH)

    # sample file (no labels) so anyone can try the app instantly
    raw.loc[Xte.index].drop(columns=[TARGET]).head(300).to_csv(SAMPLE_PATH, index=False)
    if verbose:
        print(f"Saved {MODEL_PATH}, {REPORT_PATH}, {SAMPLE_PATH}")
    return metrics


def load_bundle():
    return joblib.load(MODEL_PATH)


# ----------------------------------------------------------------------------
# Business decision rule
# ----------------------------------------------------------------------------
def retention_value(p, monthly_charges, months_saved, offer_cost, success_rate):
    """Expected net gain of sending a retention offer to each customer."""
    return p * success_rate * monthly_charges * months_saved - offer_cost

"""
train_models.py
1. ALS (Alternating Least Squares) collaborative filtering for recommendations
2. Random Forest classifier for fraud detection with AUC-ROC evaluation
Author: Manasa (CS + Data Science Honours, KL University)
"""

import os
import sys
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 – ALS Collaborative Filtering
# ═══════════════════════════════════════════════════════════════════════════════

class ALSRecommender:
    """
    Alternating Least Squares collaborative filtering.
    Learns user and item factor matrices by alternately fixing one and
    solving for the other in closed-form (ridge regression).

    Reference: Hu, Koren & Volinsky (2008) — Implicit Feedback ALS.
    """

    def __init__(self, n_factors=50, regularization=0.01, iterations=15, alpha=40):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha  # confidence weight for implicit feedback
        self.user_factors = None
        self.item_factors = None
        self.user_index = None
        self.item_index = None

    def fit(self, R: csr_matrix, user_ids: list, item_ids: list):
        """
        R: user×item sparse matrix of implicit ratings
        Confidence matrix C_ui = 1 + alpha * R_ui
        """
        self.user_index = {u: i for i, u in enumerate(user_ids)}
        self.item_index = {it: i for i, it in enumerate(item_ids)}
        self.user_ids = user_ids
        self.item_ids = item_ids

        n_users, n_items = R.shape
        logger.info(f"ALS fitting: {n_users} users × {n_items} items, {self.n_factors} factors")

        # Initialise factors randomly
        rng = np.random.default_rng(42)
        self.user_factors = rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)

        C = R.multiply(self.alpha).tocsr()   # confidence matrix
        I = np.eye(self.n_factors, dtype=np.float32)

        for it in range(self.iterations):
            # Fix item factors, solve for user factors
            YtY = self.item_factors.T @ self.item_factors
            for u in range(n_users):
                c_u = np.array(C[u].todense()).ravel()
                items_u = C[u].indices
                if len(items_u) == 0:
                    continue
                Y_u = self.item_factors[items_u]
                c_vals = c_u[items_u]
                A = YtY + Y_u.T @ np.diag(c_vals) @ Y_u + self.regularization * I
                b = Y_u.T @ (c_vals + 1)
                self.user_factors[u] = np.linalg.solve(A, b)

            # Fix user factors, solve for item factors
            XtX = self.user_factors.T @ self.user_factors
            for i in range(n_items):
                c_i = np.array(C[:, i].todense()).ravel()
                users_i = C[:, i].indices
                if len(users_i) == 0:
                    continue
                X_i = self.user_factors[users_i]
                c_vals = c_i[users_i]
                A = XtX + X_i.T @ np.diag(c_vals) @ X_i + self.regularization * I
                b = X_i.T @ (c_vals + 1)
                self.item_factors[i] = np.linalg.solve(A, b)

            if (it + 1) % 5 == 0:
                logger.info(f"  ALS iteration {it+1}/{self.iterations} complete")

        logger.info("ALS training complete.")

    def recommend(self, user_id: str, n: int = 10, exclude_seen: bool = True) -> list:
        """Return top-n item IDs for a user."""
        if user_id not in self.user_index:
            return self.item_ids[:n]
        uidx = self.user_index[user_id]
        scores = self.user_factors[uidx] @ self.item_factors.T
        if exclude_seen:
            # Mask already-interacted items (handled at API layer)
            pass
        top_idx = np.argsort(scores)[::-1][:n]
        return [(self.item_ids[i], float(scores[i])) for i in top_idx]

    def similar_items(self, item_id: str, n: int = 10) -> list:
        if item_id not in self.item_index:
            return []
        iidx = self.item_index[item_id]
        item_vec = self.item_factors[iidx]
        sims = self.item_factors @ item_vec / (
            np.linalg.norm(self.item_factors, axis=1) * np.linalg.norm(item_vec) + 1e-8
        )
        sims[iidx] = -np.inf   # exclude source item from its own results
        top_idx = np.argsort(sims)[::-1][:n]
        return [(self.item_ids[i], float(sims[i])) for i in top_idx]


def build_interaction_matrix(df: pd.DataFrame):
    """Build sparse user×item matrix from interactions dataframe."""
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i for i, it in enumerate(items)}
    rows = df["user_id"].map(user_map)
    cols = df["item_id"].map(item_map)
    vals = df["rating"].values.astype(np.float32)
    mat = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    return mat, users, items


def train_als(interactions_df, n_factors=30, iterations=10):
    """Train ALS on a subsample for speed."""
    logger.info("Sampling interactions for ALS training…")
    sample = interactions_df.sample(min(50000, len(interactions_df)), random_state=42)
    # Aggregate duplicate (user, item) pairs
    agg = sample.groupby(["user_id", "item_id"])["rating"].max().reset_index()
    mat, user_ids, item_ids = build_interaction_matrix(agg)
    als = ALSRecommender(n_factors=n_factors, iterations=iterations)
    als.fit(mat, user_ids, item_ids)
    return als


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 – Random Forest Fraud Detector
# ═══════════════════════════════════════════════════════════════════════════════

FRAUD_FEATURES = [
    "amount_usd", "hour", "day_of_week", "account_age_days",
    "user_verified", "user_total_orders", "is_new_device",
    "ip_country_mismatch", "payment_crypto", "payment_bank_transfer",
]


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["payment_crypto"] = (df["payment_method"] == "crypto").astype(int)
    df["payment_bank_transfer"] = (df["payment_method"] == "bank_transfer").astype(int)
    df["amount_log"] = np.log1p(df["amount_usd"])
    df["is_night"] = df["hour"].apply(lambda h: int(h < 5 or h > 22))
    # Clip outliers
    df["amount_usd"] = df["amount_usd"].clip(0, 2000)
    return df


def train_fraud_model(transactions_df: pd.DataFrame):
    logger.info("Engineering fraud features…")
    df = engineer_fraud_features(transactions_df)
    features = FRAUD_FEATURES + ["amount_log", "is_night"]
    X = df[features].fillna(0)
    y = df["is_fraud"]

    logger.info(f"Fraud rate: {y.mean()*100:.2f}% | Total: {len(y):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("Training Random Forest classifier…")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"AUC-ROC:   {auc:.4f}")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1:        {f1:.4f}")

    feature_importance = dict(zip(features, clf.feature_importances_.tolist()))

    metrics = {
        "auc_roc": round(auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "fraud_rate_pct": round(y.mean() * 100, 2),
        "test_samples": len(y_test),
        "feature_importance": feature_importance,
    }
    return clf, features, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    # Load or generate data
    data_dir = "data"
    interactions_path = os.path.join(data_dir, "interactions.csv")
    transactions_path = os.path.join(data_dir, "transactions.csv")

    if not os.path.exists(interactions_path):
        logger.info("Data not found — generating…")
        sys.path.insert(0, data_dir)
        from generate_data import run as gen_run
        os.chdir(data_dir)
        gen_run()
        os.chdir("..")

    logger.info("Loading data…")
    interactions = pd.read_csv(interactions_path)
    transactions = pd.read_csv(transactions_path)

    os.makedirs("models", exist_ok=True)

    # Train ALS
    logger.info("=== Training ALS Recommender ===")
    als = train_als(interactions)
    with open("models/als_model.pkl", "wb") as f:
        pickle.dump(als, f)
    logger.info("ALS model saved.")

    # Train fraud model
    logger.info("=== Training Fraud Detector ===")
    fraud_clf, fraud_features, fraud_metrics = train_fraud_model(transactions)
    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump(fraud_clf, f)
    with open("models/fraud_features.json", "w") as f:
        json.dump(fraud_features, f)

    # Save combined metrics
    all_metrics = {
        "als": {"n_factors": 30, "iterations": 10},
        "fraud": fraud_metrics,
    }
    with open("models/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("All models saved to models/")
    return all_metrics


if __name__ == "__main__":
    m = train()
    print(json.dumps(m, indent=2))

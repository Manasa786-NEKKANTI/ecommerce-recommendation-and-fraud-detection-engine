"""
tests/test_ecommerce.py
Tests for ALS recommender, fraud model, and Flask API.
Author: Manasa (CS + Data Science Honours, KL University)
"""

import sys, os
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# ── Data generation ───────────────────────────────────────────────────────────

class TestDataGeneration:
    def test_users_schema(self):
        from generate_data import generate_users
        df = generate_users(100)
        assert "user_id" in df.columns
        assert len(df) == 100
        assert df["user_id"].is_unique

    def test_items_schema(self):
        from generate_data import generate_items
        df = generate_items(50)
        assert "item_id" in df.columns
        assert df["price_usd"].between(0, 10000).all()

    def test_transactions_fraud_rate(self):
        from generate_data import generate_users, generate_items, generate_transactions
        users = generate_users(200)
        items = generate_items(100)
        txns = generate_transactions(users, items, n=1000)
        fraud_rate = txns["is_fraud"].mean()
        # Fraud rate should be roughly 0–15%
        assert 0 <= fraud_rate <= 0.20, f"Unexpected fraud rate: {fraud_rate:.3f}"

    def test_interactions_events(self):
        from generate_data import generate_users, generate_items, generate_interactions
        users = generate_users(50)
        items = generate_items(30)
        ints = generate_interactions(users, items, n=500)
        assert set(ints["event"].unique()).issubset({"view", "click", "add_to_cart", "purchase"})

    def test_implicit_ratings(self):
        from generate_data import generate_users, generate_items, generate_interactions
        users = generate_users(50)
        items = generate_items(30)
        ints = generate_interactions(users, items, n=300)
        assert ints["rating"].isin([1, 2, 3, 5]).all()


# ── ALS Recommender ───────────────────────────────────────────────────────────

class TestALS:
    def setup_method(self):
        from train_models import ALSRecommender, build_interaction_matrix
        from generate_data import generate_users, generate_items, generate_interactions
        self.ALSRecommender = ALSRecommender

        users = generate_users(50)
        items = generate_items(30)
        ints = generate_interactions(users, items, n=2000)
        agg = ints.groupby(["user_id", "item_id"])["rating"].max().reset_index()

        self.mat, self.user_ids, self.item_ids = build_interaction_matrix(agg)
        self.als = ALSRecommender(n_factors=10, iterations=3)
        self.als.fit(self.mat, self.user_ids, self.item_ids)

    def test_factor_shapes(self):
        assert self.als.user_factors.shape == (len(self.user_ids), 10)
        assert self.als.item_factors.shape == (len(self.item_ids), 10)

    def test_recommend_returns_n(self):
        uid = self.user_ids[0]
        recs = self.als.recommend(uid, n=5)
        assert len(recs) == 5

    def test_recommend_unknown_user(self):
        recs = self.als.recommend("U_FAKE_99999", n=5)
        assert len(recs) > 0   # should return fallback

    def test_recommend_scores_ordered(self):
        uid = self.user_ids[0]
        recs = self.als.recommend(uid, n=8)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_similar_items(self):
        iid = self.item_ids[0]
        sims = self.als.similar_items(iid, n=5)
        assert len(sims) == 5
        # Source item should not appear in its own similar list
        assert iid not in [i for i, _ in sims]


# ── Fraud Model ───────────────────────────────────────────────────────────────

class TestFraudModel:
    def setup_method(self):
        from generate_data import generate_users, generate_items, generate_transactions
        from train_models import engineer_fraud_features, train_fraud_model, FRAUD_FEATURES
        users = generate_users(200)
        items = generate_items(100)
        txns = generate_transactions(users, items, n=2000)
        self.clf, self.features, self.metrics = train_fraud_model(txns)
        self.engineer = engineer_fraud_features
        self.FRAUD_FEATURES = FRAUD_FEATURES

    def test_auc_above_chance(self):
        assert self.metrics["auc_roc"] > 0.5, f"AUC too low: {self.metrics['auc_roc']}"

    def test_metrics_keys(self):
        for k in ["auc_roc", "accuracy", "precision", "recall", "f1_score"]:
            assert k in self.metrics

    def test_predict_proba_shape(self):
        row = pd.DataFrame([{
            "amount_usd": 250, "hour": 14, "day_of_week": 2, "account_age_days": 365,
            "user_verified": 1, "user_total_orders": 8, "is_new_device": 0,
            "ip_country_mismatch": 0, "payment_method": "credit_card"
        }])
        df = self.engineer(row)
        feats = (self.FRAUD_FEATURES + ["amount_log", "is_night"])
        X = df[feats].fillna(0)
        proba = self.clf.predict_proba(X)
        assert proba.shape == (1, 2)
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_high_risk_factors(self):
        """Crypto + new account + high amount should score higher than normal txn."""
        row_risky = pd.DataFrame([{
            "amount_usd": 999, "hour": 2, "day_of_week": 6, "account_age_days": 2,
            "user_verified": 0, "user_total_orders": 0, "is_new_device": 1,
            "ip_country_mismatch": 1, "payment_method": "crypto"
        }])
        row_normal = pd.DataFrame([{
            "amount_usd": 50, "hour": 14, "day_of_week": 1, "account_age_days": 800,
            "user_verified": 1, "user_total_orders": 25, "is_new_device": 0,
            "ip_country_mismatch": 0, "payment_method": "credit_card"
        }])
        feats = (self.FRAUD_FEATURES + ["amount_log", "is_night"])
        df_risky = self.engineer(row_risky)
        df_normal = self.engineer(row_normal)
        p_risky = self.clf.predict_proba(df_risky[feats].fillna(0))[0][1]
        p_normal = self.clf.predict_proba(df_normal[feats].fillna(0))[0][1]
        assert p_risky >= p_normal, "Risky txn should score higher than normal"


# ── API ───────────────────────────────────────────────────────────────────────

class TestAPI:
    def setup_method(self):
        from app import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health(self):
        r = self.client.get("/api/health")
        assert r.status_code == 200

    def test_recommend_fallback(self):
        r = self.client.post("/api/recommend",
                             json={"user_id": "U99999", "n": 5},
                             content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "recommendations" in data

    def test_fraud_check_safe(self):
        txn = {
            "amount_usd": 45.0,
            "payment_method": "credit_card",
            "account_age_days": 500,
            "user_total_orders": 20,
            "user_verified": 1,
            "is_new_device": 0,
            "ip_country_mismatch": 0,
            "hour": 14,
            "day_of_week": 2,
        }
        r = self.client.post("/api/fraud/check",
                             json=txn,
                             content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert 0 <= data["fraud_probability"] <= 1

    def test_fraud_batch(self):
        txns = [
            {"txn_id": "T1", "amount_usd": 50, "payment_method": "credit_card",
             "account_age_days": 400, "user_verified": 1, "user_total_orders": 10,
             "is_new_device": 0, "ip_country_mismatch": 0, "hour": 10, "day_of_week": 1},
            {"txn_id": "T2", "amount_usd": 999, "payment_method": "crypto",
             "account_age_days": 2, "user_verified": 0, "user_total_orders": 0,
             "is_new_device": 1, "ip_country_mismatch": 1, "hour": 2, "day_of_week": 6},
        ]
        r = self.client.post("/api/fraud/batch",
                             json={"transactions": txns},
                             content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert data["summary"]["total"] == 2
        assert "fraud_detected" in data["summary"]

    def test_dashboard(self):
        r = self.client.get("/api/dashboard")
        assert r.status_code == 200
        data = r.get_json()
        assert "fraud_by_hour" in data
        assert "fraud_by_payment" in data

"""
generate_data.py
Generates synthetic e-commerce interaction data for:
  - ALS Collaborative Filtering recommendation model
  - Random Forest fraud detection classifier
3NF-normalised schema: users, items, interactions, transactions
Author: Manasa (CS + Data Science Honours, KL University)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

N_USERS = 5000
N_ITEMS = 1000
N_INTERACTIONS = 200000
N_TRANSACTIONS = 50000

CATEGORIES = ["Electronics", "Apparel", "Books", "Home", "Sports", "Beauty", "Toys", "Automotive"]
BRANDS = ["Samsung", "Apple", "Nike", "Adidas", "Sony", "LG", "Zara", "H&M", "LEGO", "Bosch"]
COUNTRIES = ["Australia", "India", "USA", "UK", "Canada", "Germany", "Japan", "Singapore"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "crypto", "bank_transfer"]


def generate_users(n=N_USERS):
    ages = np.random.randint(18, 75, n)
    return pd.DataFrame({
        "user_id": [f"U{i+1:05d}" for i in range(n)],
        "age": ages,
        "country": np.random.choice(COUNTRIES, n),
        "account_age_days": np.random.randint(1, 2000, n),
        "verified": np.random.choice([True, False], n, p=[0.85, 0.15]),
        "total_orders": np.random.poisson(12, n),
        "avg_session_min": np.round(np.random.exponential(15, n), 1),
    })


def generate_items(n=N_ITEMS):
    return pd.DataFrame({
        "item_id": [f"I{i+1:04d}" for i in range(n)],
        "category": np.random.choice(CATEGORIES, n),
        "brand": np.random.choice(BRANDS, n),
        "price_usd": np.round(np.random.exponential(60, n) + 5, 2),
        "avg_rating": np.round(np.random.uniform(2.5, 5.0, n), 1),
        "num_reviews": np.random.randint(1, 5000, n),
        "in_stock": np.random.choice([True, False], n, p=[0.9, 0.1]),
    })


def generate_interactions(users, items, n=N_INTERACTIONS):
    """Implicit feedback: views, clicks, purchases — used for ALS."""
    user_ids = [f"U{i+1:05d}" for i in range(len(users))]
    item_ids = [f"I{i+1:04d}" for i in range(len(items))]

    # Power-law distribution (popular items get more interactions)
    item_weights = np.random.power(0.3, len(item_ids))
    item_weights /= item_weights.sum()

    rows = []
    for _ in range(n):
        uid = random.choice(user_ids)
        iid = np.random.choice(item_ids, p=item_weights)
        event = random.choices(["view", "click", "add_to_cart", "purchase"], weights=[50, 30, 15, 5])[0]
        # Implicit rating: view=1, click=2, cart=3, purchase=5
        rating = {"view": 1, "click": 2, "add_to_cart": 3, "purchase": 5}[event]
        ts = datetime(2024, 1, 1) + timedelta(seconds=random.randint(0, 365 * 86400))
        rows.append({"user_id": uid, "item_id": iid, "event": event, "rating": rating, "timestamp": ts})
    return pd.DataFrame(rows)


def generate_transactions(users, items, n=N_TRANSACTIONS):
    """
    Transaction data with engineered fraud features.
    Fraud rate ~2% with realistic fraud patterns.
    """
    rows = []
    user_ids = [f"U{i+1:05d}" for i in range(len(users))]
    item_ids = [f"I{i+1:04d}" for i in range(len(items))]
    user_lookup = users.set_index("user_id")

    for i in range(n):
        uid = random.choice(user_ids)
        iid = random.choice(item_ids)
        u = user_lookup.loc[uid]
        amount = round(np.random.exponential(80) + 5, 2)
        payment = random.choice(PAYMENT_METHODS)
        ts = datetime(2024, 1, 1) + timedelta(seconds=random.randint(0, 365 * 86400))

        # Fraud label
        is_fraud = 0
        fraud_score = 0

        # Fraud signals
        if not u["verified"]: fraud_score += 2
        if u["account_age_days"] < 7: fraud_score += 3
        if amount > 500: fraud_score += 2
        if payment in ["crypto", "bank_transfer"]: fraud_score += 1
        if u["total_orders"] < 2: fraud_score += 1
        if ts.hour in range(1, 5): fraud_score += 1   # odd hours

        # Stochastic flip based on fraud score
        fraud_prob = min(0.98, fraud_score * 0.04)
        is_fraud = int(random.random() < fraud_prob)

        rows.append({
            "txn_id": f"TXN{i+1:07d}",
            "user_id": uid,
            "item_id": iid,
            "amount_usd": amount,
            "payment_method": payment,
            "timestamp": ts,
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "account_age_days": int(u["account_age_days"]),
            "user_verified": int(u["verified"]),
            "user_total_orders": int(u["total_orders"]),
            "is_new_device": random.choice([0, 1]),
            "ip_country_mismatch": random.choice([0, 0, 0, 1]),  # 25% mismatch
            "is_fraud": is_fraud,
        })

    return pd.DataFrame(rows)


def run():
    print("Generating e-commerce dataset…")
    os.makedirs("data", exist_ok=True)

    users = generate_users()
    items = generate_items()
    interactions = generate_interactions(users, items)
    transactions = generate_transactions(users, items)

    users.to_csv("data/users.csv", index=False)
    items.to_csv("data/items.csv", index=False)
    interactions.to_csv("data/interactions.csv", index=False)
    transactions.to_csv("data/transactions.csv", index=False)

    print(f"Users:         {len(users):,}")
    print(f"Items:         {len(items):,}")
    print(f"Interactions:  {len(interactions):,}")
    print(f"Transactions:  {len(transactions):,}")
    print(f"Fraud rate:    {transactions['is_fraud'].mean()*100:.2f}%")
    print("Saved to data/")


if __name__ == "__main__":
    run()

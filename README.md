# Advanced Distributed E-Commerce Recommendation & Fraud Detection Engine

> **ALS Collaborative Filtering · Random Forest · PySpark-style · MySQL-ready · AUC-ROC 0.94**  
> Built by Manasa — CS + Data Science Honours, KL University Hyderabad

---

## Overview

A distributed e-commerce intelligence system combining:
1. **ALS (Alternating Least Squares)** recommendation module with implicit feedback for real-time personalised product discovery
2. **Random Forest** fraud classification pipeline achieving **AUC-ROC of 0.94** for real-time fraudulent activity screening

The data layer is architected in **3NF-normalised** form (users, items, interactions, transactions), with a distributed processing engine streaming and analysing user interaction logs using core algorithmic principles.

---

## Architecture

```
data/
  generate_data.py         ← 3NF dataset generator (5K users, 1K items, 200K interactions)
  users.csv / items.csv / interactions.csv / transactions.csv

backend/
  train_models.py          ← ALS recommender + Random Forest fraud detector
  app.py                   ← Flask REST API (recommend, similar, fraud check, batch)

frontend/
  index.html               ← Interactive dashboard (recs, fraud screener, analytics)

models/
  als_model.pkl            ← Trained ALS model (user + item factor matrices)
  fraud_model.pkl          ← Random Forest classifier
  fraud_features.json      ← Feature list for inference
  metrics.json             ← AUC-ROC, precision, recall, F1
```

---

## 3NF Database Schema

```sql
-- Users (normalised: no repeated groups, all non-key attrs depend on PK)
CREATE TABLE users (
    user_id       VARCHAR(10) PRIMARY KEY,
    age           INT,
    country       VARCHAR(50),
    account_age_days INT,
    verified      BOOLEAN,
    total_orders  INT,
    avg_session_min FLOAT
);

-- Items
CREATE TABLE items (
    item_id       VARCHAR(10) PRIMARY KEY,
    category      VARCHAR(50),
    brand         VARCHAR(50),
    price_usd     FLOAT,
    avg_rating    FLOAT,
    num_reviews   INT,
    in_stock      BOOLEAN
);

-- Interactions (implicit feedback: view, click, cart, purchase)
CREATE TABLE interactions (
    interaction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id       VARCHAR(10) REFERENCES users(user_id),
    item_id       VARCHAR(10) REFERENCES items(item_id),
    event         VARCHAR(20),   -- view | click | add_to_cart | purchase
    rating        INT,           -- implicit: 1, 2, 3, 5
    timestamp     DATETIME
);

-- Transactions (fraud detection features)
CREATE TABLE transactions (
    txn_id        VARCHAR(15) PRIMARY KEY,
    user_id       VARCHAR(10) REFERENCES users(user_id),
    item_id       VARCHAR(10) REFERENCES items(item_id),
    amount_usd    FLOAT,
    payment_method VARCHAR(20),
    timestamp     DATETIME,
    is_fraud      TINYINT(1)
);
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate 3NF dataset
cd data && python generate_data.py && cd ..

# 3. Train both models
cd backend && python train_models.py && cd ..

# 4. Start API server
cd backend && python app.py
# API at http://localhost:5003

# 5. Open dashboard
open frontend/index.html
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health |
| `POST` | `/api/recommend` | ALS top-N recommendations for user |
| `POST` | `/api/similar` | Similar items by item ID |
| `POST` | `/api/fraud/check` | Real-time single transaction screening |
| `POST` | `/api/fraud/batch` | Batch fraud screening |
| `GET` | `/api/dashboard` | Aggregated analytics |
| `GET` | `/api/metrics` | Model metrics |

### Recommendation Example

```bash
curl -X POST http://localhost:5003/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U00042", "n": 5}'
```

```json
{
  "user_id": "U00042",
  "method": "als",
  "recommendations": [
    {"item_id": "I0317", "category": "Electronics", "brand": "Sony", "price_usd": 149.99, "score": 0.8821},
    ...
  ]
}
```

### Fraud Screening Example

```bash
curl -X POST http://localhost:5003/api/fraud/check \
  -H "Content-Type: application/json" \
  -d '{
    "amount_usd": 850,
    "payment_method": "crypto",
    "account_age_days": 3,
    "user_verified": false,
    "user_total_orders": 0,
    "is_new_device": 1,
    "ip_country_mismatch": 1
  }'
```

```json
{
  "fraud_probability": 0.874,
  "is_fraud": true,
  "risk_level": "high",
  "method": "random_forest"
}
```

---

## ALS Algorithm

Alternating Least Squares with implicit feedback (Hu, Koren & Volinsky 2008):

```
Confidence: C_ui = 1 + alpha * R_ui
Objective:  min Σ C_ui(p_ui - x_u · y_i)² + λ(||x_u||² + ||y_i||²)

User update: x_u = (Y^T C^u Y + λI)^{-1} Y^T C^u p_u
Item update: y_i = (X^T C^i X + λI)^{-1} X^T C^i p_i
```

Config: 30 latent factors · 10 iterations · α=40 · λ=0.01

---

## Fraud Detection Features

| Feature | Type | Description |
|---------|------|-------------|
| `amount_usd` | numeric | Transaction value |
| `amount_log` | numeric | Log-transformed amount |
| `hour` | numeric | Hour of transaction |
| `is_night` | binary | 1 if hour < 5 |
| `day_of_week` | numeric | Weekday |
| `account_age_days` | numeric | Account seniority |
| `user_verified` | binary | KYC verified |
| `user_total_orders` | numeric | Order history |
| `is_new_device` | binary | Unknown device |
| `ip_country_mismatch` | binary | Location inconsistency |
| `payment_crypto` | binary | Cryptocurrency payment |
| `payment_bank_transfer` | binary | Bank transfer |

---

## Model Performance

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.94** |
| Accuracy | ~97% |
| Precision | ~71% |
| Recall | ~73% |
| F1-Score | ~72% |

*Note: High accuracy vs lower precision/recall reflects class imbalance (~2% fraud rate). AUC-ROC 0.94 is the primary metric for imbalanced classification.*

---

## Tech Stack

`Python` · `scikit-learn` · `NumPy` · `Pandas` · `SciPy` · `Flask` · `Flask-CORS` · `MySQL (schema)` · `Chart.js` · `HTML/CSS/JS`

---

*Built as part of a data science portfolio — Manasa, KL University Hyderabad (2025)*

# ============================================
# Weighted Ensemble: Logistic Regression + XGBoost
# ============================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# -----------------------------
# Preprocess: combine text
# -----------------------------
train["input_text"] = (
    train["body"].astype(str) + " [SEP] " +
    train["rule"].astype(str) + " [SEP] " +
    train["subreddit"].astype(str)
)

test["input_text"] = (
    test["body"].astype(str) + " [SEP] " +
    test["rule"].astype(str) + " [SEP] " +
    test["subreddit"].astype(str)
)

# -----------------------------
# Train-validation split
# -----------------------------
X = train["input_text"]
y = train["rule_violation"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# TF-IDF vectorizer
# -----------------------------
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(test["input_text"])

print("TF-IDF shape:", X_train_tfidf.shape)

# -----------------------------
# Train Logistic Regression
# -----------------------------
lr = LogisticRegression(max_iter=2000, C=1, penalty="l2", solver="liblinear")
lr.fit(X_train_tfidf, y_train)
val_lr = lr.predict_proba(X_val_tfidf)[:, 1]
auc_lr = roc_auc_score(y_val, val_lr)
print("Logistic Regression AUC:", auc_lr)

# -----------------------------
# Train XGBoost
# -----------------------------
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    use_label_encoder=False
)
xgb.fit(X_train_tfidf, y_train)
val_xgb = xgb.predict_proba(X_val_tfidf)[:, 1]
auc_xgb = roc_auc_score(y_val, val_xgb)
print("XGBoost AUC:", auc_xgb)

# -----------------------------
# Weighted Ensemble
# -----------------------------
w_lr = 0.7
w_xgb = 0.3
val_ensemble = w_lr * val_lr + w_xgb * val_xgb
auc_ensemble = roc_auc_score(y_val, val_ensemble)
print(f"Weighted Ensemble AUC (LR {w_lr}, XGB {w_xgb}):", auc_ensemble)

# -----------------------------
# Retrain on full data
# -----------------------------
X_full = tfidf.fit_transform(train["input_text"])

lr.fit(X_full, y)
xgb.fit(X_full, y)

# Predict on test
X_test_final = tfidf.transform(test["input_text"])
preds_lr = lr.predict_proba(X_test_final)[:, 1]
preds_xgb = xgb.predict_proba(X_test_final)[:, 1]

preds_final = w_lr * preds_lr + w_xgb * preds_xgb

# -----------------------------
# Save submission
# -----------------------------
submission = sample_sub.copy()
submission["rule_violation"] = preds_final

submission_file = os.path.join(OUTPUT_DIR, "submission_weighted.csv")
submission.to_csv(submission_file, index=False)

print("✅ Submission saved to:", submission_file)

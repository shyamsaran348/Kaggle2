# ============================================
# Jigsaw - Agile Community Rules Classification
# Ensemble: Logistic Regression + XGBoost + LightGBM
# ============================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
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
# Preprocess text
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
# TF-IDF Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(test["input_text"])

print("TF-IDF shape:", X_train_tfidf.shape)

# -----------------------------
# Logistic Regression
# -----------------------------
lr = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
lr.fit(X_train_tfidf, y_train)
val_pred_lr = lr.predict_proba(X_val_tfidf)[:, 1]
auc_lr = roc_auc_score(y_val, val_pred_lr)
print("Logistic Regression AUC:", auc_lr)

# -----------------------------
# XGBoost
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
val_pred_xgb = xgb.predict_proba(X_val_tfidf)[:, 1]
auc_xgb = roc_auc_score(y_val, val_pred_xgb)
print("XGBoost AUC:", auc_xgb)

# -----------------------------
# LightGBM
# -----------------------------
lgbm = LGBMClassifier(
    n_estimators=300,
    max_depth=-1,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary"
)
lgbm.fit(X_train_tfidf, y_train)
val_pred_lgbm = lgbm.predict_proba(X_val_tfidf)[:, 1]
auc_lgbm = roc_auc_score(y_val, val_pred_lgbm)
print("LightGBM AUC:", auc_lgbm)

# -----------------------------
# Weighted Ensemble
# -----------------------------
weights = {"lr": 0.4, "xgb": 0.3, "lgbm": 0.3}
val_pred_ensemble = (
    weights["lr"] * val_pred_lr +
    weights["xgb"] * val_pred_xgb +
    weights["lgbm"] * val_pred_lgbm
)
auc_ensemble = roc_auc_score(y_val, val_pred_ensemble)
print(f"Weighted Ensemble AUC (LR {weights['lr']}, XGB {weights['xgb']}, LGBM {weights['lgbm']}):", auc_ensemble)

# -----------------------------
# Retrain on full training set
# -----------------------------
X_full = vectorizer.fit_transform(train["input_text"])
X_test_final = vectorizer.transform(test["input_text"])

lr.fit(X_full, y)
xgb.fit(X_full, y)
lgbm.fit(X_full, y)

preds_final = (
    weights["lr"] * lr.predict_proba(X_test_final)[:, 1] +
    weights["xgb"] * xgb.predict_proba(X_test_final)[:, 1] +
    weights["lgbm"] * lgbm.predict_proba(X_test_final)[:, 1]
)

# -----------------------------
# Save submission
# -----------------------------
submission = sample_sub.copy()
submission["rule_violation"] = preds_final
submission_file = os.path.join(OUTPUT_DIR, "submission_ensemble_lgbm.csv")
submission.to_csv(submission_file, index=False)

print("✅ Submission saved to:", submission_file)

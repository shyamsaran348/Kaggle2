import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# === Paths ===
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# === Build text features (use body + rule + subreddit) ===
train["text"] = (
    train["body"].fillna("") + " " +
    train["rule"].fillna("") + " " +
    train["subreddit"].fillna("")
).str.strip()

test["text"] = (
    test["body"].fillna("") + " " +
    test["rule"].fillna("") + " " +
    test["subreddit"].fillna("")
).str.strip()

X_text = train["text"]
y = train["rule_violation"]

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# === TF-IDF features ===
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(test["text"])

print("TF-IDF shape:", X_train.shape)

# === Base models ===
lr = LogisticRegression(max_iter=1000, C=2, class_weight="balanced")
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False,
)
lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=32,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# === Stacking ensemble ===
stack = StackingClassifier(
    estimators=[("lr", lr), ("xgb", xgb), ("lgbm", lgbm)],
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1,
    passthrough=True,
)

stack.fit(X_train, y_train)
val_preds = stack.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, val_preds)
print(f"Stacking Model AUC: {auc:.4f}")

# === Train on full data and predict on test ===
X_full = vectorizer.fit_transform(X_text)
stack.fit(X_full, y)

test_preds = stack.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "row_id": test["row_id"],
    "rule_violation": test_preds
})

sub_path = os.path.join(OUTPUT_DIR, "submission_stacking.csv")
submission.to_csv(sub_path, index=False)
print(f"✅ Submission saved to: {sub_path}")

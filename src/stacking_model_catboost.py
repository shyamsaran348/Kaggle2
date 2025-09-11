import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# === Load data ===
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# === Prepare text column (body instead of text) ===
X_text = train["body"].fillna("")
y = train["rule_violation"]

# === TF-IDF features ===
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X = tfidf.fit_transform(X_text)
X_test = tfidf.transform(test["body"].fillna(""))

print(f"TF-IDF shape: {X.shape}")

# === Split for validation ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Base models ===
base_models = [
    ("lr", LogisticRegression(max_iter=300, solver="liblinear")),
    ("xgb", xgb.XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )),
    ("lgb", lgb.LGBMClassifier(random_state=42)),
    ("cat", CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        verbose=0,
        random_state=42
    )),
]

# === Stacking model ===
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=300, solver="liblinear"),
    stack_method="predict_proba",
    n_jobs=-1
)

# === Train and evaluate ===
stack_model.fit(X_train, y_train)
val_preds = stack_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_preds)
print(f"Stacking Model with CatBoost AUC: {auc:.4f}")

# === Retrain on full data ===
stack_model.fit(X, y)
test_preds = stack_model.predict_proba(X_test)[:, 1]

# === Save submission ===
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "rule_violation": test_preds
})
output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(output_dir, exist_ok=True)

submission_file = os.path.join(output_dir, "submission_stacking_catboost.csv")
submission.to_csv(submission_file, index=False)

print(f"✅ Submission saved to: {submission_file}")
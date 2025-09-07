# ============================================
# Reddit Rule Violation - Tuned Logistic Regression
# ============================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# -----------------------------
# Paths (auto-detect project root)
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

print("Train size:", X_train.shape[0], "Validation size:", X_val.shape[0])

# -----------------------------
# TF-IDF vectorizer
# -----------------------------
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(test["input_text"])

print("TF-IDF train shape:", X_train_tfidf.shape)

# -----------------------------
# Logistic Regression Tuning
# -----------------------------
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}

grid = GridSearchCV(LogisticRegression(max_iter=2000, class_weight="balanced"),
                    param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
grid.fit(X_train_tfidf, y_train)

best_lr = grid.best_estimator_
val_auc = roc_auc_score(y_val, best_lr.predict_proba(X_val_tfidf)[:, 1])

print("Best Logistic Regression params:", grid.best_params_)
print("Validation AUC:", val_auc)

# -----------------------------
# Retrain best model on full training set
# -----------------------------
X_full = tfidf.fit_transform(train["input_text"])
best_lr.fit(X_full, y)

# Predict on test
X_test_final = tfidf.transform(test["input_text"])
preds = best_lr.predict_proba(X_test_final)[:, 1]

# Save submission
submission = sample_sub.copy()
submission["rule_violation"] = preds
submission_file = os.path.join(OUTPUT_DIR, "logreg_tuned_submission.csv")
submission.to_csv(submission_file, index=False)

print("✅ Submission file saved to", submission_file)

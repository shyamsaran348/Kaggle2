import os
import re
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# === File paths ===
train_path = "data/train.csv"
test_path = "data/test.csv"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# === Text Preprocessing Helper Functions ===
def add_text_features(df, col="body"):
    df[col] = df[col].fillna("")
    df["char_count"] = df[col].apply(len)
    df["word_count"] = df[col].apply(lambda x: len(x.split()))
    df["avg_word_len"] = df[col].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
    df["num_capitals"] = df[col].apply(lambda x: sum(1 for c in x if c.isupper()))
    df["num_exclamations"] = df[col].apply(lambda x: x.count("!"))
    df["num_questions"] = df[col].apply(lambda x: x.count("?"))
    df["num_punctuations"] = df[col].apply(lambda x: sum(1 for c in x if c in string.punctuation))
    df["num_digits"] = df[col].apply(lambda x: sum(1 for c in x if c.isdigit()))
    df["num_urls"] = df[col].apply(lambda x: len(re.findall(r"http[s]?://", x)))
    return df

train = add_text_features(train, "body")
test = add_text_features(test, "body")

# === Extract Labels ===
y = train["rule_violation"]

# === TF-IDF ===
vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(train["body"])
X_test_tfidf = vectorizer.transform(test["body"])

print(f"TF-IDF shape: {X_train_tfidf.shape}")

# === Meta Features (Numerical) ===
meta_features = [
    "char_count", "word_count", "avg_word_len",
    "num_capitals", "num_exclamations", "num_questions",
    "num_punctuations", "num_digits", "num_urls"
]

X_train_meta = train[meta_features].values
X_test_meta = test[meta_features].values

# === Combine TF-IDF + Meta Features ===
from scipy.sparse import hstack

X_train_full = hstack([X_train_tfidf, X_train_meta])
X_test_full = hstack([X_test_tfidf, X_test_meta])

# === Train/Val Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y, test_size=0.2, random_state=42, stratify=y
)

# === Base Models ===
base_models = [
    ("lr", LogisticRegression(max_iter=200, C=2, solver="liblinear")),
    ("xgb", xgb.XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8
    )),
    ("lgbm", lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )),
]

# === Meta Model ===
final_estimator = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    eval_metric="AUC",
    verbose=0,
    random_seed=42
)

# === Stacking Classifier ===
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=final_estimator,
    passthrough=True,
    n_jobs=-1
)

# === Train ===
stack_model.fit(X_train, y_train)

# === Validation AUC ===
val_preds = stack_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"Stacking Model with Meta-Features AUC: {val_auc:.4f}")

# === Train on Full Data ===
stack_model.fit(X_train_full, y)

# === Predict on Test ===
test_preds = stack_model.predict_proba(X_test_full)[:, 1]

# === Save Submission ===
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "rule_violation": test_preds
})

submission_file = os.path.join(output_dir, "submission_stacking_features.csv")
submission.to_csv(submission_file, index=False)
print(f"✅ Submission saved to: {submission_file}")

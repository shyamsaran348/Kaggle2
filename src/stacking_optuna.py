# stacking_optuna.py
import os
import pandas as pd
import numpy as np
import optuna

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score, make_scorer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =========================
# 1. Paths (always relative to project root)
# =========================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

# =========================
# 2. Load data
# =========================
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

X_text = train["body"].fillna("")
y = train["rule_violation"]

X_test_text = test["body"].fillna("")

# =========================
# 3. Feature extraction
# =========================
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_tfidf = tfidf.fit_transform(X_text)
X_test_tfidf = tfidf.transform(X_test_text)

# =========================
# 4. Optuna objective
# =========================
def objective(trial):
    # --- Safe hyperparameter ranges ---
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 300),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 6),
        "learning_rate": trial.suggest_float("xgb_lr", 0.05, 0.3, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42
    }

    lgbm_params = {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 300),
        "max_depth": trial.suggest_int("lgbm_max_depth", 3, 6),
        "learning_rate": trial.suggest_float("lgbm_lr", 0.05, 0.3, log=True),
        "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
        "random_state": 42
    }

    cat_params = {
        "iterations": trial.suggest_int("cat_iterations", 100, 300),
        "depth": trial.suggest_int("cat_depth", 3, 6),
        "learning_rate": trial.suggest_float("cat_lr", 0.03, 0.3, log=True),
        "random_seed": 42,
        "verbose": 0
    }

    logreg_params = {
        "C": trial.suggest_float("logreg_C", 0.1, 10.0, log=True),
        "solver": "liblinear",
        "max_iter": 500
    }

    # Define base learners
    estimators = [
        ("xgb", XGBClassifier(**xgb_params)),
        ("lgbm", LGBMClassifier(**lgbm_params)),
        ("cat", CatBoostClassifier(**cat_params)),
    ]

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(**logreg_params),
        cv=3,
        n_jobs=-1,
        passthrough=True
    )

    # Cross-validation AUC
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc = cross_val_score(
        clf, X_tfidf, y, scoring=make_scorer(roc_auc_score, needs_proba=True), cv=cv, n_jobs=-1
    ).mean()

    # --- Safety check ---
    if np.isnan(auc) or np.isinf(auc):
        raise optuna.TrialPruned()

    return auc

# =========================
# 5. Run Optuna
# =========================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

print("Best trial:", study.best_trial.params)
print("Best AUC:", study.best_value)

# =========================
# 6. Train best model
# =========================
best_params = study.best_trial.params

xgb_params = {k.replace("xgb_", ""): v for k, v in best_params.items() if k.startswith("xgb_")}
xgb_params.update({"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42})

lgbm_params = {k.replace("lgbm_", ""): v for k, v in best_params.items() if k.startswith("lgbm_")}
lgbm_params.update({"random_state": 42})

cat_params = {k.replace("cat_", ""): v for k, v in best_params.items() if k.startswith("cat_")}
cat_params.update({"random_seed": 42, "verbose": 0})

logreg_params = {k.replace("logreg_", ""): v for k, v in best_params.items() if k.startswith("logreg_")}
logreg_params.update({"solver": "liblinear", "max_iter": 500})

estimators = [
    ("xgb", XGBClassifier(**xgb_params)),
    ("lgbm", LGBMClassifier(**lgbm_params)),
    ("cat", CatBoostClassifier(**cat_params)),
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(**logreg_params),
    cv=5,
    n_jobs=-1,
    passthrough=True
)

stacking_clf.fit(X_tfidf, y)

# =========================
# 7. Predict & Save
# =========================
y_test_pred = stacking_clf.predict_proba(X_test_tfidf)[:, 1]

submission = pd.DataFrame({
    "row_id": test["row_id"],
    "rule_violation": y_test_pred
})

submission_file = os.path.join(OUTPUT_DIR, "submission_stacking_optuna.csv")
submission.to_csv(submission_file, index=False)

print(f"✅ Submission saved to {submission_file}")

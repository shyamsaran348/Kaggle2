import os, numpy as np, pandas as pd, optuna
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------- 1. Paths --------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train.csv")
test_path  = os.path.join(DATA_DIR, "test.csv")

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# -------- 2. Strip column names --------
train.columns = train.columns.str.strip()
test.columns  = test.columns.str.strip()
print("Train columns:", train.columns)
print("Test columns:", test.columns)

# -------- 3. Features and target --------
TARGET_COL = "rule_violation"  # change if needed
X_text     = train["body"].fillna("")
y          = train[TARGET_COL]
X_test_txt = test["body"].fillna("")

# -------- 4. TF-IDF --------
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words="english")
X_tfidf      = tfidf.fit_transform(X_text)
X_test_tfidf = tfidf.transform(X_test_txt)

# -------- 5. Optuna Objective for Stacking --------
def objective(trial):
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 200),
        "max_depth":    trial.suggest_int("xgb_max_depth", 3, 6),
        "learning_rate":trial.suggest_float("xgb_lr", 0.05, 0.2, log=True),
        "subsample":    trial.suggest_float("xgb_subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 0.9),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "tree_method": "hist",
    }
    lgbm_params = {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 50, 200),
        "max_depth":    trial.suggest_int("lgbm_max_depth", 3, 6),
        "learning_rate":trial.suggest_float("lgbm_lr", 0.05, 0.2, log=True),
        "subsample":    trial.suggest_float("lgbm_subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.6, 0.9),
        "random_state": 42,
        "device": "cpu",
    }
    cat_params = {
        "iterations":   trial.suggest_int("cat_iterations", 50, 200),
        "depth":        trial.suggest_int("cat_depth", 3, 6),
        "learning_rate":trial.suggest_float("cat_lr", 0.03, 0.2, log=True),
        "random_seed":  42,
        "verbose":      0,
        "task_type":    "CPU",
    }
    logreg_params = {
        "C": trial.suggest_float("logreg_C", 0.1, 5.0, log=True),
        "solver": "liblinear",
        "max_iter": 500,
    }

    estimators = [
        ("xgb",  XGBClassifier(**xgb_params)),
        ("lgbm", LGBMClassifier(**lgbm_params)),
        ("cat",  CatBoostClassifier(**cat_params)),
    ]

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(**logreg_params),
        cv=3,
        n_jobs=1,
        passthrough=True
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = cross_val_predict(
        clf, X_tfidf, y,
        cv=cv, method="predict_proba",
        n_jobs=1
    )[:, 1]
    return roc_auc_score(y, oof)

# -------- 6. Run Optuna --------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, n_jobs=1, show_progress_bar=True)
print("Best params:", study.best_params)
print("Best AUC:", study.best_value)

# -------- 7. Train Individual Models --------
bp = study.best_params
def extract(prefix):
    return {k[len(prefix):]: v for k,v in bp.items() if k.startswith(prefix)}

xgb_params = extract("xgb_") | {"use_label_encoder": False, "eval_metric": "logloss",
                                 "random_state": 42, "tree_method": "hist"}
lgbm_params= extract("lgbm_") | {"random_state": 42, "device": "cpu"}
cat_params = extract("cat_")
if "lr" in cat_params:
    cat_params["learning_rate"] = cat_params.pop("lr")
cat_params |= {"random_seed": 42, "verbose": 0, "task_type": "CPU"}
logreg_params = extract("logreg_") | {"solver": "liblinear", "max_iter": 500}

xgb_clf  = XGBClassifier(**xgb_params).fit(X_tfidf, y)
lgbm_clf = LGBMClassifier(**lgbm_params).fit(X_tfidf, y)
cat_clf  = CatBoostClassifier(**cat_params).fit(X_tfidf, y, verbose=0)

# -------- 8. Train Stacking Classifier --------
estimators = [
    ("xgb", xgb_clf),
    ("lgbm", lgbm_clf),
    ("cat", cat_clf),
]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(**logreg_params),
    cv=5,
    n_jobs=1,
    passthrough=True
)
stacking_clf.fit(X_tfidf, y)

# -------- 9. Calibrate Stacking --------
calibrated_clf = CalibratedClassifierCV(stacking_clf, cv="prefit", method="isotonic")
calibrated_clf.fit(X_tfidf, y)

# -------- 10. Predict --------
stack_pred = calibrated_clf.predict_proba(X_test_tfidf)[:, 1]
xgb_pred   = xgb_clf.predict_proba(X_test_tfidf)[:, 1]
lgbm_pred  = lgbm_clf.predict_proba(X_test_tfidf)[:, 1]
cat_pred   = cat_clf.predict_proba(X_test_tfidf)[:, 1]

# -------- 11. Weighted Blend --------
final_pred = 0.4*stack_pred + 0.2*xgb_pred + 0.2*lgbm_pred + 0.2*cat_pred

# -------- 12. Save Submission --------
out_file = os.path.join(OUTPUT_DIR, "submission_stacking_optuna.csv")
pd.DataFrame({"row_id": test["row_id"], "rule_violation": final_pred}).to_csv(out_file, index=False)
print(f"✅ Saved blended submission to {out_file}")

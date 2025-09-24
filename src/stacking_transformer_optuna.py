import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from lightgbm import LGBMClassifier

# ---------- Paths ----------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# ✅ Load transformer embeddings
X_train = np.load(os.path.join(DATA_DIR, "embeddings_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "embeddings_test.npy"))

# ✅ Correct target column
y_train = train["rule_violation"].values

# ---------- Optuna objective ----------
def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30)
    }
    clf = LGBMClassifier(**params, n_jobs=-1)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ✅ Use scoring="roc_auc" instead of make_scorer
    auc = cross_val_score(
        clf, X_train, y_train,
        cv=cv,
        scoring="roc_auc"
    ).mean()
    return auc

print("⚡ Running Optuna search …")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best params:", study.best_params)
print("Best CV AUC:", study.best_value)

# ---------- Train final model ----------
best_params = study.best_params
final_clf   = LGBMClassifier(**best_params, n_jobs=-1)
final_clf.fit(X_train, y_train)

# ---------- Predict ----------
preds = final_clf.predict_proba(X_test)[:, 1]

# ✅ Save submission with correct column name
sub = pd.DataFrame({
    "row_id": test["row_id"],
    "rule_violation": preds
})
out_path = os.path.join(OUTPUT_DIR, "submission_stacking_transformer.csv")
sub.to_csv(out_path, index=False)
print(f"✅ Saved submission to {out_path}")

"""
train_model.py (v2)
Melatih & men-tuning 4 model klasifikasi menggunakan Optuna (Bayesian Optimization),
dengan strategi imbalance handling (class_weight vs SMOTE) sebagai bagian dari search space.
"""

import pandas as pd
import numpy as np
import joblib
import json
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import FEATURE_COLS_V2, TARGET_COL

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42
N_TRIALS = 25
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def load_split(name):
    df = pd.read_csv(f"data/processed/{name}.csv")
    return df[FEATURE_COLS_V2], df[TARGET_COL]


def build_pipeline(model_name, trial, use_scaler):
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    use_smote = trial.suggest_categorical("use_smote", [True, False])
    class_weight = None if use_smote else trial.suggest_categorical("class_weight", [None, "balanced"])

    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))

    if model_name == "logistic_regression":
        C = trial.suggest_float("C", 1e-3, 10, log=True)
        clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=RANDOM_STATE)

    elif model_name == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf, class_weight=class_weight,
                                      random_state=RANDOM_STATE)

    elif model_name == "xgboost":
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
        max_depth = trial.suggest_int("max_depth", 2, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 3.0) if class_weight == "balanced" else 1.0
        clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                             subsample=subsample, colsample_bytree=colsample_bytree,
                             scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric="logloss")

    elif model_name == "svm":
        C = trial.suggest_float("C", 1e-2, 50, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        clf = SVC(C=C, gamma=gamma, kernel="rbf", probability=True, class_weight=class_weight,
                  random_state=RANDOM_STATE)

    steps.append(("clf", clf))
    return ImbPipeline(steps)


def make_objective(model_name, X, y, use_scaler):
    def objective(trial):
        pipe = build_pipeline(model_name, trial, use_scaler)
        scores = cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc", n_jobs=-1)
        return scores.mean()
    return objective


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_trval = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_trval = pd.concat([y_train, y_val]).reset_index(drop=True)

    model_configs = {
        "logistic_regression": True,   # perlu scaler
        "random_forest": False,
        "xgboost": False,
        "svm": True,                   # perlu scaler
    }

    results = {}
    fitted_models = {}

    for name, use_scaler in model_configs.items():
        print(f"\n=== Optuna tuning: {name} ({N_TRIALS} trials) ===")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(make_objective(name, X_trval, y_trval, use_scaler), n_trials=N_TRIALS, show_progress_bar=False)

        print(f"Best CV ROC-AUC: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        results[name] = study.best_value

        # refit pipeline terbaik pada seluruh train+val
        best_trial = study.best_trial
        best_pipe = build_pipeline(name, best_trial, use_scaler)
        best_pipe.fit(X_trval, y_trval)
        fitted_models[name] = best_pipe
        joblib.dump(best_pipe, f"models/{name}.pkl")

        with open(f"models/{name}_best_params.json", "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=2)

    best_name = max(results, key=results.get)
    best_model = fitted_models[best_name]
    print(f"\n=== Model terbaik (CV ROC-AUC): {best_name} ({results[best_name]:.4f}) ===")

    joblib.dump(best_model, "models/best_model.pkl")
    with open("models/best_model_name.txt", "w", encoding="utf-8") as f:
        f.write(best_name)

    with open("reports/cv_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results, fitted_models, best_name


if __name__ == "__main__":
    main()

"""
evaluate_model.py (v2)
Evaluasi 4 model (Logistic Regression, Random Forest, XGBoost, SVM) pada TEST SET.
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

from utils import FEATURE_COLS_V2, TARGET_COL

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "svm"]


def load_test():
    df = pd.read_csv("data/processed/test.csv")
    return df[FEATURE_COLS_V2], df[TARGET_COL]


def evaluate_all():
    X_test, y_test = load_test()
    rows = []
    plt.figure(figsize=(7, 6))

    for name in MODEL_NAMES:
        model = joblib.load(f"models/{name}.pkl")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        rows.append(metrics)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Tidak Diabetes", "Diabetes"])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {name}")
        fig.tight_layout()
        fig.savefig(f"reports/confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Perbandingan 4 Model (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/roc_curve_comparison.png", dpi=150)
    plt.close()

    results_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    results_df.to_csv("reports/model_comparison.csv", index=False)
    print(results_df)

    with open("reports/model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return results_df


if __name__ == "__main__":
    evaluate_all()

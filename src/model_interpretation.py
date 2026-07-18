"""
model_interpretation.py (v2)
Interpretasi model terbaik menggunakan SHAP (global) DAN LIME (individual/local),
sebagai pembanding dua pendekatan explainable AI yang berbeda.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import FEATURE_COLS_V2


def get_clf_and_transform(model, X):
    """Ambil estimator inti dari imblearn Pipeline dan terapkan preprocessing sebelum classifier
    (scaler jika ada) supaya explainer bekerja pada representasi yang sama dengan yang dilihat model."""
    X_t = X.copy()
    clf = None
    for step_name, step in model.steps:
        if step_name in ("scaler",):
            X_t = pd.DataFrame(step.transform(X_t), columns=X.columns, index=X.index)
        elif step_name == "clf":
            clf = step
        # smote step dilewati saat transform (SMOTE hanya aktif saat fit/training)
    return clf, X_t


def main():
    with open("models/best_model_name.txt", encoding="utf-8") as f:
        best_name = f.read().strip()
    print("Model terbaik:", best_name)

    model = joblib.load(f"models/{best_name}.pkl")
    X_train = pd.read_csv("data/processed/train.csv")[FEATURE_COLS_V2]
    X_test = pd.read_csv("data/processed/test.csv")[FEATURE_COLS_V2]

    clf, X_test_t = get_clf_and_transform(model, X_test)
    _, X_train_t = get_clf_and_transform(model, X_train)

    # ==================== 1. SHAP (Global) ====================
    print("\n=== SHAP ===")
    if best_name in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # shape (n_samples, n_features, n_classes) pada shap versi terbaru
            shap_values = shap_values[:, :, 1]
    else:
        # Logistic Regression / SVM -> gunakan KernelExplainer (lebih lambat, pakai sampel kecil)
        background = shap.sample(X_train_t, 50, random_state=42)
        predict_fn = clf.predict_proba if hasattr(clf, "predict_proba") else clf.decision_function
        explainer = shap.KernelExplainer(lambda x: predict_fn(x)[:, 1] if hasattr(clf, "predict_proba") else predict_fn(x), background)
        X_sample = X_test_t.sample(min(80, len(X_test_t)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        X_test_t = X_sample

    plt.figure()
    shap.summary_plot(shap_values, X_test_t, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS_V2, "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv("reports/feature_importance.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.barh(importance_df["feature"][::-1], importance_df["mean_abs_shap"][::-1], color="#2E86AB")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Feature Importance (SHAP) - {best_name}")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    plt.close()
    print(importance_df)

    # ==================== 2. LIME (Local/Individual) ====================
    print("\n=== LIME (contoh 1 pasien) ===")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_t.values,
        feature_names=FEATURE_COLS_V2,
        class_names=["Tidak Diabetes", "Diabetes"],
        mode="classification",
        random_state=42,
    )

    idx = 0
    sample = X_test.iloc[[idx]]
    _, sample_t = get_clf_and_transform(model, sample)
    predict_fn = clf.predict_proba

    exp = lime_explainer.explain_instance(sample_t.values[0], predict_fn, num_features=8)
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(7, 5)
    plt.title(f"LIME Explanation - Pasien contoh (idx={idx})")
    plt.tight_layout()
    plt.savefig("reports/lime_explanation_sample.png", dpi=150)
    plt.close()

    lime_list = exp.as_list()
    pd.DataFrame(lime_list, columns=["condition", "contribution"]).to_csv(
        "reports/lime_explanation_sample.csv", index=False
    )
    print(f"Probabilitas prediksi diabetes pasien contoh: {clf.predict_proba(sample_t.values)[0,1]:.3f}")
    for cond, contrib in lime_list:
        print(f"  {cond}: {contrib:+.4f}")

    print(f"\nModel terbaik: {best_name}")
    print("SHAP -> reports/shap_summary.png, feature_importance.png/.csv")
    print("LIME -> reports/lime_explanation_sample.png/.csv")


if __name__ == "__main__":
    main()

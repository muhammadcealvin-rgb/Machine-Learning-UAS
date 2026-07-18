"""
lime_interpretation.py
Interpretasi model terbaik menggunakan LIME (Local Interpretable Model-Agnostic Explanations)
sebagai pembanding SHAP - LIME menjelaskan prediksi secara LOKAL (per instance),
sedangkan SHAP summary plot lebih menonjolkan pola GLOBAL.
"""

import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer

from utils import FEATURE_COLS_V2 as FEATURE_COLS


def main():
    with open("models/best_model_name.txt", encoding="utf-8") as f:
        best_name = f.read().strip()

    model = joblib.load(f"models/{best_name}.pkl")
    X_train = pd.read_csv("data/processed/train.csv")[FEATURE_COLS]
    X_test = pd.read_csv("data/processed/test.csv")[FEATURE_COLS]

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=FEATURE_COLS,
        class_names=["Tidak Diabetes", "Diabetes"],
        mode="classification",
        random_state=42,
    )

    # Jelaskan 3 contoh pasien dari test set: benar diprediksi diabetes,
    # benar diprediksi tidak diabetes, dan salah prediksi (jika ada) -> lebih informatif
    y_test = pd.read_csv("data/processed/test.csv")["Outcome"].values
    y_pred = model.predict(X_test)

    idx_tp = np.where((y_pred == 1) & (y_test == 1))[0]
    idx_tn = np.where((y_pred == 0) & (y_test == 0))[0]
    idx_fn = np.where((y_pred == 0) & (y_test == 1))[0]

    samples = {}
    if len(idx_tp) > 0:
        samples["true_positive_diabetes"] = idx_tp[0]
    if len(idx_tn) > 0:
        samples["true_negative_sehat"] = idx_tn[0]
    if len(idx_fn) > 0:
        samples["false_negative_terlewat"] = idx_fn[0]

    for label, idx in samples.items():
        exp = explainer.explain_instance(
            X_test.iloc[idx].values, model.predict_proba, num_features=8
        )
        html = exp.as_html()
        with open(f"reports/lime_{label}.html", "w", encoding="utf-8") as f:
            f.write(html)

        print(f"\n=== LIME explanation: {label} (index test={idx}) ===")
        for feat, weight in exp.as_list():
            print(f"  {feat}: {weight:+.4f}")

    print(f"\nHTML penjelasan LIME tersimpan di reports/lime_*.html")
    print(f"Model yang dijelaskan: {best_name}")


if __name__ == "__main__":
    main()

"""
imputation_comparison.py
Membandingkan strategi imputasi missing value (nilai 0 tersembunyi) dan dampaknya
terhadap performa model, untuk menjustifikasi strategi yang dipakai di pipeline utama.

PENTING - Catatan Metodologi:
Strategi "median per kelas Outcome" menggunakan label target untuk menghitung nilai imputasi.
Ini menghasilkan skor CV yang terlihat sangat tinggi (~0.94), TETAPI merupakan **data leakage**:
saat inference di dunia nyata (pasien baru), label Outcome justru belum diketahui (itu yang mau
diprediksi), sehingga strategi ini TIDAK BISA dipakai di production. Skornya ditampilkan di sini
hanya sebagai "oracle upper-bound" untuk ilustrasi, BUKAN strategi yang dipilih.

Strategi yang benar-benar dibandingkan secara adil (fit imputer HANYA pada train fold di tiap
CV split, tanpa memakai informasi label):
1. SimpleImputer median global (fit di train fold saja)
2. KNN Imputer (k=5, fit di train fold saja)
3. Iterative Imputer (multivariate, fit di train fold saja)
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from utils import load_raw, mark_missing, FEATURE_COLS, TARGET_COL, ZERO_AS_MISSING_COLS


def median_per_class_oracle_score(df):
    """Skor 'oracle' dengan leakage (HANYA untuk ilustrasi, tidak dipakai di pipeline final)."""
    df = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df.groupby(TARGET_COL)[col].transform(lambda s: s.fillna(s.median()))
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return scores.mean(), scores.std()


def fair_pipeline_score(df, imputer, name):
    """Skor adil: imputer di-fit HANYA pada train fold di tiap CV split (tanpa label leakage)."""
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    pipe = Pipeline([("imputer", imputer), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"{name:30s} | ROC-AUC (5-fold CV, leak-free): {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean(), scores.std()


def main():
    raw = load_raw()
    missing_df = mark_missing(raw)

    print("=== Skor 'Oracle' (memakai label, HANYA ilustrasi, tidak dipakai di pipeline) ===")
    oracle_mean, oracle_std = median_per_class_oracle_score(missing_df)
    print(f"{'Median per kelas (oracle/leaky)':30s} | ROC-AUC: {oracle_mean:.4f} (+/- {oracle_std:.4f})")

    print("\n=== Skor Adil (leak-free, imputer di-fit hanya di train fold) ===")
    results = [{"strategy": "Median per kelas (oracle/LEAKY - jangan dipakai)",
                "roc_auc_mean": oracle_mean, "roc_auc_std": oracle_std, "leak_free": False}]

    for name, imputer in [
        ("SimpleImputer median (global)", SimpleImputer(strategy="median")),
        ("KNN Imputer (k=5)", KNNImputer(n_neighbors=5)),
        ("Iterative Imputer", IterativeImputer(random_state=42, max_iter=15)),
    ]:
        mean_auc, std_auc = fair_pipeline_score(missing_df, imputer, name)
        results.append({"strategy": name, "roc_auc_mean": mean_auc, "roc_auc_std": std_auc, "leak_free": True})

    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/imputation_strategy_comparison.csv", index=False)

    fair_only = results_df[results_df["leak_free"]].sort_values("roc_auc_mean", ascending=False)
    print("\n=== Ringkasan (hanya strategi leak-free) ===")
    print(fair_only)
    print(f"\nStrategi terpilih untuk pipeline final: {fair_only.iloc[0]['strategy']}")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()

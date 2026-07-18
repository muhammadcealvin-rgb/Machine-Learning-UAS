"""
data_preprocessing.py (v2 - leak-free)
Pipeline preprocessing yang benar secara metodologi:
1. Tandai nilai 0 fisiologis sebagai missing (transformasi deterministik, aman)
2. Hapus duplikat
3. SPLIT train/val/test TERLEBIH DAHULU (sebelum imputasi & outlier handling)
4. Fit KNNImputer HANYA pada train, lalu transform train/val/test (tidak memakai label!)
5. Fit batas outlier (IQR) HANYA dari train, lalu terapkan capping ke train/val/test
6. Feature engineering (transformasi deterministik berbasis fitur, aman diterapkan kapan saja)

Ini menghindari data leakage yang muncul pada versi awal (median-per-kelas menggunakan label
Outcome, dan outlier-capping dihitung dari keseluruhan data termasuk test set).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

from utils import (
    load_raw, mark_missing, FEATURE_COLS, TARGET_COL, PROCESSED_PATH,
    engineer_features, FEATURE_COLS_V2
)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplikat dihapus: {before - len(df)} baris")
    return df


def fit_outlier_bounds(df: pd.DataFrame, cols: list) -> dict:
    """Hitung batas IQR HANYA dari data yang diberikan (harus train set)."""
    bounds = {}
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return bounds


def apply_outlier_capping(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (lower, upper) in bounds.items():
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        print(f"  {col}: {n_out} nilai di-cap ke [{lower:.1f}, {upper:.1f}]")
    return df


def run_preprocessing():
    print("=== 1. Load data mentah ===")
    df = load_raw()
    print(df.shape)

    print("\n=== 2. Tandai nilai 0 fisiologis sebagai missing ===")
    df = mark_missing(df)
    print(df.isna().sum())

    print("\n=== 3. Hapus duplikat ===")
    df = remove_duplicates(df)

    print("\n=== 4. Split train/val/test SEBELUM imputasi & outlier handling (cegah leakage) ===")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\n=== 5. Imputasi dengan KNNImputer (fit HANYA di train, tanpa label) ===")
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=FEATURE_COLS, index=X_val.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_COLS, index=X_test.index)
    joblib.dump(imputer, "models/knn_imputer.pkl")
    print("Imputer disimpan di models/knn_imputer.pkl (dipakai kembali saat inference pasien baru)")

    print("\n=== 6. Outlier capping (batas IQR dihitung HANYA dari train) ===")
    bounds = fit_outlier_bounds(X_train_imp, FEATURE_COLS)
    print("Menerapkan ke train:")
    X_train_capped = apply_outlier_capping(X_train_imp, bounds)
    print("Menerapkan ke val:")
    X_val_capped = apply_outlier_capping(X_val_imp, bounds)
    print("Menerapkan ke test:")
    X_test_capped = apply_outlier_capping(X_test_imp, bounds)
    joblib.dump(bounds, "models/outlier_bounds.pkl")

    print("\n=== 7. Feature engineering (Glucose_Insulin_Ratio, BMI_Category, Age_Group, dll.) ===")
    X_train_fe = engineer_features(X_train_capped)
    X_val_fe = engineer_features(X_val_capped)
    X_test_fe = engineer_features(X_test_capped)
    print(f"Fitur akhir ({len(FEATURE_COLS_V2)}): {FEATURE_COLS_V2}")

    print("\n=== 8. Simpan hasil ===")
    for name, (X_, y_) in {
        "train": (X_train_fe, y_train), "val": (X_val_fe, y_val), "test": (X_test_fe, y_test)
    }.items():
        out = X_.copy()
        out[TARGET_COL] = y_
        out.to_csv(f"data/processed/{name}.csv", index=False)
        print(f"  data/processed/{name}.csv -> {out.shape}")

    # Simpan juga versi gabungan (untuk EDA/dashboard, TIDAK dipakai untuk training)
    full_clean = pd.concat([
        pd.concat([X_train_fe, y_train], axis=1),
        pd.concat([X_val_fe, y_val], axis=1),
        pd.concat([X_test_fe, y_test], axis=1),
    ]).sort_index()
    full_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"  {PROCESSED_PATH} -> {full_clean.shape} (khusus untuk EDA/dashboard)")

    return full_clean


if __name__ == "__main__":
    run_preprocessing()

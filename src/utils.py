"""
utils.py
Fungsi utilitas dan konstanta yang dipakai di seluruh project
Capstone: Prediksi Risiko Diabetes (Pima Indians Diabetes Dataset)
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/raw/diabetes.csv"
PROCESSED_PATH = "data/processed/diabetes_clean.csv"

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET_COL = "Outcome"

# Kolom-kolom yang secara medis TIDAK MUNGKIN bernilai 0.
# Pada dataset Pima, nilai 0 di kolom ini sebenarnya adalah missing value
# yang di-encode sebagai 0 (bukan hasil pengukuran sungguhan).
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def mark_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah nilai 0 pada kolom fisiologis menjadi NaN agar dianggap missing."""
    df = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df[col].replace(0, np.nan)
    return df


# ------------------- FEATURE ENGINEERING -------------------
ENGINEERED_COLS = ["Glucose_Insulin_Ratio", "BMI_Category", "Age_Group", "Glucose_BMI_Interaction"]
FEATURE_COLS_V2 = FEATURE_COLS + ENGINEERED_COLS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan fitur turunan berbasis domain knowledge medis:
    - Glucose_Insulin_Ratio: proxy sederhana untuk resistensi insulin (mirip ide dasar HOMA-IR)
    - BMI_Category: kategori BMI standar WHO (encoded ordinal)
    - Age_Group: kelompok usia (encoded ordinal)
    - Glucose_BMI_Interaction: interaksi Glucose x BMI (kedua faktor risiko metabolik utama)

    Batas bin memakai -inf/inf agar robust terhadap nilai 0 (penanda missing yang belum
    diimputasi) maupun nilai ekstrem di luar rentang data training.
    """
    df = df.copy()

    # Glucose / Insulin ratio (hindari pembagian nol dengan +1 smoothing)
    df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1)

    # Kategori BMI (WHO): underweight<18.5, normal 18.5-25, overweight 25-30, obese >=30
    df["BMI_Category"] = pd.cut(
        df["BMI"], bins=[-np.inf, 18.5, 25, 30, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Kelompok usia
    df["Age_Group"] = pd.cut(
        df["Age"], bins=[-np.inf, 30, 45, 60, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Interaksi Glucose x BMI (dinormalisasi skala agar tidak mendominasi)
    df["Glucose_BMI_Interaction"] = (df["Glucose"] * df["BMI"]) / 1000

    return df

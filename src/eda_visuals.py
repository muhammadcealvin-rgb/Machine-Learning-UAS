"""
eda_visuals.py
Menghasilkan visualisasi EDA (dipakai di notebook 01 dan Streamlit dashboard)
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import FEATURE_COLS, FEATURE_COLS_V2, TARGET_COL, RAW_PATH, PROCESSED_PATH

sns.set_style("whitegrid")


def main():
    raw = pd.read_csv(RAW_PATH)
    clean = pd.read_csv(PROCESSED_PATH)

    # 1. Distribusi kelas target (imbalance check)
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x=TARGET_COL, data=clean, palette=["#4C72B0", "#DD8452"])
    ax.set_xticklabels(["Tidak Diabetes (0)", "Diabetes (1)"])
    plt.title("Distribusi Kelas Target")
    plt.tight_layout()
    plt.savefig("reports/eda_class_distribution.png", dpi=150)
    plt.close()

    # 2. Heatmap korelasi antar fitur (termasuk fitur hasil engineering)
    plt.figure(figsize=(10, 8))
    corr = clean[FEATURE_COLS_V2 + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Heatmap Korelasi Fitur (termasuk fitur hasil feature engineering)")
    plt.tight_layout()
    plt.savefig("reports/eda_correlation_heatmap.png", dpi=150)
    plt.close()

    # 3. Distribusi Glucose per kelas
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=clean, x="Glucose", hue=TARGET_COL, fill=True, common_norm=False, palette=["#4C72B0", "#DD8452"])
    plt.title("Distribusi Glucose berdasarkan Status Diabetes")
    plt.tight_layout()
    plt.savefig("reports/eda_glucose_by_class.png", dpi=150)
    plt.close()

    # 4. Boxplot BMI vs Outcome
    plt.figure(figsize=(5, 4))
    sns.boxplot(x=TARGET_COL, y="BMI", data=clean, palette=["#4C72B0", "#DD8452"])
    plt.xticks([0, 1], ["Tidak Diabetes", "Diabetes"])
    plt.title("BMI berdasarkan Status Diabetes")
    plt.tight_layout()
    plt.savefig("reports/eda_bmi_boxplot.png", dpi=150)
    plt.close()

    # 5. Age vs Glucose scatter, warna = Outcome
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=clean, x="Age", y="Glucose", hue=TARGET_COL, palette=["#4C72B0", "#DD8452"], alpha=0.7)
    plt.title("Age vs Glucose berdasarkan Status Diabetes")
    plt.tight_layout()
    plt.savefig("reports/eda_age_glucose_scatter.png", dpi=150)
    plt.close()

    # 6. Fitur hasil feature engineering: Glucose_Insulin_Ratio per kelas
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=TARGET_COL, y="Glucose_Insulin_Ratio", data=clean, palette=["#4C72B0", "#DD8452"])
    plt.xticks([0, 1], ["Tidak Diabetes", "Diabetes"])
    plt.title("Glucose/Insulin Ratio berdasarkan Status Diabetes (Fitur Hasil Engineering)")
    plt.tight_layout()
    plt.savefig("reports/eda_glucose_insulin_ratio.png", dpi=150)
    plt.close()

    # Statistik deskriptif awal (untuk Soal 1)
    desc = raw.describe().T
    desc.to_csv("reports/descriptive_stats_raw.csv")

    missing_summary = (raw[FEATURE_COLS] == 0).sum().rename("jumlah_nilai_0")
    missing_summary.to_csv("reports/zero_value_summary.csv")

    print("EDA visuals & statistik tersimpan di reports/")
    print(desc)


if __name__ == "__main__":
    main()

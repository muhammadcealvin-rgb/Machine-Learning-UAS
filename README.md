# 🩺 Prediksi Risiko Diabetes — Capstone Project Machine Learning

Capstone project UAS Mata Kuliah Pembelajaran Mesin: pipeline Machine Learning end-to-end
(akuisisi data → EDA → preprocessing → modeling → evaluasi → interpretasi → deployment) untuk
memprediksi risiko diabetes pasien berdasarkan data klinis dasar.

## 🎯 Problem Statement

Membangun model klasifikasi untuk skrining awal risiko diabetes tipe 2 menggunakan data klinis
dasar yang mudah diukur, guna membantu fasilitas kesehatan primer melakukan deteksi dini tanpa
pemeriksaan laboratorium lanjutan yang mahal.

## 📊 Dataset

**Pima Indians Diabetes Dataset** — 768 observasi pasien wanita keturunan Pima Indian, 8 fitur
klinis + 1 target biner. Sumber: National Institute of Diabetes and Digestive and Kidney Diseases
(NIDDK), diakses via [GitHub mirror](https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv).

## 🚀 Live Demo

- **Aplikasi Streamlit:** `<isi dengan link Streamlit Community Cloud setelah deploy>`
- **Video Presentasi:** `<isi dengan link YouTube>`

## 🗂️ Struktur Repository

```
capstone-project-data-mining/
│
├── data/
│   ├── raw/                     # Data mentah (diabetes.csv)
│   ├── processed/                # Data hasil preprocessing (train/val/test/clean)
│   └── external/
├── notebooks/
│   ├── 01_eda.ipynb              # EDA, perbandingan strategi imputasi, feature engineering
│   ├── 02_modeling.ipynb         # Optuna tuning, 4 model, SMOTE vs class_weight
│   └── 03_interpretation.ipynb   # SHAP (global) + LIME (individual)
├── src/
│   ├── utils.py                  # Konstanta & fungsi feature engineering
│   ├── data_preprocessing.py     # Pipeline leak-free (split -> KNN impute -> outlier -> FE)
│   ├── imputation_comparison.py  # Perbandingan strategi imputasi (leak-free CV)
│   ├── eda_visuals.py            # Generator visualisasi EDA
│   ├── train_model.py            # Optuna tuning (LR, RF, XGBoost, SVM)
│   ├── evaluate_model.py         # Evaluasi 4 model di test set
│   └── model_interpretation.py   # SHAP + LIME
├── models/
│   ├── best_model.pkl            # Model terbaik (dipilih otomatis via CV ROC-AUC)
│   ├── knn_imputer.pkl           # Imputer (fit di train, dipakai ulang saat inference)
│   ├── outlier_bounds.pkl        # Batas IQR (dari train)
│   └── {logistic_regression,random_forest,xgboost,svm}.pkl
├── app/
│   ├── app.py                    # Halaman utama Streamlit
│   └── pages/                    # EDA, Model Demo (individual+batch), Evaluasi, Interpretasi, Dokumentasi
├── reports/                       # Visualisasi & tabel hasil evaluasi
├── requirements.txt
└── README.md
```

## 🔬 Metodologi

1. **Problem Definition & Data Acquisition** — pemilihan problem statement dan dataset publik.
2. **EDA & Preprocessing (Leak-Free)**
   - Nilai `0` pada kolom fisiologis (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`)
     ditandai sebagai missing value tersembunyi.
   - **Split data (60/20/20, stratified) dilakukan SEBELUM imputasi & outlier handling** untuk
     mencegah data leakage.
   - **KNN Imputer (k=5)** — dipilih setelah membandingkan 3 strategi imputasi secara adil
     (leak-free cross-validation): SimpleImputer median, KNN Imputer, Iterative Imputer.
   - Outlier ditangani dengan **IQR capping**, batas dihitung hanya dari train set.
   - **Feature engineering**: `Glucose_Insulin_Ratio`, `BMI_Category`, `Age_Group`,
     `Glucose_BMI_Interaction` (berbasis domain knowledge medis).
3. **Modeling & Evaluation**
   - **4 model**: Logistic Regression, Random Forest, XGBoost, **SVM**.
   - Tuning hyperparameter dengan **Optuna (Bayesian/TPE Optimization)**.
   - Perbandingan sistematis penanganan imbalance: **class_weight vs SMOTE** (bagian dari search
     space Optuna).
   - Evaluasi test set: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
4. **Interpretation** — **SHAP** (global feature importance) **+ LIME** (penjelasan per-pasien),
   dua pendekatan explainable AI yang saling melengkapi.
5. **Deployment** — aplikasi Streamlit multi-halaman, termasuk **prediksi batch via upload CSV**.

## ⚠️ Catatan Metodologi Penting: Data Leakage

Eksperimen awal memakai imputasi *median per kelas Outcome*, yang menghasilkan skor ROC-AUC ~0.94
— namun ini **data leakage** (memakai label target untuk mengisi fitur, padahal label tersebut
belum diketahui saat prediksi pasien baru). Setelah pipeline diperbaiki agar **leak-free** (imputer
di-*fit* hanya pada train, split dilakukan sebelum preprocessing), skor turun ke rentang realistis
**ROC-AUC 0.80–0.85**. Detail perbandingan ada di `notebooks/01_eda.ipynb` dan
`src/imputation_comparison.py`.

## 🏆 Hasil Model Terbaik (Test Set, Pipeline Leak-Free)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** ⭐ (CV) | 0.734 | 0.594 | 0.717 | 0.650 | 0.799 |
| XGBoost | 0.753 | 0.603 | 0.830 | 0.698 | **0.807** |
| SVM | 0.760 | 0.700 | 0.528 | 0.602 | 0.806 |
| Logistic Regression | 0.721 | 0.578 | 0.698 | 0.632 | 0.800 |

*Model terbaik dipilih berdasarkan CV ROC-AUC selama tuning Optuna (Random Forest, 0.851); pada
test set performa keempat model relatif kompetitif satu sama lain.*

**Fitur paling berpengaruh (SHAP):** `Glucose_BMI_Interaction` (hasil feature engineering),
`Glucose`, `Insulin` — memvalidasi bahwa interaksi dua faktor risiko metabolik utama menangkap
sinyal risiko lebih baik dibanding fitur tunggal.

## ⚙️ Cara Menjalankan

```bash
# 1. Clone repository & install dependencies
git clone <repo-url>
cd capstone-project-data-mining
pip install -r requirements.txt

# 2. (Opsional) Jalankan ulang pipeline dari awal
python src/data_preprocessing.py       # leak-free: split -> KNN impute -> outlier -> feature engineering
python src/imputation_comparison.py    # (opsional) reproduksi perbandingan strategi imputasi
python src/train_model.py              # Optuna tuning: LR, RF, XGBoost, SVM
python src/evaluate_model.py
python src/model_interpretation.py     # SHAP + LIME
python src/eda_visuals.py

# 3. Jalankan aplikasi Streamlit
streamlit run app/app.py
```

## 📦 Libraries Utama

Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn (SMOTE), Optuna, Matplotlib, Seaborn,
Plotly, SHAP, LIME, Streamlit, Joblib.

## ⚠️ Disclaimer

Aplikasi ini dibuat untuk keperluan akademik (UAS Pembelajaran Mesin) dan **tidak dimaksudkan
sebagai alat diagnosis medis resmi**. Selalu konsultasikan kondisi kesehatan dengan tenaga medis
profesional.

## 👤 Author

Muhammad Cealvin_A11.2024.16020 — Fakultas Ilmu Komputer, Universitas Dian Nuswantoro

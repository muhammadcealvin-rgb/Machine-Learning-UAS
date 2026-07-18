"""
5_Dokumentasi.py (v2) - Penjelasan dataset, metodologi, dan cara penggunaan aplikasi
"""

import streamlit as st

st.set_page_config(page_title="Dokumentasi", page_icon="📄", layout="wide")
st.title("📄 Dokumentasi Proyek")

st.subheader("1. Dataset")
st.markdown("""
**Nama:** Pima Indians Diabetes Dataset
**Sumber:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), diakses via
[GitHub mirror](https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv)
**Jumlah data:** 768 pasien wanita keturunan Pima Indian, usia ≥ 21 tahun
**Jumlah fitur asli:** 8 fitur prediktor + 1 target biner, ditambah **4 fitur hasil feature engineering**

| Fitur | Deskripsi |
|---|---|
| Pregnancies | Jumlah kehamilan |
| Glucose | Kadar glukosa plasma (2 jam oral glucose tolerance test) |
| BloodPressure | Tekanan darah diastolik (mmHg) |
| SkinThickness | Ketebalan lipatan kulit triceps (mm) |
| Insulin | Kadar insulin serum 2-jam (mu U/ml) |
| BMI | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Skor riwayat genetik diabetes keluarga |
| Age | Usia (tahun) |
| Outcome | Target: 0 = tidak diabetes, 1 = diabetes |
| **Glucose_Insulin_Ratio*** | Proxy sederhana resistensi insulin (mirip ide dasar HOMA-IR) |
| **BMI_Category*** | Kategori BMI standar WHO (encoded ordinal 0-3) |
| **Age_Group*** | Kelompok usia (encoded ordinal 0-3) |
| **Glucose_BMI_Interaction*** | Interaksi Glucose x BMI — fitur paling berpengaruh menurut SHAP |

*\\* = hasil feature engineering*
""")

st.subheader("2. Metodologi (Alur Kerja)")
st.markdown("""
1. **Problem Definition & Data Acquisition** — menentukan problem statement skrining diabetes dan
   mengumpulkan dataset publik.
2. **EDA & Preprocessing (leak-free)**
   - Nilai `0` pada kolom fisiologis ditandai sebagai missing value tersembunyi.
   - **Split train (60%) / val (20%) / test (20%) dilakukan SEBELUM imputasi & outlier handling**
     untuk mencegah data leakage.
   - Imputasi memakai **KNN Imputer (k=5)**, di-*fit* hanya pada data train lalu diterapkan ke
     val/test — dipilih setelah membandingkan 3 strategi imputasi secara adil (leak-free CV).
   - Outlier di-capping dengan IQR yang batasnya dihitung hanya dari train.
   - **Feature engineering**: 4 fitur turunan berbasis domain knowledge medis.
3. **Modeling & Evaluation**
   - **4 model**: Logistic Regression, Random Forest, XGBoost, **SVM**.
   - Tuning hyperparameter dengan **Optuna (Bayesian/TPE Optimization)**.
   - Penanganan class imbalance dibandingkan secara sistematis: **class_weight vs SMOTE**
     (dipilih otomatis sebagai bagian dari search space Optuna).
   - Evaluasi test set: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
4. **Interpretation** — **SHAP** (global feature importance) **dan LIME** (penjelasan per-pasien),
   dua pendekatan explainable AI yang saling melengkapi.
5. **Deployment** — aplikasi Streamlit multi-halaman, termasuk **prediksi batch via upload CSV**.
""")

st.subheader("3. Cara Menggunakan Aplikasi")
st.markdown("""
- **📊 Dashboard EDA** — jelajahi data secara interaktif (termasuk fitur hasil engineering),
  gunakan filter usia dan status diabetes di sidebar.
- **🔮 Model Demo** —
  - Tab *Prediksi 1 Pasien*: masukkan data klinis, klik **Prediksi Sekarang**.
  - Tab *Prediksi Batch*: unggah CSV berisi banyak pasien sekaligus, unduh hasilnya.
- **📈 Evaluasi Model** — bandingkan performa 4 model pada test set, ROC curve, confusion matrix.
- **🧠 Interpretasi Model** — SHAP (pola global) dan insight bisnis dari feature importance.
""")

st.subheader("4. Struktur Repository")
st.code("""
capstone-project-data-mining/
├── data/{raw,processed,external}/
├── notebooks/{01_eda, 02_modeling, 03_interpretation}.ipynb
├── src/
│   ├── utils.py                  # Konstanta & feature engineering
│   ├── data_preprocessing.py     # Pipeline leak-free (split -> impute -> outlier -> FE)
│   ├── imputation_comparison.py  # Perbandingan strategi imputasi (leak-free)
│   ├── eda_visuals.py
│   ├── train_model.py            # Optuna tuning, 4 model, SMOTE vs class_weight
│   ├── evaluate_model.py
│   └── model_interpretation.py   # SHAP + LIME
├── models/                        # 4 model + knn_imputer.pkl + outlier_bounds.pkl
├── app/{app.py, pages/}          # 5 halaman Streamlit
├── reports/
├── requirements.txt
└── README.md
""", language="text")

st.subheader("5. Catatan Metodologi Penting")
st.info("""
**Mengapa skor model "hanya" ~0.80-0.85 ROC-AUC, bukan 0.94?**

Eksperimen awal memakai imputasi *median per kelas Outcome*, yang menghasilkan skor ROC-AUC 0.94
— namun ini adalah **data leakage** (memakai label target untuk mengisi fitur, padahal label
tersebut belum diketahui saat prediksi pasien baru). Setelah diperbaiki dengan pipeline **leak-free**
(imputer di-*fit* hanya di train), skor turun ke rentang 0.80-0.85 yang **lebih realistis dan
dapat dipercaya** untuk deployment produksi.
""")

st.subheader("6. Batasan & Catatan Etis")
st.warning("""
- Aplikasi ini dibuat untuk **keperluan akademik** dan tidak boleh digunakan sebagai pengganti
  diagnosis medis resmi.
- Dataset berasal dari populasi spesifik (wanita keturunan Pima Indian), sehingga generalisasi ke
  populasi lain (pria, etnis lain) perlu divalidasi ulang.
- Model dapat mengandung bias yang melekat pada data historis; hasil prediksi sebaiknya selalu
  dikonfirmasi oleh tenaga medis profesional.
""")

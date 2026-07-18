"""
app.py - Halaman utama (Home) aplikasi Streamlit
Capstone Project: Prediksi Risiko Diabetes
"""

import streamlit as st
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent  # root repo (capstone-project-data-mining/)

st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Aplikasi Prediksi Risiko Diabetes")
st.caption("Capstone Project — Ujian Akhir Semester Pembelajaran Mesin")

st.markdown("""
Aplikasi ini adalah hasil akhir dari sebuah **pipeline Machine Learning end-to-end**:
mulai dari akuisisi data, eksplorasi data, preprocessing, pemodelan, evaluasi,
interpretasi model, hingga deployment — untuk membantu **skrining awal risiko diabetes**
berdasarkan data klinis dasar pasien.

Gunakan menu di **sidebar kiri** untuk menjelajahi setiap tahap proyek:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    - 📊 **Dashboard EDA** — eksplorasi data interaktif
    - 🔮 **Model Demo** — coba prediksi risiko diabetes pasien baru
    - 📈 **Evaluasi Model** — metrik & perbandingan performa model
    """)
with col2:
    st.markdown("""
    - 🧠 **Interpretasi Model** — SHAP feature importance & insight bisnis
    - 📄 **Dokumentasi** — metodologi, dataset, dan cara pakai aplikasi
    """)

st.divider()

st.subheader("Ringkasan Proyek")

try:
    df = pd.read_csv(BASE_DIR / "data" / "processed" / "diabetes_clean.csv")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data Pasien", f"{len(df)}")
    c2.metric("Jumlah Fitur", "8 asli + 4 hasil engineering")
    c3.metric("Kasus Diabetes", f"{int(df['Outcome'].sum())} ({df['Outcome'].mean()*100:.1f}%)")
    c4.metric("Kasus Non-Diabetes", f"{int((df['Outcome']==0).sum())} ({(1-df['Outcome'].mean())*100:.1f}%)")
except FileNotFoundError:
    st.warning("Data belum ditemukan. Jalankan `src/data_preprocessing.py` terlebih dahulu.")

st.divider()
st.subheader("Problem Statement")
st.markdown("""
Diabetes melitus tipe 2 sering tidak terdiagnosis pada tahap awal karena gejalanya yang
tidak spesifik, padahal deteksi dini sangat penting untuk mencegah komplikasi jangka panjang
(penyakit jantung, gagal ginjal, kebutaan). Fasilitas kesehatan primer, khususnya di daerah
dengan keterbatasan tenaga medis, membutuhkan alat bantu skrining cepat berbasis data klinis
dasar yang mudah diukur (jumlah kehamilan, kadar glukosa, tekanan darah, BMI, dll.) tanpa
memerlukan pemeriksaan laboratorium lanjutan yang mahal.

**Tujuan bisnis/analisis:** Membangun model klasifikasi yang dapat memprediksi probabilitas
seorang pasien berisiko diabetes berdasarkan data klinis dasar, sebagai alat bantu **skrining
awal** (bukan pengganti diagnosis dokter).

**Metrik kesuksesan:** ROC-AUC dan Recall tinggi diprioritaskan (meminimalkan kasus diabetes
yang terlewat / false negative), dengan tetap menjaga Precision agar tidak terlalu banyak
false alarm.

**Sumber data:** [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv)
(768 observasi, National Institute of Diabetes and Digestive and Kidney Diseases, AS).
""")

st.info("⚠️ **Disclaimer:** Aplikasi ini dibuat untuk keperluan akademik (tugas UAS) dan "
        "TIDAK dimaksudkan sebagai alat diagnosis medis resmi. Selalu konsultasikan kondisi "
        "kesehatan dengan tenaga medis profesional.")

"""
4_Interpretasi_Model.py - Penjelasan model dan insights bisnis
"""

import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="Interpretasi Model", page_icon="🧠", layout="wide")
st.title("🧠 Interpretasi Model (SHAP)")

with open(BASE_DIR / "models" / "best_model_name.txt", encoding="utf-8") as f:
    best_model_name = f.read().strip()
MODEL_DISPLAY_NAMES = {
    "svm": "SVM", "xgboost": "XGBoost", "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}
model_display = MODEL_DISPLAY_NAMES.get(best_model_name, best_model_name)

st.markdown(f"""
Karena model terbaik (**{model_display}**) bersifat kurang transparan secara langsung, kami menggunakan
**SHAP (SHapley Additive exPlanations)** untuk menjelaskan kontribusi tiap fitur terhadap prediksi
risiko diabetes.
""")

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Global Feature Importance (SHAP Summary)")
    st.image(str(BASE_DIR / "reports" / "shap_summary.png"), use_container_width=True)

with col2:
    st.subheader("Ranking Fitur")
    importance = pd.read_csv(BASE_DIR / "reports" / "feature_importance.csv")
    st.dataframe(importance, use_container_width=True, hide_index=True)
    st.image(str(BASE_DIR / "reports" / "feature_importance.png"), use_container_width=True)

st.divider()
st.subheader("🔬 Interpretasi Lokal (LIME) — Pembanding SHAP")
st.markdown("""
Jika SHAP di atas menunjukkan pola **global** (rata-rata seluruh data), **LIME** menjelaskan
prediksi untuk **satu pasien spesifik** dengan mendekati model kompleks menggunakan model linear
sederhana di sekitar titik data tersebut.
""")

lime_files = {
    "true_positive_diabetes": "✅ Contoh: Pasien diabetes, terdeteksi benar oleh model",
    "true_negative_sehat": "✅ Contoh: Pasien sehat, terdeteksi benar oleh model",
    "false_negative_terlewat": "⚠️ Contoh: Pasien diabetes, TERLEWAT oleh model (false negative)",
}

lime_tabs = st.tabs(list(lime_files.values()))
for tab, (key, _) in zip(lime_tabs, lime_files.items()):
    with tab:
        html_path = BASE_DIR / "reports" / f"lime_{key}.html"
        if html_path.exists():
            st.components.v1.html(html_path.read_text(encoding="utf-8"), height=400, scrolling=True)
        else:
            st.info("Penjelasan LIME untuk contoh ini belum tersedia. Jalankan `src/lime_interpretation.py`.")

st.divider()
st.subheader("💡 Insight Bisnis")
st.markdown("""
1. **Insulin dan Glucose** adalah dua faktor paling dominan yang mendorong prediksi risiko diabetes —
   sejalan dengan mekanisme medis diabetes tipe 2 (resistensi insulin & hiperglikemia).
2. **BMI, DiabetesPedigreeFunction (riwayat keturunan), dan Age** memberi kontribusi sedang — 
   mengonfirmasi bahwa faktor gaya hidup dan genetik tetap relevan meski bukan yang paling dominan.
3. **BloodPressure dan Pregnancies** memiliki pengaruh relatif kecil terhadap prediksi model 
   dibanding fitur metabolik langsung.
4. **Rekomendasi untuk fasilitas kesehatan:** prioritaskan pemeriksaan kadar **Glucose** dan 
   **Insulin** sebagai alat skrining awal paling efisien secara biaya, sebelum melakukan pemeriksaan 
   lanjutan yang lebih mahal.
5. Transparansi SHAP membantu tenaga medis memvalidasi bahwa keputusan model **konsisten dengan 
   pengetahuan domain**, bukan sekadar pola statistik yang tidak masuk akal secara klinis.
""")

st.info("Untuk melihat interpretasi tiap prediksi individual (waterfall plot per pasien), "
        "lihat notebook `notebooks/03_interpretation.ipynb`.")

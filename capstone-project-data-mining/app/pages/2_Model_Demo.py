"""
2_Model_Demo.py (v2) - Interface input data pasien -> prediksi dari model terbaik.
Menerapkan pipeline lengkap: outlier capping (batas dari train) + feature engineering,
identik dengan proses saat training, agar tidak ada train-serving skew.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "src"))
from utils import engineer_features, mark_missing, FEATURE_COLS, FEATURE_COLS_V2  # noqa: E402

st.set_page_config(page_title="Model Demo", page_icon="🔮", layout="wide")
st.title("🔮 Model Demo — Prediksi Risiko Diabetes")
st.markdown("Masukkan data klinis pasien untuk mendapatkan prediksi risiko diabetes dari model "
            "terbaik. Input akan melalui pipeline yang **identik** dengan proses training "
            "(penanganan nilai hilang + outlier capping + feature engineering) agar prediksi konsisten.")


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
    bounds = joblib.load(BASE_DIR / "models" / "outlier_bounds.pkl")
    imputer = joblib.load(BASE_DIR / "models" / "knn_imputer.pkl")
    with open(BASE_DIR / "models" / "best_model_name.txt", encoding="utf-8") as f:
        name = f.read().strip()
    return model, bounds, imputer, name


model, outlier_bounds, knn_imputer, model_name = load_artifacts()


def preprocess_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Terapkan pipeline IDENTIK dengan training: mark_missing -> KNN imputation ->
    outlier capping (batas dari train) -> feature engineering. Urutan ini harus sama persis
    dengan src/data_preprocessing.py, kalau tidak prediksi akan salah (train-serving skew)."""
    df = raw_df.copy()
    df = mark_missing(df)  # nilai 0 pada kolom fisiologis -> NaN (sesuai definisi dataset asli)
    df = pd.DataFrame(knn_imputer.transform(df), columns=FEATURE_COLS, index=df.index)
    for col, (lower, upper) in outlier_bounds.items():
        df[col] = df[col].clip(lower, upper)
    df = engineer_features(df)
    return df[FEATURE_COLS_V2]


tab1, tab2 = st.tabs(["🧍 Prediksi 1 Pasien", "📁 Prediksi Batch (Upload CSV)"])

# ======================= TAB 1: Prediksi Individual =======================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", 0, 20, 2,
                                       help="Jumlah total kehamilan yang pernah dialami pasien.")
        glucose = st.number_input("Kadar Glukosa Plasma (mg/dL)", 0, 250, 120,
                                   help="Kadar glukosa plasma 2 jam setelah oral glucose tolerance test. "
                                        "Normal: <140 mg/dL, Prediabetes: 140-199, Diabetes: ≥200.")
        blood_pressure = st.number_input("Tekanan Darah Diastolik (mmHg)", 0, 150, 70,
                                          help="Tekanan darah diastolik. Normal: 60-80 mmHg.")
        skin_thickness = st.number_input("Ketebalan Lipatan Kulit Triceps (mm)", 0, 100, 20,
                                          help="Indikator tidak langsung dari lemak tubuh subkutan.")
    with col2:
        insulin = st.number_input("Kadar Insulin Serum 2-jam (mu U/ml)", 0, 900, 80,
                                   help="Kadar insulin serum 2 jam setelah tes toleransi glukosa. "
                                        "Nilai tinggi dapat mengindikasikan resistensi insulin.")
        bmi = st.number_input("Body Mass Index (kg/m²)", 0.0, 70.0, 28.0, step=0.1,
                               help="BMI = berat(kg) / tinggi(m)². Normal: 18.5-25, Overweight: 25-30, Obese: ≥30.")
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.4, step=0.01,
                               help="Skor yang merepresentasikan riwayat genetik/keturunan diabetes dalam keluarga. "
                                    "Semakin tinggi, semakin kuat riwayat keturunan.")
        age = st.number_input("Usia (tahun)", 15, 100, 33)

    st.divider()

    if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
        raw_input = pd.DataFrame([{
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age
        }])
        input_fe = preprocess_input(raw_input)

        proba = model.predict_proba(input_fe)[0, 1]
        pred = model.predict(input_fe)[0]

        c1, c2 = st.columns([1, 2])
        with c1:
            if pred == 1:
                st.error("### ⚠️ Berisiko Diabetes")
            else:
                st.success("### ✅ Risiko Rendah")
            st.metric("Probabilitas Diabetes", f"{proba*100:.1f}%")
        with c2:
            st.progress(min(max(proba, 0.0), 1.0))
            if proba >= 0.7:
                st.warning("Risiko **tinggi**. Disarankan pemeriksaan lanjutan ke dokter/laboratorium.")
            elif proba >= 0.4:
                st.info("Risiko **sedang**. Perhatikan pola hidup dan lakukan pemeriksaan rutin.")
            else:
                st.info("Risiko **rendah** berdasarkan data yang dimasukkan.")

        st.caption(f"Model yang digunakan: **{model_name}**. Prediksi ini hanya untuk skrining awal, "
                   "bukan diagnosis medis final.")

        with st.expander("Lihat fitur hasil feature engineering yang dipakai model"):
            st.dataframe(input_fe, use_container_width=True)

# ======================= TAB 2: Prediksi Batch =======================
with tab2:
    st.markdown("""
    Unggah file CSV berisi banyak pasien sekaligus untuk mendapatkan prediksi secara batch.
    Kolom yang dibutuhkan (persis nama berikut):
    """)
    st.code(", ".join(FEATURE_COLS), language="text")

    template_csv = pd.DataFrame([{
        "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
        "Insulin": 80, "BMI": 28.0, "DiabetesPedigreeFunction": 0.4, "Age": 33
    }]).to_csv(index=False)
    st.download_button("📥 Download Template CSV", template_csv, file_name="template_pasien.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV data pasien", type=["csv"])
    if uploaded is not None:
        try:
            batch_raw = pd.read_csv(uploaded)
            missing_cols = set(FEATURE_COLS) - set(batch_raw.columns)
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan di file: {missing_cols}")
            else:
                batch_fe = preprocess_input(batch_raw[FEATURE_COLS])
                probas = model.predict_proba(batch_fe)[:, 1]
                preds = model.predict(batch_fe)

                result = batch_raw[FEATURE_COLS].copy()
                result["Probabilitas_Diabetes"] = (probas * 100).round(1)
                result["Prediksi"] = pd.Series(preds).map({0: "Tidak Diabetes", 1: "Diabetes"})

                st.success(f"Berhasil memprediksi {len(result)} pasien.")
                st.dataframe(result, use_container_width=True)

                c1, c2 = st.columns(2)
                c1.metric("Jumlah Diprediksi Diabetes", int((preds == 1).sum()))
                c2.metric("Jumlah Diprediksi Tidak Diabetes", int((preds == 0).sum()))

                st.download_button("📥 Download Hasil Prediksi (CSV)", result.to_csv(index=False),
                                    file_name="hasil_prediksi_diabetes.csv", mime="text/csv",
                                    use_container_width=True)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")

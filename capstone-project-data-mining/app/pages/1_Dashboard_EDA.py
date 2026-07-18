"""
1_Dashboard_EDA.py - Dashboard EDA interaktif
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="Dashboard EDA", page_icon="📊", layout="wide")
st.title("📊 Dashboard EDA — Eksplorasi Data")

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "processed" / "diabetes_clean.csv")

df = load_data()

st.sidebar.header("Filter Data")
age_range = st.sidebar.slider("Rentang Usia", int(df.Age.min()), int(df.Age.max()),
                               (int(df.Age.min()), int(df.Age.max())))
outcome_filter = st.sidebar.multiselect("Status Diabetes", options=[0, 1],
                                         default=[0, 1],
                                         format_func=lambda x: "Diabetes" if x == 1 else "Tidak Diabetes")

filtered = df[(df.Age.between(*age_range)) & (df.Outcome.isin(outcome_filter))]
st.sidebar.metric("Jumlah data setelah filter", len(filtered))

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Distribusi Fitur", "Korelasi", "Data Mentah"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Pasien", len(filtered))
    c2.metric("Rata-rata Glucose", f"{filtered.Glucose.mean():.1f}")
    c3.metric("Rata-rata BMI", f"{filtered.BMI.mean():.1f}")
    c4.metric("% Diabetes", f"{filtered.Outcome.mean()*100:.1f}%")

    fig = px.pie(filtered, names=filtered["Outcome"].map({0: "Tidak Diabetes", 1: "Diabetes"}),
                 title="Distribusi Kelas Target", color_discrete_sequence=["#4C72B0", "#DD8452"])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    feature = st.selectbox("Pilih fitur untuk dilihat distribusinya", 
                            ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                             "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                             "Glucose_Insulin_Ratio", "Glucose_BMI_Interaction",
                             "BMI_Category", "Age_Group"],
                            help="4 fitur terakhir adalah hasil feature engineering (lihat notebook 01_eda)")
    fig = px.histogram(filtered, x=feature, color=filtered["Outcome"].map({0: "Tidak Diabetes", 1: "Diabetes"}),
                        marginal="box", barmode="overlay", opacity=0.7,
                        color_discrete_sequence=["#4C72B0", "#DD8452"],
                        title=f"Distribusi {feature} berdasarkan Status Diabetes")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(filtered, x="Age", y="Glucose",
                       color=filtered["Outcome"].map({0: "Tidak Diabetes", 1: "Diabetes"}),
                       color_discrete_sequence=["#4C72B0", "#DD8452"],
                       title="Age vs Glucose", opacity=0.7)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    corr_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                 "Glucose_Insulin_Ratio", "Glucose_BMI_Interaction",
                 "BMI_Category", "Age_Group", "Outcome"]
    corr = filtered[corr_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Heatmap Korelasi Fitur (termasuk hasil feature engineering)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** `Glucose_BMI_Interaction` (fitur hasil engineering) dan `Glucose` "
                "memiliki korelasi tertinggi terhadap `Outcome` — konsisten dengan hasil SHAP di "
                "halaman Interpretasi Model.")

with tab4:
    st.dataframe(filtered, use_container_width=True)
    st.download_button("Download data terfilter (CSV)", filtered.to_csv(index=False),
                        file_name="diabetes_filtered.csv", mime="text/csv")

"""
3_Evaluasi_Model.py - Tampilan metrik dan visualisasi evaluasi model
"""

import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="Evaluasi Model", page_icon="📈", layout="wide")
st.title("📈 Evaluasi Model")
st.caption("5 model dibandingkan • tuning dengan Optuna (Bayesian Optimization) • SMOTE untuk menangani class imbalance")

comparison = pd.read_csv(BASE_DIR / "reports" / "model_comparison.csv")

st.subheader("Tabel Perbandingan Performa (Test Set)")
st.dataframe(
    comparison.style.highlight_max(subset=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
                                    color="#c6f6d5"),
    use_container_width=True
)

best_row = comparison.sort_values("roc_auc", ascending=False).iloc[0]
st.success(f"🏆 Model terbaik: **{best_row['model']}** dengan ROC-AUC **{best_row['roc_auc']:.3f}**")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{best_row['accuracy']*100:.1f}%")
c2.metric("Precision", f"{best_row['precision']*100:.1f}%")
c3.metric("Recall", f"{best_row['recall']*100:.1f}%")
c4.metric("F1-Score", f"{best_row['f1_score']*100:.1f}%")
c5.metric("ROC-AUC", f"{best_row['roc_auc']:.3f}")

st.divider()
st.subheader("ROC Curve — Perbandingan Semua Model")
st.image(str(BASE_DIR / "reports" / "roc_curve_comparison.png"), use_container_width=True)

st.divider()
st.subheader("Confusion Matrix per Model")
n_models = len(comparison)
cols = st.columns(min(n_models, 3))
for i, name in enumerate(comparison["model"]):
    img_path = BASE_DIR / "reports" / f"confusion_matrix_{name}.png"
    if img_path.exists():
        cols[i % len(cols)].image(str(img_path), caption=name, use_container_width=True)

st.divider()
st.subheader("Penjelasan Metrik")
st.markdown("""
- **Accuracy**: proporsi prediksi benar secara keseluruhan.
- **Precision**: dari semua pasien yang diprediksi diabetes, berapa persen yang benar-benar diabetes
  (penting untuk menghindari false alarm).
- **Recall**: dari semua pasien yang sebenarnya diabetes, berapa persen yang berhasil terdeteksi model
  (penting agar kasus diabetes tidak terlewat).
- **F1-Score**: rata-rata harmonik Precision dan Recall, berguna saat data tidak seimbang.
- **ROC-AUC**: kemampuan model membedakan kelas diabetes vs tidak, di seluruh ambang batas keputusan
  (0.5 = acak, 1.0 = sempurna).
""")

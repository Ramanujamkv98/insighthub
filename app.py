# -------------------------------------------------------------
# InsightHub – Landing Page + Premium Analytics UI
# -------------------------------------------------------------

import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="InsightHub - AI Data Cleaning",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.hero-container {
    text-align: center;
    padding-top: 60px;
    padding-bottom: 20px;
}

.hero-title {
    font-size: 46px;
    font-weight: 700;
    color: #ffffff;
    padding-bottom: 14px;
}

.hero-subtitle {
    font-size: 20px;
    font-weight: 300;
    color: #9CA3AF;
    margin-bottom: 30px;
}

.cta-button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    padding: 14px 28px;
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    margin-top: 18px;
    box-shadow: 0px 0px 20px rgba(0,153,255,0.35);
}
.cta-button:hover {
    transform: scale(1.04);
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.35);
    margin-bottom: 24px;
}

.metric-card {
    background: #1f2937;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    box-shadow: inset 0px 0px 10px rgba(255,255,255,0.05);
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #4ade80;
}

.metric-label {
    font-size: 14px;
    color: #d1d5db;
}

canvas {
    border-radius: 16px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# PLOTLY CLEAN THEME
# -------------------------------------------------------------
pio.templates.default = "plotly_dark"
CUSTOM_PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E1E1E1", size=14),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
)


# -------------------------------------------------------------
# LANDING PAGE (Hero + WebGL Animation)
# -------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":

    st.markdown("<div class='hero-container'>", unsafe_allow_html=True)

    st.markdown("<div class='hero-title'>See Your Data Clean Itself.</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>An interactive visualization showing how AI transforms chaos into clarity.</div>", unsafe_allow_html=True)

    # WebGL – Three.js shader animation visualizing "messy → clean" data
    st.components.v1.html("""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <div id="container"></div>
        <script>
            const container = document.getElementById('container');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(70, window.innerWidth / 400, 0.1, 1000);
            camera.position.z = 3;

            const renderer = new THREE.WebGLRenderer({ alpha: true });
            renderer.setSize(window.innerWidth * 0.9, 400);
            container.appendChild(renderer.domElement);

            const geometry = new THREE.BufferGeometry();
            const count = 1500;
            const positions = new Float32Array(count * 3);

            for (let i = 0; i < count; i++) {
                positions[i * 3] = (Math.random() - 0.5) * 4;
                positions[i * 3 + 1] = (Math.random() - 0.5) * 4;
                positions[i * 3 + 2] = (Math.random() - 0.5) * 4;
            }

            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

            const material = new THREE.PointsMaterial({
                size: 0.035,
                color: 0x66ccff,
                transparent: true
            });

            const points = new THREE.Points(geometry, material);
            scene.add(points);

            let clean = false;

            function animate() {
                requestAnimationFrame(animate);

                if (clean) {
                    geometry.attributes.position.array.forEach((val, idx) => {
                        geometry.attributes.position.array[idx] *= 0.97;
                    });
                    geometry.attributes.position.needsUpdate = true;
                } else {
                    points.rotation.x += 0.002;
                    points.rotation.y += 0.002;
                }

                renderer.render(scene, camera);
            }
            animate();

            container.addEventListener('click', () => { clean = !clean; });
        </script>
    """, height=420)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Enter InsightHub", key="enter", help="Start analyzing your data"):
        st.session_state.page = "app"
        st.experimental_rerun()

    st.stop()


# -------------------------------------------------------------
# ANALYTICS APP (Your premium UI)
# -------------------------------------------------------------
# --------- Utility functions -----------

def load_data(uploaded_file: BytesIO):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def clean_dataframe(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in df.columns:
        if any(x in col.lower() for x in ["date", "time", "day"]):
            try: df[col] = pd.to_datetime(df[col])
            except: pass

    for col in df.select_dtypes("object"):
        cleaned = (
            df[col].astype(str)
            .str.replace(",", "")
            .str.replace("$", "")
            .str.strip()
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() > 0.4:
            df[col] = numeric

    df = df.loc[:, ~df.columns.duplicated()]  # <--- FIX FOR NARWHALS
    return df

def detect_date_column(df):
    for c in df.columns:
        if "date" in c.lower():
            return c
    return None

def detect_target_column(df):
    for c in df.select_dtypes([np.number]).columns:
        if any(k in c.lower() for k in ["revenue", "sales", "amount"]):
            return c
    return df.select_dtypes([np.number]).columns[-1]

def compute_kpis(df, target, date_col):
    series = df[target]
    return {
        "Total": float(series.sum()),
        "Average": float(series.mean()),
        "Max": float(series.max()),
        "Min": float(series.min())
    }


# -------------------------------------------------------------
# APP UI
# -------------------------------------------------------------

st.title("📊 InsightHub – Data Analytics Dashboard")

uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

if uploaded:
    df_raw = load_data(uploaded)
    df = clean_dataframe(df_raw)

    st.success("File uploaded and cleaned.")

    st.dataframe(df.head(50), use_container_width=True)

    date_col = detect_date_column(df)
    target_col = detect_target_column(df)

    kpis = compute_kpis(df, target_col, date_col)

    st.subheader("📌 KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{kpis['Total']:,.2f}")
    c2.metric("Average", f"{kpis['Average']:,.2f}")
    c3.metric("Max", f"{kpis['Max']:,.2f}")
    c4.metric("Min", f"{kpis['Min']:,.2f}")

    st.subheader("📈 Histogram")
    num_cols = df.select_dtypes([np.number]).columns.tolist()
    col = st.selectbox("Select column", num_cols)

    fig = px.histogram(df, x=col, nbins=30, opacity=0.85)
    fig.update_layout(**CUSTOM_PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a file to begin.")


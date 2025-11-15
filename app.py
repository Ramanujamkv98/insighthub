# -------------------------------------------------------------
# InsightHub – GPT-Powered Auto-Clean + Auto-Visualize (OpenAI v1.0+)
# -------------------------------------------------------------

import os
import json
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from openai import OpenAI

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="InsightHub - GPT Data Analyst",
    page_icon="📊",
    layout="wide",
)

# Load API key
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Plotly dark theme
pio.templates.default = "plotly_dark"
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E1E1E1", size=14),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
)

# -------------------------------------------------------------
# SIMPLE CSS
# -------------------------------------------------------------
st.markdown(
"""
<style>
html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.card {
    background: #0f172a;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #1e293b;
    box-shadow: 0px 0px 18px rgba(30,41,59,0.35);
    margin-bottom: 20px;
}
.stButton > button {
    border-radius: 8px;
    background: linear-gradient(90deg, #4f46e5, #0ea5e9);
    color: white;
    padding: 0.6rem 1.3rem;
    font-weight: 600;
    border: none;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# GPT FUNCTIONS
# -------------------------------------------------------------

def build_dataframe_spec(df: pd.DataFrame, max_rows=20) -> str:
    """Convert DataFrame into JSON summary for GPT."""
    spec = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": df.isna().sum().to_dict(),
        "sample_rows": df.head(max_rows).to_dict(orient="records"),
    }
    return json.dumps(spec)


SYSTEM_PROMPT = """
You are an elite Python data analyst.
Your job is to:
1. Analyze a dataset (schema + sample rows provided).
2. Write CLEANING CODE in Python inside:  def clean_df(df): ...
3. Write VISUALIZATION CODE inside:       def create_charts(df): ...
4. Return a SHORT INSIGHT SUMMARY.

Rules:
- Return ONLY valid JSON.
- cleaning_code MUST define clean_df(df) and return df.
- chart_code MUST define create_charts(df) and return a list named figures.
- Plotting library: plotly ONLY.
- NO seaborn, NO matplotlib, NO sklearn, NO external libs.
"""

def call_gpt(df):
    """Send structure + sample rows to GPT."""
    spec = build_dataframe_spec(df)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": spec},
        ]
    )
    content = response.choices[0].message.content


    # Remove accidental code fences
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]

    return json.loads(cleaned)


def exec_cleaning_code(df, code):
    """Execute GPT cleaning code safely."""
    namespace = {
        "pd": pd,
        "np": np,
    }
    local_env = {}
    exec(code, namespace, local_env)

    if "clean_df" not in local_env:
        raise ValueError("clean_df(df) not found in GPT code.")

    return local_env["clean_df"](df.copy())


def exec_chart_code(df, code):
    """Execute GPT chart code safely."""
    namespace = {
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
    }
    local_env = {}
    exec(code, namespace, local_env)

    if "create_charts" not in local_env:
        raise ValueError("create_charts(df) not found.")

    return local_env["create_charts"](df.copy())

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.markdown("<h1>📊 InsightHub – GPT Auto-EDA</h1>", unsafe_allow_html=True)
st.caption("Upload data → GPT cleans it → GPT analyzes it → GPT builds charts.")

uploaded = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])
rows_to_send = st.sidebar.slider("Sample rows to send to GPT", 10, 80, 25)

run_btn = st.sidebar.button("Run GPT Analysis")

# Session state
if "gpt_result" not in st.session_state:
    st.session_state.gpt_result = None
    st.session_state.cleaned = None


# -------------------------------------------------------------
# LOAD FILE
# -------------------------------------------------------------
if not uploaded:
    st.info("Upload a CSV or Excel file to start.")
    st.stop()

try:
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Cannot read file: {e}")
    st.stop()

# Show sample
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📄 Raw Data Preview")
st.dataframe(df_raw.head(50), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# RUN GPT
# -------------------------------------------------------------
if run_btn:
    try:
        st.session_state.gpt_result = call_gpt(df_raw.head(rows_to_send))
        st.session_state.cleaned = exec_cleaning_code(
            df_raw,
            st.session_state.gpt_result["cleaning_code"]
        )
    except Exception as e:
        st.error(f"GPT Analysis Failed: {e}")
        st.stop()

if st.session_state.gpt_result is None:
    st.info("Click **Run GPT Analysis** to begin.")
    st.stop()

gpt = st.session_state.gpt_result
cleaned = st.session_state.cleaned

# -------------------------------------------------------------
# SHOW CLEANED DATA
# -------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧹 Cleaned Data (GPT Output)")
st.dataframe(cleaned.head(50), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SHOW INSIGHTS
# -------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧠 GPT Insights")
st.markdown(gpt["insights"])
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SHOW GPT CHARTS
# -------------------------------------------------------------
st.subheader("📈 GPT Recommended Visualizations")

try:
    figs = exec_chart_code(cleaned, gpt["chart_code"])
    for f in figs:
        if isinstance(f, go.Figure):
            f.update_layout(**PLOTLY_THEME)
            st.plotly_chart(f, use_container_width=True)
        else:
            st.warning("GPT returned non-Plotly object.")
except Exception as e:
    st.error(f"Chart generation failed: {e}")

# -------------------------------------------------------------
# ADVANCED PANEL
# -------------------------------------------------------------
with st.expander("🔧 View GPT-Generated Code"):
    st.write("### Cleaning Code")
    st.code(gpt["cleaning_code"])
    st.write("### Chart Code")
    st.code(gpt["chart_code"])

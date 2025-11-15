# -------------------------------------------------------------
# InsightHub – GPT AutoClean + AutoVisualize (OpenAI v1.0+)
# JSON VALIDATION + SAFE FALLBACKS
# -------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from openai import OpenAI


# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="InsightHub - GPT Data Analyst",
    page_icon="📊",
    layout="wide",
)

# -------------------------------------------------------------
# LOAD API KEY
# -------------------------------------------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Add OPENAI_API_KEY to your Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------------------------------------
# THEME + CSS
# -------------------------------------------------------------
pio.templates.default = "plotly_dark"

st.markdown("""
<style>
.card {
    background: #0f172a;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #1e293b;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# GPT HELPER FUNCTIONS
# -------------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert Python data analyst.

Your response MUST be ONLY valid JSON in this exact structure:

{
  "cleaning_code": "python code here",
  "chart_code": "python code here",
  "insights": "short text summary"
}

Rules:
- cleaning_code must define:  def clean_df(df): ... return df
- chart_code must define:    def create_charts(df): ... return figures
- Use ONLY pandas, numpy, plotly.
- DO NOT include markdown, DO NOT include text before or after JSON.
"""

def build_df_payload(df, n=25):
    return json.dumps({
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing": df.isna().sum().to_dict(),
        "sample": df.head(n).to_dict(orient="records")
    })


def call_gpt(df):
    """Call GPT and ensure JSON parsing."""
    payload = build_df_payload(df)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ]
    )

    raw = completion.choices[0].message.content.strip()

    # ---------- DEBUG PANEL ----------
    st.session_state["raw_gpt"] = raw

    # Remove code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    # Try JSON parse
    try:
        return json.loads(raw)
    except Exception:
        # Show raw output, don't crash
        st.error("GPT returned invalid JSON. See raw output below.")
        st.code(st.session_state["raw_gpt"])
        raise ValueError("GPT returned non-JSON output.")


def exec_cleaning(df, code):
    env = {"pd": pd, "np": np}
    loc = {}
    exec(code, env, loc)
    return loc["clean_df"](df.copy())


def exec_charts(df, code):
    env = {"pd": pd, "np": np, "px": px, "go": go}
    loc = {}
    exec(code, env, loc)
    return loc["create_charts"](df.copy())


# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.title("📊 InsightHub – GPT Auto EDA")

uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
rows = st.sidebar.slider("Rows sent to GPT", 10, 80, 25)
run_btn = st.sidebar.button("Run GPT Analysis")

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.cleaned = None


# -------------------------------------------------------------
# LOAD FILE
# -------------------------------------------------------------
if not uploaded:
    st.info("Upload a dataset to begin.")
    st.stop()

try:
    df_raw = (
        pd.read_csv(uploaded)
        if uploaded.name.endswith(".csv")
        else pd.read_excel(uploaded)
    )
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📄 Raw Data")
st.dataframe(df_raw.head(50), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# RUN GPT
# -------------------------------------------------------------
if run_btn:
    try:
        gpt_json = call_gpt(df_raw.head(rows))
        cleaned = exec_cleaning(df_raw, gpt_json["cleaning_code"])

        st.session_state.result = gpt_json
        st.session_state.cleaned = cleaned

    except Exception as e:
        st.error(f"GPT Analysis Failed: {e}")
        st.stop()

if st.session_state.result is None:
    st.info("Click **Run GPT Analysis** to start.")
    st.stop()

result = st.session_state.result
cleaned = st.session_state.cleaned


# -------------------------------------------------------------
# CLEANED DATA
# -------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧹 Cleaned Data")
st.dataframe(cleaned.head(50), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# INSIGHTS
# -------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧠 GPT Insights")
st.markdown(result["insights"])
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# CHARTS
# -------------------------------------------------------------
st.subheader("📈 GPT-Generated Charts")

try:
    figs = exec_charts(cleaned, result["chart_code"])
    for fig in figs:
        if isinstance(fig, go.Figure):
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("GPT returned non-figure.")
except Exception as e:
    st.error(f"Chart Error: {e}")

# -------------------------------------------------------------
# RAW GPT OUTPUT (DEBUG)
# -------------------------------------------------------------
with st.expander("🔧 Raw GPT Output (for debugging)"):
    st.code(st.session_state.get("raw_gpt", "No response recorded"))

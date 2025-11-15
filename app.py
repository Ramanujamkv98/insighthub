# -------------------------------------------------------------
# InsightHub – GPT AutoClean + AutoVisualize (Stable Version)
# Currency-safe cleaning + OpenAI v1.0 compatible
# -------------------------------------------------------------

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
# API KEY
# -------------------------------------------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Add OPENAI_API_KEY to Streamlit Secrets")
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
# SYSTEM PROMPT (NEW + FIXED)
# -------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert Python data analyst.

You MUST return ONLY valid JSON in this exact schema:

{
  "cleaning_code": "python code here",
  "chart_code": "python code here",
  "insights": "text summary"
}

Rules for cleaning_code:
- cleaning_code MUST define: def clean_df(df): ... return df
- It MUST remove ALL of these BEFORE converting numeric columns:
      "$", "₹", ",", " ", "%"
- Detect numeric-like columns automatically:
      values like "123", "1,234", "$123.45", "45%" must convert to float.
- Use this universal cleaning snippet inside clean_df:

    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = df[col].str.replace("$", "", regex=False)
        df[col] = df[col].str.replace("₹", "", regex=False)
        df[col] = df[col].str.replace("%", "", regex=False)
        df[col] = df[col].str.strip()
        df[col] = pd.to_numeric(df[col], errors="ignore")

- DO NOT hardcode column names.
- No markdown, no fences, no explanations.

Rules for chart_code:
- MUST define create_charts(df) → return list of Plotly figures.
- Use ONLY Plotly, pandas, numpy.

Respond ONLY with JSON.
"""

# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------

def build_df_payload(df, n=25):
    return json.dumps({
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing": df.isna().sum().to_dict(),
        "sample": df.head(n).to_dict(orient="records"),
    })

def sanitize_for_json(text: str) -> str:
    """
    Fix invalid escape sequences like \$ which break JSON.
    """
    text = text.replace("\\$", "$")
    return text

def call_gpt(df):
    payload = build_df_payload(df)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        temperature=0.2,
    )

    raw = completion.choices[0].message.content.strip()
    st.session_state["raw_gpt"] = raw

    if raw.startswith("```"):
        raw = raw.split("```")[1].strip()

    raw = sanitize_for_json(raw)

    try:
        return json.loads(raw)
    except Exception as e:
        st.error("GPT returned invalid JSON → inspect raw output below.")
        st.code(st.session_state["raw_gpt"])
        raise ValueError(f"JSON parse error: {e}")

def exec_cleaning(df, code):
    env = {"pd": pd, "np": np}
    loc = {}
    exec(code, env, loc)
    return loc["clean_df"](df.copy())

def exec_charts(df, code):
    env = {"pd": pd, "np": np, "px": px, "go": go}
    loc = {}
    exec(code, env, loc)

    figs = loc["create_charts"](df.copy())
    if isinstance(figs, dict):
        return list(figs.values())
    return figs

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.title("📊 InsightHub – GPT Auto EDA")

uploaded = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
rows = st.sidebar.slider("Rows passed to GPT", 10, 80, 25)
run_btn = st.sidebar.button("Run GPT")

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.cleaned = None

if not uploaded:
    st.info("Upload a dataset to begin.")
    st.stop()

# Load dataset
try:
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📄 Raw Data")
st.dataframe(df_raw.head(50), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Run GPT
if run_btn:
    try:
        gpt_json = call_gpt(df_raw.head(rows))
        cleaned = exec_cleaning(df_raw.copy(), gpt_json["cleaning_code"])

        st.session_state.result = gpt_json
        st.session_state.cleaned = cleaned

    except Exception as e:
        st.error(f"GPT Analysis Failed: {e}")
        st.stop()

if st.session_state.result is None:
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
st.subheader("📈 GPT Charts")
try:
    charts = exec_charts(cleaned, result["chart_code"])
    for fig in charts:
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Chart error: {e}")

# -------------------------------------------------------------
# RAW GPT DEBUG
# -------------------------------------------------------------
with st.expander("🔧 Raw GPT Output"):
    st.code(st.session_state.get("raw_gpt", "No output"))

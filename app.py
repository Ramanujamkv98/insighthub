# -------------------------------------------------------------
# InsightHub – GPT AutoClean + AutoVisualize (Robust JSON + OpenAI v1+)
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
    st.error("Please set OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------------------------------------
# THEME + CSS
# -------------------------------------------------------------
pio.templates.default = "plotly_dark"
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E1E1E1", size=14),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
)

st.markdown(
    """
<style>
.card {
    background: #0f172a;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #1e293b;
    margin-top: 1rem;
}
.stButton > button {
    border-radius: 8px;
    background: linear-gradient(90deg, #4f46e5, #0ea5e9);
    color: white;
    padding: 0.6rem 1.2rem;
    border: none;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# GPT HELPERS
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
- Do NOT include markdown, code fences, or explanations outside JSON.
"""

def build_df_payload(df: pd.DataFrame, n: int = 25) -> str:
    """Create a compact JSON summary of the DataFrame for GPT."""
    return json.dumps(
        {
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "missing": df.isna().sum().to_dict(),
            "sample": df.head(n).to_dict(orient="records"),
        }
    )

def sanitize_json_text(text: str) -> str:
    """
    Fix common invalid escape sequences from GPT like '\$',
    which are illegal in JSON but valid in Python.
    """
    # Fix the exact issue you hit: "\$" inside strings
    text = text.replace("\\$", "$")
    # You can add more targeted replacements here if needed
    return text

def call_gpt(df: pd.DataFrame) -> dict:
    """Call GPT and return parsed JSON with cleaning_code, chart_code, insights."""
    payload = build_df_payload(df)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
    )

    raw = completion.choices[0].message.content.strip()
    st.session_state["raw_gpt"] = raw  # store for debugging

    # Strip markdown fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    # Sanitize invalid JSON escapes
    raw = sanitize_json_text(raw)

    try:
        data = json.loads(raw)
    except Exception as e:
        st.error("GPT returned invalid JSON. See raw output below.")
        st.code(st.session_state["raw_gpt"])
        raise ValueError(f"JSON parse error: {e}")

    # Basic validation
    for key in ("cleaning_code", "chart_code", "insights"):
        if key not in data:
            raise ValueError(f"GPT JSON missing key: {key}")

    return data

def exec_cleaning(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Execute cleaning_code to get clean_df(df)."""
    env = {"pd": pd, "np": np}
    loc = {}
    exec(code, env, loc)

    if "clean_df" not in loc:
        raise ValueError("cleaning_code did not define clean_df(df).")

    out = loc["clean_df"](df.copy())
    if not isinstance(out, pd.DataFrame):
        raise ValueError("clean_df(df) did not return a DataFrame.")
    return out

def exec_charts(df: pd.DataFrame, code: str):
    """Execute chart_code to get figures from create_charts(df)."""
    env = {"pd": pd, "np": np, "px": px, "go": go}
    loc = {}
    exec(code, env, loc)

    if "create_charts" not in loc:
        raise ValueError("chart_code did not define create_charts(df).")

    figs = loc["create_charts"](df.copy())

    # Your model returned dict earlier; support both dict and list
    if isinstance(figs, dict):
        figs = list(figs.values())
    return figs

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
    st.session_state.raw_gpt = ""

# -------------------------------------------------------------
# LOAD FILE
# -------------------------------------------------------------
if not uploaded:
    st.info("Upload a dataset to begin.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
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
st.caption("Output of GPT's clean_df(df).")
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
    if not figs:
        st.info("GPT did not return any figures.")
    else:
        for fig in figs:
            if isinstance(fig, go.Figure):
                fig.update_layout(**PLOTLY_THEME)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("GPT returned a non-Plotly object.")
except Exception as e:
    st.error(f"Chart Error: {e}")

# -------------------------------------------------------------
# RAW GPT OUTPUT (DEBUG)
# -------------------------------------------------------------
with st.expander("🔧 Raw GPT Output (for debugging)"):
    st.code(st.session_state.get("raw_gpt", "No response yet"))

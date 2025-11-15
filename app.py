# -------------------------------------------------------------
# InsightHub – GPT-Powered Auto-Clean + Auto-Visualize
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

import openai  # uses the classic ChatCompletion API

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="InsightHub - GPT Data Analyst",
    page_icon="📊",
    layout="wide",
)

openai.api_key = os.getenv("OPENAI_API_KEY", None)
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

# Plotly dark theme
pio.templates.default = "plotly_dark"
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E1E1E1", size=14),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
)

# -------------------------------------------------------------
# MINIMAL CUSTOM CSS
# -------------------------------------------------------------
st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Cards */
.card {
    background: #111827;
    padding: 20px 22px;
    border-radius: 16px;
    box-shadow: 0 18px 40px rgba(15,23,42,0.55);
    border: 1px solid rgba(148,163,184,0.45);
    margin-bottom: 24px;
}

/* Hero */
.hero {
    padding: 18px 22px 12px 22px;
    border-radius: 18px;
    background: radial-gradient(circle at top left, #4f46e5 0, #0ea5e9 45%, #020617 100%);
    color: #f9fafb;
    box-shadow: 0 24px 60px rgba(15,23,42,0.75);
    margin-bottom: 18px;
}

/* Buttons */
.stButton > button {
    border-radius: 999px;
    padding: 0.45rem 1.3rem;
    border: none;
    background: linear-gradient(135deg, #4f46e5, #0ea5e9);
    color: white;
    font-weight: 600;
}
.stButton > button:hover {
    filter: brightness(1.05);
}

/* DataFrame wrapper */
[data-testid="stDataFrame"] {
    border-radius: 0.75rem;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,0.35);
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# GPT HELPERS
# -------------------------------------------------------------
def build_dataframe_spec(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Build a JSON summary of the DataFrame for GPT:
    - schema (column + dtype)
    - missing counts
    - basic statistics
    - sample rows
    """
    spec = {}
    spec["n_rows"], spec["n_cols"] = df.shape
    spec["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    spec["missing_counts"] = df.isna().sum().to_dict()

    # Describe numeric & categorical separately to keep it compact
    try:
        desc_num = df.select_dtypes(include=[np.number]).describe().to_dict()
    except Exception:
        desc_num = {}
    try:
        desc_cat = (
            df.select_dtypes(exclude=[np.number])
            .describe(include="all", datetime_is_numeric=True)
            .to_dict()
        )
    except Exception:
        desc_cat = {}

    spec["describe_numeric"] = desc_num
    spec["describe_other"] = desc_cat

    sample = df.head(max_rows).copy()
    # Convert datetimes to string for JSON
    for col in sample.columns:
        if np.issubdtype(sample[col].dtype, np.datetime64):
            sample[col] = sample[col].astype(str)
    spec["sample_rows"] = sample.to_dict(orient="records")

    return json.dumps(spec, default=str)


SYSTEM_PROMPT = """
You are an expert data analyst and Python developer.
You are helping a non-technical business user explore a pandas DataFrame called `df`.

You will receive a JSON description of the DataFrame: schema, statistics and a few sample rows.
Your job is to:
1. Decide how to CLEAN the data (types, missing values, duplicates, obvious outliers).
2. Generate Python CODE that does this cleaning in a function `def clean_df(df):`.
3. Design 2–5 meaningful Plotly visualizations and generate Python CODE in a function
   `def create_charts(df):` that:
      - takes the CLEANED DataFrame as input
      - returns a list named `figures` containing Plotly Figure objects
      - each figure should already have a title, axis labels, etc.
4. Write a concise, business-friendly INSIGHT SUMMARY in Markdown.

Constraints:
- Libraries you may use in code: pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go.
- DO NOT use seaborn, statsmodels, matplotlib, or any other library.
- Do not read or write any files.
- Do not create or expect any global variables other than `df`.
- Cleaning code should be robust and not crash when some columns are missing or non-numeric.
- Prefer simple models (no ML libraries).

Return your answer as STRICT JSON with this structure:

{
  "cleaning_code": "python code that defines clean_df(df)",
  "chart_code": "python code that defines create_charts(df)",
  "insights": "markdown string with 5-10 bullet points of insights"
}

Important:
- The value of `cleaning_code` MUST define a function `clean_df(df)` and end by returning the cleaned df.
- The value of `chart_code` MUST define a function `create_charts(df)` and finish by returning a list called `figures`.
- Do NOT wrap any code in backticks.
- Do NOT include explanations outside the JSON.
"""

def call_gpt_for_analysis(df: pd.DataFrame) -> dict:
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or Streamlit secrets.")

    spec_json = build_dataframe_spec(df)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": spec_json},
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    raw_text = completion["choices"][0]["message"]["content"]

    # Clean possible ```json code fences
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # strip first and last triple backticks
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
            # remove optional 'json' prefix
            if cleaned.strip().startswith("json"):
                cleaned = cleaned.strip()[4:]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse GPT JSON: {e}\nRaw text:\n{raw_text[:1500]}")

    # Basic validation
    for key in ["cleaning_code", "chart_code", "insights"]:
        if key not in parsed:
            raise ValueError(f"GPT response missing key: {key}")

    return parsed


def execute_cleaning_code(df: pd.DataFrame, cleaning_code: str) -> pd.DataFrame:
    """
    Execute the GPT-generated cleaning code in a restricted namespace.
    Expect it to define clean_df(df).
    """
    local_env = {}
    global_env = {
        "pd": pd,
        "np": np,
    }
    exec(cleaning_code, global_env, local_env)

    if "clean_df" not in local_env:
        raise ValueError("cleaning_code did not define a function clean_df(df).")

    cleaned_df = local_env["clean_df"](df.copy())
    if not isinstance(cleaned_df, pd.DataFrame):
        raise ValueError("clean_df(df) did not return a pandas DataFrame.")
    return cleaned_df


def execute_chart_code(df: pd.DataFrame, chart_code: str):
    """
    Execute the GPT-generated chart code.
    Expect it to define create_charts(df) that returns list of figures.
    """
    local_env = {}
    global_env = {
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
    }
    exec(chart_code, global_env, local_env)

    if "create_charts" not in local_env:
        raise ValueError("chart_code did not define a function create_charts(df).")

    figures = local_env["create_charts"](df.copy())
    if not isinstance(figures, (list, tuple)):
        raise ValueError("create_charts(df) did not return a list of figures.")
    return figures


# -------------------------------------------------------------
# APP UI
# -------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1 style="margin-bottom:4px;">📊 InsightHub – GPT Data Analyst</h1>
  <p style="margin:0;font-size:0.95rem;color:#e5e7eb;">
    Upload a CSV/Excel file. GPT will automatically clean the data, choose the best charts,
    and explain what matters in plain language.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("1️⃣ Upload your data")
uploaded_file = st.sidebar.file_uploader(
    "CSV or Excel", type=["csv", "xls", "xlsx"], help="Files stay in your session only."
)

max_rows_for_spec = st.sidebar.slider(
    "Rows to send to GPT (sample size)", min_value=10, max_value=100, value=25, step=5
)

run_button = st.sidebar.button("2️⃣ Run GPT Analysis")

# Space to store results in session to avoid recalling API on every rerun
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

# ---------------- MAIN LOGIC ----------------
if uploaded_file is None:
    st.info("👆 Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

# Load data
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# Show preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📄 Data Preview")
st.caption("First 100 rows of your uploaded data (raw, before cleaning).")
st.dataframe(df_raw.head(100), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Optionally run GPT
if run_button:
    # Reset previous state
    st.session_state.ai_result = None
    st.session_state.cleaned_df = None

    with st.spinner("🤖 Calling GPT to analyze your data..."):
        try:
            # Limit rows for summary
            df_for_spec = df_raw.head(max_rows_for_spec).copy()
            ai_result = call_gpt_for_analysis(df_for_spec)
            # Run cleaning code
            cleaned_df = execute_cleaning_code(df_raw, ai_result["cleaning_code"])
            st.session_state.ai_result = ai_result
            st.session_state.cleaned_df = cleaned_df
        except Exception as e:
            st.error(f"AI analysis failed: {e}")
            st.stop()

# If we already have results, display them
if st.session_state.ai_result is None or st.session_state.cleaned_df is None:
    st.info("Click **'Run GPT Analysis'** in the sidebar to let GPT clean and analyze your data.")
    st.stop()

ai_result = st.session_state.ai_result
cleaned_df = st.session_state.cleaned_df

# ---------------- CLEANED DATA CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧹 Cleaned Data (GPT Output)")
st.caption("This is the DataFrame after GPT's cleaning logic (types, missing values, etc.).")
st.dataframe(cleaned_df.head(100), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- INSIGHTS CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧠 AI Insights")
st.caption("GPT's narrative summary of key patterns and findings in your data.")
st.markdown(ai_result.get("insights", "_No insights returned._"))
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CHARTS CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 GPT-Recommended Visualizations")

try:
    figures = execute_chart_code(cleaned_df, ai_result["chart_code"])
    if not figures:
        st.write("GPT did not return any figures.")
    else:
        for i, fig in enumerate(figures):
            # Ensure it's a Plotly Figure
            if isinstance(fig, (go.Figure, px.Figure)):
                fig.update_layout(**PLOTLY_THEME)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Item {i} from create_charts is not a Plotly Figure object.")
except Exception as e:
    st.error(f"Error while generating charts from GPT code: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ADVANCED: SHOW GENERATED CODE ----------------
with st.expander("🔍 Advanced: View GPT-Generated Cleaning & Chart Code"):
    st.markdown("#### Cleaning Code (`clean_df(df)`)")
    st.code(ai_result["cleaning_code"], language="python")
    st.markdown("#### Chart Code (`create_charts(df)`)")
    st.code(ai_result["chart_code"], language="python")

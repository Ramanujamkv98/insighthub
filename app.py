# InsightHub 3.0 – GPT-powered Auto EDA
# Uses OpenAI v1 Python client and ast.literal_eval to avoid JSON issues.

import ast
import json
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from openai import OpenAI

# ------------------ page config & theme ------------------ #
st.set_page_config(
    page_title="InsightHub – GPT Auto EDA",
    page_icon="📊",
    layout="wide",
)

pio.templates.default = "plotly_dark"

st.markdown(
    """
<style>
    body { font-family: "Inter", system-ui, sans-serif; }
    .block-container { padding-top: 1.2rem; }
    .card {
        background: #0f172a;
        padding: 1rem 1.25rem;
        border-radius: 16px;
        border: 1px solid #1e293b;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #020617;
        padding: 0.75rem 1rem;
        border-radius: 14px;
        border: 1px solid #1f2937;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e5e7eb;
    }
</style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 InsightHub 3.0 – GPT Auto EDA")
st.caption("Upload a dataset → AI cleans it → AI builds charts → Ask questions in plain English.")

# ------------------ OpenAI client ------------------ #
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("Please add OPENAI_API_KEY in your Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)


# ------------------ helper functions ------------------ #
def dataframe_profile(df: pd.DataFrame, sample_rows: int = 200) -> str:
    """Create a compact textual profile of the dataframe for GPT."""
    sample = df.head(sample_rows)
    profile = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing": df.isna().sum().to_dict(),
        "sample_csv": sample.to_csv(index=False),
    }
    return json.dumps(profile, indent=2)


def detect_date_col(df: pd.DataFrame):
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in ["date", "week", "day", "month", "year"]):
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                continue
    return None


def detect_target_col(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    priority = [
        "revenue",
        "sales",
        "amount",
        "gmv",
        "turnover",
        "profit",
        "value",
        "total",
    ]
    for col in numeric_cols:
        l = col.lower()
        if any(p in l for p in priority):
            return col
    return numeric_cols[-1] if numeric_cols else None


def call_gpt_for_eda(df: pd.DataFrame) -> dict:
    """
    Ask GPT to return a Python dict literal with:
      - cleaning_code: defines clean_df(df)
      - eda_code: defines make_figures(df)
      - insights: bullet text
    We parse with ast.literal_eval to avoid JSON escaping issues.
    """
    profile_text = dataframe_profile(df)

    system_msg = """
You are a senior data analyst and Python expert.

You will receive a JSON description of a pandas DataFrame called df.

You MUST return a single valid Python dict literal, NOT JSON, NOT markdown.
No backticks. No ```json blocks. No comments outside the dict.

Return exactly:

{
  "cleaning_code": "...python code string...",
  "eda_code": "...python code string...",
  "insights": "...markdown bullet list..."
}

Rules for cleaning_code:
- It MUST define:  def clean_df(df):  ...  return df
- Do NOT read files from disk (no pd.read_csv).
- The input DataFrame is already named df.
- Handle monetary / numeric strings: strip '$', '₹', ',', '%', whitespace,
  then convert to numeric where appropriate using pd.to_numeric(..., errors="ignore").
- Drop columns whose name starts with 'Unnamed'.
- Try to parse obvious date-like columns to datetime.

Rules for eda_code:
- It MUST define:  def make_figures(df): ... return figures
- figures MUST be a dict: { "chart_title": plotly_figure, ... }
- Use ONLY plotly.express or plotly.graph_objects.
- Create at least 5–8 insightful charts, e.g.:
    * time series of main metric over date/week
    * correlation heatmap
    * top 2 scatter plots (e.g. spends vs revenue)
    * channel breakdown bar chart(s)
    * distribution / boxplots to show outliers
- Assume df is already cleaned by clean_df.

Rules for insights:
- Markdown bullet list of 5–10 bullets.
- Each bullet = clear business takeaway from the EDA.

Again: respond ONLY with a valid Python dict literal. No markdown fences.
"""

    user_msg = f"""
Here is a structured description of the DataFrame:

{profile_text}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content.strip()

    # strip accidental fences if any
    if text.startswith("```"):
        # take content between first and last fence
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    try:
        result = ast.literal_eval(text)
    except Exception as e:
        st.error("Failed to parse GPT response as Python dict.")
        st.code(text)
        raise e

    required_keys = {"cleaning_code", "eda_code", "insights"}
    if not required_keys.issubset(result.keys()):
        raise ValueError(f"GPT output missing keys: {required_keys - set(result.keys())}")

    return result, text


def run_cleaning(df_raw: pd.DataFrame, cleaning_code: str) -> pd.DataFrame:
    """Execute GPT cleaning code."""
    env = {"pd": pd, "np": np}
    local_vars = {"df": df_raw.copy()}
    try:
        exec(cleaning_code, env, local_vars)
    except Exception as e:
        st.error("Error while executing cleaning_code.")
        st.code(cleaning_code)
        raise e

    if "clean_df" in local_vars:
        cleaned = local_vars["clean_df"](df_raw.copy())
    elif "df" in local_vars:
        cleaned = local_vars["df"]
    else:
        cleaned = df_raw.copy()

    if not isinstance(cleaned, pd.DataFrame):
        raise ValueError("cleaning_code did not return a DataFrame.")
    return cleaned


def run_eda_code(df_clean: pd.DataFrame, eda_code: str):
    """Execute GPT EDA code to obtain dict of figures."""
    env = {"pd": pd, "np": np, "px": px, "go": go}
    local_vars = {}
    try:
        exec(eda_code, env, local_vars)
    except Exception as e:
        st.error("Error while executing eda_code.")
        st.code(eda_code)
        raise e

    if "make_figures" not in local_vars:
        raise ValueError("eda_code did not define make_figures(df).")

    figures = local_vars["make_figures"](df_clean.copy())
    if isinstance(figures, list):
        # convert to dict with generic titles
        figures = {f"Chart {i+1}": f for i, f in enumerate(figures)}

    return figures


def quick_kpis(df: pd.DataFrame):
    """Compute simple KPIs based on detected target column."""
    target = detect_target_col(df)
    out = {}
    if target and target in df.columns and np.issubdtype(df[target].dtype, np.number):
        series = df[target].dropna()
        if not series.empty:
            out["target_col"] = target
            out["sum"] = float(series.sum())
            out["mean"] = float(series.mean())
            out["min"] = float(series.min())
            out["max"] = float(series.max())
    return out


def display_kpi_cards(kpis: dict):
    cols = st.columns(4)
    if not kpis:
        st.write("No numeric target column detected for KPIs.")
        return
    cols[0].markdown('<div class="metric-card"><div class="metric-label">Target</div>'
                     f'<div class="metric-value">{kpis["target_col"]}</div></div>',
                     unsafe_allow_html=True)
    cols[1].markdown('<div class="metric-card"><div class="metric-label">Total</div>'
                     f'<div class="metric-value">{kpis["sum"]:,.2f}</div></div>',
                     unsafe_allow_html=True)
    cols[2].markdown('<div class="metric-card"><div class="metric-label">Average</div>'
                     f'<div class="metric-value">{kpis["mean"]:,.2f}</div></div>',
                     unsafe_allow_html=True)
    cols[3].markdown('<div class="metric-card"><div class="metric-label">Max</div>'
                     f'<div class="metric-value">{kpis["max"]:,.2f}</div></div>',
                     unsafe_allow_html=True)


# ------------------ sidebar: upload ------------------ #
st.sidebar.header("📂 Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not uploaded:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

if uploaded.name.lower().endswith(".csv"):
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = pd.read_excel(uploaded)

# ------------------ raw preview ------------------ #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📄 Raw Data Preview")
st.dataframe(df_raw.head(200), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------ run GPT (cached for this file) ------------------ #
@st.cache_data(show_spinner=False)
def run_gpt_pipeline(data: pd.DataFrame):
    eda_bundle, raw_text = call_gpt_for_eda(data)
    cleaned = run_cleaning(data, eda_bundle["cleaning_code"])
    figures = run_eda_code(cleaned, eda_bundle["eda_code"])
    insights_md = eda_bundle["insights"]
    return cleaned, figures, insights_md, eda_bundle, raw_text


with st.spinner("🤖 Letting GPT analyze and design your EDA..."):
    df_clean, gpt_figures, gpt_insights, gpt_bundle, gpt_raw_text = run_gpt_pipeline(
        df_raw
    )

# ------------------ layout: tabs ------------------ #
tab_overview, tab_charts, tab_insights, tab_ask = st.tabs(
    ["📌 Overview", "📊 Charts", "🧠 Insights & Recommendations", "💬 Ask the Data (AI)"]
)

# ---- Overview tab ---- #
with tab_overview:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧹 Cleaned Data (Head)")
    st.dataframe(df_clean.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Key Metrics (Auto-detected)")
    kpis = quick_kpis(df_clean)
    display_kpi_cards(kpis)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧮 Dataset Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape**:", df_clean.shape)
        st.write("**Columns:**", ", ".join(df_clean.columns))
    with col2:
        st.write("**Dtypes:**")
        st.write(df_clean.dtypes.astype(str))
    st.markdown("</div>", unsafe_allow_html=True)


# ---- Charts tab ---- #
with tab_charts:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 GPT Recommended Visualizations")
    if not gpt_figures:
        st.write("GPT did not return any charts.")
    else:
        for name, fig in gpt_figures.items():
            if isinstance(fig, (go.Figure,)):
                st.markdown(f"### {name.replace('_', ' ').title()}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"Skipping non-figure for key: {name}")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔧 View GPT EDA Code"):
        st.write("**Cleaning code (clean_df)**")
        st.code(gpt_bundle["cleaning_code"], language="python")
        st.write("**EDA code (make_figures)**")
        st.code(gpt_bundle["eda_code"], language="python")


# ---- Insights tab ---- #
with tab_insights:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 GPT Insights (Business Interpretation)")
    st.markdown(gpt_insights)
    st.markdown("</div>", unsafe_allow_html=True)

    # Simple extra: our own correlation insight
    num_df = df_clean.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        stacked = corr.where(~np.eye(len(corr), dtype=bool)).stack()
        if not stacked.empty:
            top_pair = stacked.abs().sort_values(ascending=False).head(1)
            (c1, c2), val = top_pair.index[0], top_pair.iloc[0]
            st.markdown(
                f"> 🔍 Extra Insight: The strongest numeric relationship is between **{c1}** and **{c2}** "
                f"with correlation **{val:.2f}**. Consider exploring this relationship deeper."
            )


# ---- Ask-the-data tab ---- #
with tab_ask:
    st.subheader("💬 Ask Questions About This Data")

    user_q = st.text_area(
        "Type a question (examples: 'Which channels look most efficient?', 'Any signs of saturation?', 'How do holidays affect revenue?')",
        height=100,
    )
    ask_btn = st.button("Ask AI")

    if ask_btn and user_q.strip():
        # build a small context sample
        sample_csv = df_clean.head(200).to_csv(index=False)
        qa_system = """
You are a data analyst assistant. Answer questions about the user's dataset.

You will be given:
- A small CSV sample of the cleaned DataFrame.
- The high-level GPT insights already generated.

Your job:
- Answer in clear, structured Markdown.
- Refer to trends, channels, correlations, segments.
- If something is unclear or not visible from the sample, say so honestly.
"""
        qa_user = f"""
CLEANED DATA SAMPLE (CSV):
{sample_csv}

EARLIER INSIGHTS:
{gpt_insights}

QUESTION:
{user_q}
"""

        with st.spinner("Thinking about your question..."):
            qa_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": qa_system},
                    {"role": "user", "content": qa_user},
                ],
                temperature=0.3,
            )
        answer = qa_resp.choices[0].message.content.strip()
        st.markdown("### 🔎 Answer")
        st.markdown(answer)

    st.markdown("---")
    with st.expander("Debug: raw GPT EDA response text"):
        st.code(gpt_raw_text, language="python")

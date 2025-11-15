# InsightHub 5.0 – Scientific Analyst Mode
# Universal cleaning + EDA + Linear Regression (simple & multiple) + GPT insights

import json
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
import streamlit as st
from openai import OpenAI

# ------------------ Page config & styling ------------------ #

st.set_page_config(
    page_title="InsightHub 5.0 – Scientific Analyst Mode",
    page_icon=None,
    layout="wide",
)

pio.templates.default = "plotly_dark"

st.markdown(
    """
<style>
body {
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.block-container {
    padding-top: 1rem;
}
.card {
    background: #0B1120;
    padding: 1rem 1.25rem;
    border-radius: 16px;
    border: 1px solid #1F2937;
    margin-bottom: 1rem;
}
.metric-card {
    background: #020617;
    padding: 0.75rem 1rem;
    border-radius: 14px;
    border: 1px solid #111827;
}
.metric-label {
    font-size: 0.8rem;
    color: #9CA3AF;
}
.metric-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: #E5E7EB;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("InsightHub 5.0 – Scientific Analyst Mode")
st.caption("Built for Indian SMBs: clean your data, run regressions, and get interpretable insights.")

# ------------------ OpenAI client ------------------ #

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Please add OPENAI_API_KEY in your Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)


# ------------------ Universal Cleaning Engine ------------------ #

def auto_clean_df(df: pd.DataFrame):
    """
    Universal cleaning for messy business data.
    Handles:
      - Unnamed columns
      - All-null rows
      - Currency symbols (₹, $, commas, %)
      - Accounting negatives: (500) -> -500
      - Object columns that are numeric-like or datetime-like
      - Inf / -Inf
    Returns:
      cleaned_df, cleaning_info (dict)
    """
    df = df.copy()
    info = {
        "dropped_columns": [],
        "rows_dropped_all_null": 0,
        "numeric_converted": [],
        "datetime_converted": [],
    }

    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop Excel "Unnamed" columns
    mask_unnamed = df.columns.str.contains("^unnamed", case=False, regex=True)
    dropped = df.columns[mask_unnamed].tolist()
    if dropped:
        info["dropped_columns"].extend(dropped)
        df = df.loc[:, ~mask_unnamed]

    # Drop fully null rows
    before_rows = len(df)
    df = df.dropna(how="all")
    info["rows_dropped_all_null"] = before_rows - len(df)

    # Replace inf / -inf
    df = df.replace([np.inf, -np.inf], np.nan)

    # Clean object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        s = df[col].astype(str).str.strip()

        # 1) Try datetime
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        if dt.notna().mean() >= 0.7:
            df[col] = dt
            info["datetime_converted"].append(col)
            continue

        # 2) Try numeric-like
        cleaned = (
            s.str.replace(r"\((.*)\)", r"-\1", regex=True)  # accounting negatives
             .str.replace("[₹$,]", "", regex=True)         # currency symbols and commas
             .str.replace("%", "", regex=False)            # percentages
             .str.replace(r"\s+", "", regex=True)          # extra spaces
        )
        cleaned = cleaned.replace("", np.nan)
        num = pd.to_numeric(cleaned, errors="coerce")

        if num.notna().mean() >= 0.5:
            df[col] = num
            info["numeric_converted"].append(col)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df, info


def detect_date_column(df: pd.DataFrame):
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
        name = str(col).lower()
        if any(k in name for k in ["date", "day", "week", "month", "year"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                if parsed.notna().mean() > 0.6:
                    return col
            except Exception:
                continue
    return None


def detect_target_column(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None
    priority = ["revenue", "sales", "amount", "gmv", "turnover", "profit", "value", "total"]
    for col in numeric_cols:
        low = str(col).lower()
        if any(p in low for p in priority):
            return col
    # fallback: numeric column with largest variance
    return df[numeric_cols].var().sort_values(ascending=False).index[0]


# ------------------ Modelling and EDA helpers ------------------ #

def compute_kpis(df: pd.DataFrame):
    target = detect_target_column(df)
    if not target:
        return {}
    s = df[target].dropna()
    if s.empty:
        return {}
    return {
        "target_col": target,
        "sum": float(s.sum()),
        "mean": float(s.mean()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def render_kpi_cards(kpis: dict):
    if not kpis:
        st.write("No numeric target detected for KPIs.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><div class="metric-label">Target metric</div>'
        f'<div class="metric-value">{kpis["target_col"]}</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card"><div class="metric-label">Total</div>'
        f'<div class="metric-value">{kpis["sum"]:,.2f}</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-card"><div class="metric-label">Average</div>'
        f'<div class="metric-value">{kpis["mean"]:,.2f}</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="metric-card"><div class="metric-label">Maximum</div>'
        f'<div class="metric-value">{kpis["max"]:,.2f}</div></div>',
        unsafe_allow_html=True,
    )


def make_missing_bar(df: pd.DataFrame):
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return None
    mdf = missing.reset_index()
    mdf.columns = ["column", "missing"]
    fig = px.bar(
        mdf,
        x="column",
        y="missing",
        title="Missing values per column",
        text="missing",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def make_corr_matrix(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None, None
    corr = num_df.corr().round(3)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation heatmap (numeric columns)",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    return corr, fig


def make_time_series(df: pd.DataFrame):
    date_col = detect_date_column(df)
    target = detect_target_column(df)
    if not date_col or not target:
        return None
    if not np.issubdtype(df[target].dtype, np.number):
        return None
    tmp = df[[date_col, target]].dropna()
    if tmp.empty:
        return None
    tmp = tmp.sort_values(by=date_col)
    fig = px.line(
        tmp,
        x=date_col,
        y=target,
        title=f"{target} over time ({date_col})",
    )
    return fig


def make_target_distribution(df: pd.DataFrame):
    target = detect_target_column(df)
    if not target:
        return None
    if not np.issubdtype(df[target].dtype, np.number):
        return None
    fig = px.histogram(
        df,
        x=target,
        nbins=30,
        title=f"Distribution of {target}",
    )
    return fig


def make_category_breakdown(df: pd.DataFrame):
    target = detect_target_column(df)
    if not target:
        return None
    if not np.issubdtype(df[target].dtype, np.number):
        return None

    cat_candidates = []
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= 20:
                cat_candidates.append((col, nunique))

    if not cat_candidates:
        return None

    cat_candidates = sorted(cat_candidates, key=lambda x: x[1])
    cat_col = cat_candidates[0][0]

    tmp = df.groupby(cat_col)[target].mean().reset_index()
    tmp = tmp.sort_values(by=target, ascending=False)
    fig = px.bar(
        tmp,
        x=cat_col,
        y=target,
        title=f"Average {target} by {cat_col}",
        text_auto=".2f",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


# ------------------ Regression: simple & multiple ------------------ #

def compute_simple_linear_regressions(df: pd.DataFrame):
    """
    For each numeric feature (except target), fit:
        target = a + b * feature
    Return a DataFrame with slope, intercept, R², p-value.
    """
    target = detect_target_column(df)
    if not target:
        return None

    num_df = df.select_dtypes(include=[np.number])
    if target not in num_df.columns:
        return None

    results = []
    for feature in num_df.columns:
        if feature == target:
            continue
        sub = df[[feature, target]].dropna()
        if sub.shape[0] < 10:
            continue

        X = sm.add_constant(sub[feature])
        y = sub[target]
        try:
            model = sm.OLS(y, X).fit()
        except Exception:
            continue

        slope = model.params[feature]
        intercept = model.params["const"]
        r2 = model.rsquared
        pval = model.pvalues[feature]

        results.append(
            {
                "feature": feature,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "p_value": pval,
                "n_obs": int(sub.shape[0]),
            }
        )

    if not results:
        return None

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by="r2", ascending=False)
    return res_df


def compute_multiple_regression(df: pd.DataFrame):
    """
    Fit multiple linear regression:
        target ~ all other numeric predictors
    Returns:
      coef_df (DataFrame) with coefficient, std error, p-value
      summary_text (string)
    """
    target = detect_target_column(df)
    if not target:
        return None, None

    num_df = df.select_dtypes(include=[np.number])
    if target not in num_df.columns or num_df.shape[1] < 2:
        return None, None

    X = num_df.drop(columns=[target]).dropna()
    if X.shape[0] < 20:
        return None, None

    # Align y with X indexes
    y = df[target].loc[X.index]
    X_const = sm.add_constant(X)
    try:
        model = sm.OLS(y, X_const).fit()
    except Exception:
        return None, None

    coef_df = pd.DataFrame(
        {
            "variable": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "p_value": model.pvalues.values,
        }
    )
    coef_df = coef_df[coef_df["variable"] != "const"]
    coef_df = coef_df.sort_values(by="p_value")

    summary_text = model.summary().as_text()
    return coef_df, summary_text


# ------------------ GPT: insights and Q&A (scientific-aware) ------------------ #

def build_profile_for_gpt(df_clean, cleaning_info, simple_reg_df, multi_coef_df):
    sample = df_clean.head(200)
    profile = {
        "shape": list(df_clean.shape),
        "columns": list(df_clean.columns),
        "dtypes": {c: str(df_clean[c].dtype) for c in df_clean.columns},
        "cleaning_info": cleaning_info,
        "missing_counts": df_clean.isna().sum().to_dict(),
        "simple_regression_top5": (
            simple_reg_df.head(5).to_dict(orient="records") if simple_reg_df is not None else None
        ),
        "multiple_regression_coefs": (
            multi_coef_df.to_dict(orient="records") if multi_coef_df is not None else None
        ),
        "sample_csv": sample.to_csv(index=False),
    }
    return json.dumps(profile, indent=2)


def call_gpt_insights(df_clean, cleaning_info, simple_reg_df, multi_coef_df) -> str:
    profile_json = build_profile_for_gpt(df_clean, cleaning_info, simple_reg_df, multi_coef_df)

    system_msg = """
You are a senior analytics consultant working with small and medium businesses in India.

You will receive a JSON summary of a cleaned dataset plus results from:
- simple linear regressions (target vs each predictor)
- multiple linear regression

Your task is to write a concise yet rigorous narrative:
- 5 to 10 bullet points of key insights.
- Focus on which variables are most strongly associated with the target metric.
- Distinguish strong relationships (high |slope|, high R², low p-value) from weak ones.
- Highlight any potential multicollinearity or caution where interpretation is limited.
- Avoid claiming causality; speak in terms of association and practical business guidance.

Then add a short section:
- "Recommended next analyses" – 3 to 5 bullet points of what the SMB should do next
  (e.g., experiment with budget shifts, segment analysis, seasonality checks, etc.).

Use a professional tone. Do not output any Python code.
"""

    user_msg = f"""
Here is the dataset and modelling summary:

{profile_json}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.25,
    )
    return resp.choices[0].message.content.strip()


def call_gpt_qa(df_clean, insights_md, simple_reg_df, multi_coef_df, question: str) -> str:
    sample_csv = df_clean.head(200).to_csv(index=False)
    simple_json = (
        simple_reg_df.head(10).to_dict(orient="records") if simple_reg_df is not None else None
    )
    multi_json = (
        multi_coef_df.to_dict(orient="records") if multi_coef_df is not None else None
    )

    system_msg = """
You are a data analyst answering questions for an Indian SMB.
Use the sample data and regression outputs to ground your answer.
Be explicit when you rely on the regression evidence.
Avoid overclaiming causality; talk in terms of association and likely impact.

Answer in clear Markdown. Do not output Python code.
"""

    user_msg = f"""
CLEANED DATA SAMPLE (CSV):
{sample_csv}

PRIOR INSIGHTS (Markdown):
{insights_md}

SIMPLE LINEAR REGRESSIONS (top features):
{json.dumps(simple_json, indent=2)}

MULTIPLE REGRESSION COEFFICIENTS:
{json.dumps(multi_json, indent=2)}

QUESTION:
{question}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ------------------ Sidebar: upload ------------------ #

st.sidebar.header("Upload data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not uploaded:
    st.info("Upload a CSV or Excel file to get started.")
    st.stop()

if uploaded.name.lower().endswith(".csv"):
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = pd.read_excel(uploaded)

# ------------------ Raw preview ------------------ #

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Raw data preview</div>', unsafe_allow_html=True)
st.dataframe(df_raw.head(200), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ------------------ Pipeline (cached) ------------------ #

@st.cache_data(show_spinner=False)
def run_pipeline(data: pd.DataFrame):
    cleaned, info = auto_clean_df(data)
    simple_reg = compute_simple_linear_regressions(cleaned)
    multi_coef, multi_summary = compute_multiple_regression(cleaned)
    insights_md = call_gpt_insights(cleaned, info, simple_reg, multi_coef)
    return cleaned, info, simple_reg, multi_coef, multi_summary, insights_md


with st.spinner("Cleaning data and running models..."):
    df_clean, clean_info, simple_reg_df, multi_coef_df, multi_summary_text, gpt_insights = run_pipeline(
        df_raw
    )

# ------------------ Tabs ------------------ #

tab_overview, tab_data_quality, tab_eda, tab_models, tab_insights, tab_ask = st.tabs(
    [
        "Overview",
        "Data quality",
        "Visual EDA",
        "Modelling (regressions)",
        "GPT interpretation",
        "Ask the data",
    ]
)

# ---- Overview ---- #

with tab_overview:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cleaned data (head)</div>', unsafe_allow_html=True)
    st.dataframe(df_clean.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key metrics</div>', unsafe_allow_html=True)
    kpis = compute_kpis(df_clean)
    render_kpi_cards(kpis)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Data quality ---- #

with tab_data_quality:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cleaning summary</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape after cleaning:", df_clean.shape)
        st.write("Dropped columns:", clean_info["dropped_columns"] or "None")
        st.write("Rows dropped (all null):", clean_info["rows_dropped_all_null"])
    with col2:
        st.write("Converted to numeric:", clean_info["numeric_converted"] or "None")
        st.write("Converted to datetime:", clean_info["datetime_converted"] or "None")
    st.markdown("</div>", unsafe_allow_html=True)

    miss_fig = make_missing_bar(df_clean)
    if miss_fig is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Missing values</div>', unsafe_allow_html=True)
        st.plotly_chart(miss_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---- Visual EDA ---- #

with tab_eda:
    ts_fig = make_time_series(df_clean)
    dist_fig = make_target_distribution(df_clean)
    corr_matrix, corr_fig = make_corr_matrix(df_clean)
    cat_fig = make_category_breakdown(df_clean)

    if ts_fig is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Time series</div>', unsafe_allow_html=True)
        st.plotly_chart(ts_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if dist_fig is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Target distribution</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(dist_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if corr_fig is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Correlation heatmap</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(corr_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if cat_fig is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Category breakdown</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(cat_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---- Modelling ---- #

with tab_models:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Simple linear regressions (target vs each predictor)</div>',
        unsafe_allow_html=True,
    )
    if simple_reg_df is None:
        st.write("Not enough numeric variables to run simple linear regressions.")
    else:
        st.dataframe(simple_reg_df, use_container_width=True)
        st.markdown(
            "Higher absolute slope and higher R² indicate a stronger linear association. "
            "Low p-values suggest the slope is statistically different from zero.",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Multiple linear regression</div>',
        unsafe_allow_html=True,
    )
    if multi_coef_df is None:
        st.write(
            "Not enough suitable numeric predictors or observations to fit a multiple regression model."
        )
    else:
        st.dataframe(multi_coef_df, use_container_width=True)
        with st.expander("Full regression summary"):
            st.text(multi_summary_text)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- GPT interpretation ---- #

with tab_insights:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">GPT interpretation</div>', unsafe_allow_html=True)
    st.markdown(gpt_insights)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Ask the data ---- #

with tab_ask:
    st.markdown('<div class="section-title">Ask questions about this dataset</div>', unsafe_allow_html=True)
    question = st.text_area(
        "Example: 'Which channels should I prioritise in October?', "
        "'How sensitive is revenue to TV spend?', "
        "'Is there evidence of diminishing returns on digital spend?'",
        height=120,
    )
    ask_button = st.button("Ask AI")

    if ask_button and question.strip():
        with st.spinner("Preparing an answer based on the model outputs..."):
            answer = call_gpt_qa(df_clean, gpt_insights, simple_reg_df, multi_coef_df, question)
        st.markdown(answer)

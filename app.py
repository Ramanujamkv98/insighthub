# InsightHub 6.0 – Scientific Analyst Mode (SMB Edition)

import json
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------
# Streamlit page configuration and global Plotly theme
# ---------------------------------------------------

st.set_page_config(
    page_title="InsightHub 6.0 – Scientific Analyst Mode",
    page_icon=None,
    layout="wide",
)

pio.templates.default = "plotly_dark"

# Basic SaaS-style dark theme
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
    background: #020617;
    padding: 1rem 1.25rem;
    border-radius: 16px;
    border: 1px solid #1f2933;
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

st.title("InsightHub 6.0 – Scientific Analyst Mode")
st.caption("Built for Indian small and medium businesses – clean your data, understand drivers, and act on plain-English insights.")

# ---------------------------------------------------
# OpenAI client
# ---------------------------------------------------

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Please add OPENAI_API_KEY in your Streamlit secrets to use the AI features.")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------------------------------------------
# Cleaning utilities
# ---------------------------------------------------

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

        # Try datetime
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        if dt.notna().mean() >= 0.7:
            df[col] = dt
            info["datetime_converted"].append(col)
            continue

        # Try numeric-like cleaning
        cleaned = (
            s.str.replace(r"\((.*)\)", r"-\1", regex=True)  # accounting negatives
             .str.replace("[₹$,]", "", regex=True)         # currency symbols and commas
             .str.replace("%", "", regex=False)            # percentage sign
             .str.replace(r"\s+", "", regex=True)          # extra spaces
        )
        cleaned = cleaned.replace("", np.nan)
        num = pd.to_numeric(cleaned, errors="coerce")

        # If at least half the column can be numeric, keep it
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

# ---------------------------------------------------
# KPI helpers
# ---------------------------------------------------

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
        st.write("No clear numeric target metric detected yet. Try a dataset with revenue or sales.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><div class="metric-label">Main metric</div>'
        f'<div class="metric-value">{kpis["target_col"]}</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card"><div class="metric-label">Total</div>'
        f'<div class="metric-value">{kpis["sum"]:,.2f}</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-card"><div class="metric-label">Average per row</div>'
        f'<div class="metric-value">{kpis["mean"]:,.2f}</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="metric-card"><div class="metric-label">Highest value</div>'
        f'<div class="metric-value">{kpis["max"]:,.2f}</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------
# Chart helpers – SaaS theme
# ---------------------------------------------------

def apply_theme(fig):
    fig.update_layout(
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font=dict(color="white", size=14),
        title=dict(font=dict(size=20)),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


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
        color_discrete_sequence=["#6366F1"],
    )
    fig.update_layout(xaxis_tickangle=-45, bargap=0.25)
    return apply_theme(fig)


def make_trend(df: pd.DataFrame, date_col: str, value_col: str):
    tmp = df[[date_col, value_col]].dropna().sort_values(date_col)
    if tmp.empty:
        return None
    fig = px.line(
        tmp,
        x=date_col,
        y=value_col,
        markers=True,
        title=f"{value_col.replace('_',' ').title()} over time",
        color_discrete_sequence=["#4F46E5"],
    )
    fig.update_traces(line_width=2)
    return apply_theme(fig)


def make_hist(df: pd.DataFrame, col: str):
    tmp = df[col].dropna()
    if tmp.empty:
        return None
    fig = px.histogram(
        df,
        x=col,
        nbins=20,
        opacity=0.85,
        title=f"How {col.replace('_',' ').title()} is spread",
        color_discrete_sequence=["#6366F1"],
    )
    fig.update_layout(bargap=0.15, xaxis_title=col.replace("_", " ").title(), yaxis_title="Number of rows")
    return apply_theme(fig)


def make_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Numbers that move together",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    return apply_theme(fig)


def make_category_chart(df: pd.DataFrame, col: str, target: str | None = None):
    if target and target in df.columns and np.issubdtype(df[target].dtype, np.number):
        tmp = df.groupby(col)[target].mean().reset_index()
        tmp = tmp.sort_values(by=target, ascending=False).head(20)
        fig = px.bar(
            tmp,
            x=col,
            y=target,
            title=f"Average {target.replace('_',' ').title()} by {col}",
            text_auto=".2f",
            color_discrete_sequence=["#22C55E"],
        )
    else:
        vc = df[col].value_counts().head(20).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(
            vc,
            x=col,
            y="count",
            title=f"Most common {col.replace('_',' ').title()}",
            text_auto=True,
            color_discrete_sequence=["#22C55E"],
        )

    fig.update_layout(xaxis_tickangle=-45)
    return apply_theme(fig)


def make_scatter(df: pd.DataFrame, x: str, y: str):
    tmp = df[[x, y]].dropna()
    if tmp.empty:
        return None
    fig = px.scatter(
        tmp,
        x=x,
        y=y,
        opacity=0.8,
        title=f"{y.replace('_',' ').title()} vs {x.replace('_',' ').title()}",
        color_discrete_sequence=["#10B981"],
    )
    return apply_theme(fig)


def auto_charts(df: pd.DataFrame):
    """Pick a reasonable set of charts for the EDA tab."""
    figs = {}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_col = detect_date_column(df)
    target = detect_target_column(df)

    # Trend chart
    if date_col and target:
        trend_fig = make_trend(df, date_col, target)
        if trend_fig is not None:
            figs["trend"] = trend_fig

    # Histograms for up to 2 numeric columns
    for col in numeric[:2]:
        h = make_hist(df, col)
        if h is not None:
            figs[f"hist_{col}"] = h

    # Category chart
    if categorical:
        cat_fig = make_category_chart(df, categorical[0], target)
        if cat_fig is not None:
            figs["category"] = cat_fig

    # Scatter of first numeric vs target
    if target and target in numeric:
        for col in numeric:
            if col != target:
                sc = make_scatter(df, col, target)
                if sc is not None:
                    figs[f"scatter_{col}"] = sc
                break

    # Heatmap
    heat = make_heatmap(df)
    if heat is not None:
        figs["heatmap"] = heat

    return figs

# ---------------------------------------------------
# Regression: simple & multiple
# ---------------------------------------------------

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
        if sub.shape[0] < 20:
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
    if X.shape[0] < 40:
        return None, None

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

# ---------------------------------------------------
# GPT helpers – SMB-friendly prompts
# ---------------------------------------------------

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
You are InsightHub, a simple English business advisor for small shops and small businesses in India.

You will receive:
- A summary of the dataset
- Some basic model results about what relates to the main metric (for example, revenue or total sales).

Your job is to explain the patterns in very simple language.

Important:
- Do NOT use words like "p-value", "R-squared", "regression", "multicollinearity", "coefficient", "statistical significance".
- Avoid any heavy academic or technical terminology.
- Use examples and explanations a shopkeeper or small business owner can understand.
- Focus on:
  * Which numbers seem to move together with the main metric
  * Which products, channels or columns look important
  * Any clear up or down trends over time
  * Fast-moving vs slow-moving items
  * High-value vs low-value items

Structure your answer as:

1. Key findings (5–8 short bullet points)
2. What this means for your business (3–5 bullet points)
3. What you should do next (3–6 very concrete action points)
"""

    user_msg = f"""
Here is the dataset and modelling summary as JSON:

{profile_json}
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


def call_gpt_qa(df_clean, insights_md, simple_reg_df, multi_coef_df, question: str) -> str:
    sample_csv = df_clean.head(200).to_csv(index=False)
    simple_json = (
        simple_reg_df.head(10).to_dict(orient="records") if simple_reg_df is not None else None
    )
    multi_json = (
        multi_coef_df.to_dict(orient="records") if multi_coef_df is not None else None
    )

    system_msg = """
You are an analytics assistant helping a small business owner in India.
Use simple, everyday language.

You have:
- A sample of the cleaned data
- Some model outputs about which columns relate to the main metric
- A previous high-level explanation of the data

When you answer:

- Always ground your answer in the data and model information you were given.
- Do NOT use words like "p-value", "R-squared", "regression", "multicollinearity".
- Instead, say things like "looks strongly connected", "weak relationship", "goes up when this goes up".
- If there is not enough information, clearly say so.

End your answer with a short section called "Practical next steps" with 2–4 bullets.
"""

    user_msg = f"""
CLEANED DATA SAMPLE (CSV):
{sample_csv}

PREVIOUS INSIGHTS (Markdown):
{insights_md}

SIMPLE LINE RELATIONSHIPS (top features):
{json.dumps(simple_json, indent=2)}

MULTI-FACTOR MODEL SUMMARY:
{json.dumps(multi_json, indent=2)}

QUESTION FROM THE USER:
{question}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.35,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------
# Dynamic example questions
# ---------------------------------------------------

def generate_dynamic_examples(df_clean: pd.DataFrame):
    examples = []
    cols = df_clean.columns.tolist()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    date_col = detect_date_column(df_clean)
    target = detect_target_column(df_clean)

    # Main metric questions
    if target:
        examples.append(f"What are the main things that seem to affect {target}?")
        examples.append(f"Which items or columns are most linked to higher {target}?")

    # Time-based questions
    if date_col and target:
        examples.append(f"Do you see any monthly or seasonal ups and downs in {target}?")
        examples.append(f"Is {target} generally going up or down over time?")

    # Spend / marketing type questions
    spend_like = [c for c in cols if "spend" in c.lower() or "budget" in c.lower()]
    if spend_like and target:
        examples.append("Which spend channel seems to give better returns?")
        examples.append("If I increase one type of spend, which one is likely to help the most?")

    # Categorical questions
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols and target:
        cat = cat_cols[0]
        examples.append(f"Which {cat} groups are doing better on {target}?")
        examples.append(f"Are there any {cat} groups that look weak and need attention?")

    # General questions
    examples.append("Which items look like fast movers and which are slow?")
    examples.append("Are there any weeks or months where performance suddenly changed?")

    # Remove duplicates and keep first 6–7
    examples = list(dict.fromkeys(examples))
    return examples[:7]

# ---------------------------------------------------
# Streamlit sidebar – upload
# ---------------------------------------------------

st.sidebar.header("Upload data")
uploaded = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if not uploaded:
    st.info("Upload a CSV or Excel file to start the analysis.")
    st.stop()

if uploaded.name.lower().endswith(".csv"):
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = pd.read_excel(uploaded)

# ---------------------------------------------------
# Raw data preview
# ---------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Raw data preview</div>', unsafe_allow_html=True)
st.dataframe(df_raw.head(200), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Cached pipeline
# ---------------------------------------------------

@st.cache_data(show_spinner=False)
def run_pipeline(data: pd.DataFrame):
    cleaned, info = auto_clean_df(data)
    simple_reg = compute_simple_linear_regressions(cleaned)
    multi_coef, multi_summary = compute_multiple_regression(cleaned)
    insights_md = call_gpt_insights(cleaned, info, simple_reg, multi_coef)
    return cleaned, info, simple_reg, multi_coef, multi_summary, insights_md

with st.spinner("Cleaning data and running analysis..."):
    df_clean, clean_info, simple_reg_df, multi_coef_df, multi_summary_text, gpt_insights = run_pipeline(
        df_raw
    )

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------

tab_overview, tab_quality, tab_eda, tab_models, tab_insights, tab_ask = st.tabs(
    ["Overview", "Data quality", "Visual EDA", "Drivers & models", "AI summary", "Ask the data"]
)

# ---------------- Overview ---------------- #

with tab_overview:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cleaned data (first 200 rows)</div>', unsafe_allow_html=True)
    st.dataframe(df_clean.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key numbers</div>', unsafe_allow_html=True)
    kpis = compute_kpis(df_clean)
    render_kpi_cards(kpis)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Data quality ---------------- #

with tab_quality:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cleaning summary</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Rows after cleaning:", df_clean.shape[0])
        st.write("Columns after cleaning:", df_clean.shape[1])
        st.write("Dropped columns:", clean_info["dropped_columns"] or "None")
        st.write("Rows removed because everything was empty:", clean_info["rows_dropped_all_null"])
    with col2:
        st.write("Columns converted to numbers:", clean_info["numeric_converted"] or "None")
        st.write("Columns converted to dates:", clean_info["datetime_converted"] or "None")
    st.markdown("</div>", unsafe_allow_html=True)

    missing_fig = make_missing_bar(df_clean)
    if missing_fig is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Missing values</div>', unsafe_allow_html=True)
        st.plotly_chart(missing_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Visual EDA ---------------- #

with tab_eda:
    figs = auto_charts(df_clean)

    for key, fig in figs.items():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if not figs:
        st.write("Not enough numeric or categorical information to build useful charts yet.")

# ---------------- Models ---------------- #

with tab_models:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Simple relationships (one column vs main metric)</div>',
        unsafe_allow_html=True,
    )
    if simple_reg_df is None:
        st.write("Not enough numeric columns to compute relationships.")
    else:
        # Rename columns to more friendly labels
        display_df = simple_reg_df.rename(
            columns={
                "feature": "column",
                "slope": "impact_per_unit",
                "intercept": "baseline_level",
                "r2": "fit_score",
                "p_value": "evidence_score",
                "n_obs": "rows_used",
            }
        )
        st.dataframe(display_df, use_container_width=True)
        st.markdown(
            "Higher absolute values in **impact_per_unit** and **fit_score** usually mean a stronger link with the main metric. "
            "Smaller **evidence_score** values indicate stronger evidence that this link is real and not just noise.",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Many-columns model (all main numbers together)</div>', unsafe_allow_html=True)
    if multi_coef_df is None:
        st.write(
            "There was not enough clean numeric information to build a reliable many-columns model. "
            "Try adding more data or more numeric columns."
        )
    else:
        display_multi = multi_coef_df.rename(
            columns={
                "variable": "column",
                "coefficient": "impact_per_unit",
                "std_error": "uncertainty",
                "p_value": "evidence_score",
            }
        )
        st.dataframe(display_multi, use_container_width=True)
        st.markdown(
            "Columns with larger absolute **impact_per_unit** and lower **evidence_score** "
            "are more likely to matter for the main business outcome."
        )
        with st.expander("Technical details (optional)"):
            st.text(multi_summary_text)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- AI summary ---------------- #

with tab_insights:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Plain-English summary and recommendations</div>', unsafe_allow_html=True)
    st.markdown(gpt_insights)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Ask the data ---------------- #

with tab_ask:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Ask questions about this data</div>', unsafe_allow_html=True)

    example_questions = generate_dynamic_examples(df_clean)
    placeholder = "You can ask things like:\n" + "\n".join(f"- {q}" for q in example_questions)

    question = st.text_area(
        placeholder,
        height=160,
    )
    ask_button = st.button("Ask AI")

    if ask_button and question.strip():
        with st.spinner("Thinking based on the data and models..."):
            answer = call_gpt_qa(df_clean, gpt_insights, simple_reg_df, multi_coef_df, question)
        st.markdown(answer)

    st.markdown("</div>", unsafe_allow_html=True)

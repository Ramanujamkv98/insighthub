import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from openai import OpenAI

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(
    page_title="InsightHub – GPT Auto-EDA",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    body { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 InsightHub – GPT Auto EDA")
st.caption("AI-powered data cleaning, visual analysis & insights.")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------
# File Upload
# ----------------------------
st.sidebar.header("📂 Upload CSV / Excel")
uploaded = st.sidebar.file_uploader("Upload your file", type=["csv", "xlsx"])

if not uploaded:
    st.info("Upload a dataset to begin.")
    st.stop()

# Read file
if uploaded.name.endswith(".csv"):
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = pd.read_excel(uploaded)

st.subheader("📄 Raw Data")
st.dataframe(df_raw.head(200), use_container_width=True)

# Pass entire dataset to GPT (safe)
rows_to_pass = len(df_raw)

# Create preview sample (only 100-200 rows max to reduce token)
sample_df = df_raw.head(200)

csv_preview = sample_df.to_csv(index=False)

# ----------------------------
# GPT Prompt
# ----------------------------
prompt = f"""
You are a senior data analyst.

The user uploaded a dataset (first 200 rows shown).

Your tasks:

1. **Clean the dataset**:
   - Convert numeric columns with $ or commas to floats.
   - Fix dates.
   - Remove dummy / unnamed columns.
   - Output Python code that cleans the DataFrame `df`.

2. **Recommend at least 6 insightful visualizations**:
   Use only Plotly.
   Create:
     - Correlation heatmap  
     - Top 2 scatterplots  
     - Spend vs revenue  
     - Week-over-week trends  
     - Channel breakdown  
     - Outlier detection  
   Output Python code that creates figs as a dict called `figures`.

3. **Generate high-quality English insights** (bullet points, business style).

4. **Return STRICT JSON only**:
{{
  "cleaning_code": "python code...",
  "chart_code": "python code...",
  "insights": "bullet points..."
}}

DATA PREVIEW:
{csv_preview}
"""

with st.spinner("Analyzing with GPT..."):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

raw_output = response.choices[0].message.content.strip()

# ----------------------------
# Parse JSON safely
# ----------------------------
try:
    gpt_json = json.loads(raw_output)
except:
    st.error("GPT returned invalid JSON. Showing raw output:")
    st.code(raw_output)
    st.stop()

cleaning_code = gpt_json["cleaning_code"]
chart_code = gpt_json["chart_code"]
insights_text = gpt_json["insights"]

# ----------------------------
# Execute Cleaning Code
# ----------------------------
local_env = {"df": df_raw.copy(), "pd": pd, "np": np}
exec(cleaning_code, local_env)
df_clean = local_env["df"]

st.subheader("🧹 Cleaned Data")
st.dataframe(df_clean.head(200), use_container_width=True)

# ----------------------------
# Execute Chart Code
# ----------------------------
chart_env = {"df": df_clean, "px": px, "go": go, "np": np, "pd": pd}
exec(chart_code, chart_env)
figures = chart_env["figures"]

# ----------------------------
# INSIGHTS
# ----------------------------
st.subheader("💡 GPT Insights")
st.markdown(insights_text)

# ----------------------------
# CHARTS
# ----------------------------
st.subheader("📊 GPT Visualizations")

for name, fig in figures.items():
    st.markdown(f"### {name.replace('_',' ').title()}")
    st.plotly_chart(fig, use_container_width=True)


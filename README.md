****README Summary for DataPilot (Short & Polished)**
**About DataPilot**
**DataPilot is a lightweight, AI-powered analytics application designed to help small and medium-scale businesses unlock insights from their data without needing a full data team. Many SMBs struggle with messy spreadsheets, inconsistent column names, and limited visualization tools. DataPilot automates cleaning, semantic column matching, KPI detection, and EDA visualizations — all through a simple, no-code Streamlit interface.

With built-in OpenAI analysis, the tool can automatically:
Clean and harmonize datasets
Detect KPI categories based on semantics
Generate executive summary metrics
Produce Auto-EDA insights
Answer natural-language questions about the data

My goal in building this was to make analytics more accessible for organizations that lack technical resources, giving them the ability to make faster, data-driven decisions.
🔐 Data Privacy & Security
DataPilot runs locally inside the user’s browser session or secured Streamlit deployment. Uploaded data is:
Not stored,
Not logged,
Not shared with third parties,
Only used in-memory for generating insights.

When AI analysis is used, only a small sample (first 40 rows) is sent to the model to maintain minimal data exposure. For sensitive industries (healthcare, finance, field operations), 
this approach reduces the risk of revealing personal or regulated information.

The tool is intentionally designed to be:

Lightweight
Privacy-respectful
Suitable for internal workflows
Users retain full control of their data at every step.

 Why I Built This

During my experience working with healthcare and supply-chain teams — and from seeing how SMBs operate 
I realized that many organizations want to use data but lack tooling, analysts, or engineering support. DataPilot is my attempt to simplify that journey by giving them:

Immediate KPIs
Automated insights
Cleaned data
Visualizations
Natural-language querying
All without needing SQL, Python, or BI tools.


"""
LLM-as-a-Judge Experiment Dashboard

Run with: streamlit run dashboard.py
"""

import streamlit as st

st.set_page_config(
    page_title="LLM Judge Experiments",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🔬 LLM Judge Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Comparison", "Sample Explorer", "Similarity Analysis"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Quick Help
- **Overview**: View metrics for a single experiment
- **Comparison**: Compare multiple experiments side-by-side
- **Sample Explorer**: Browse individual samples and responses
- **Similarity Analysis**: Analyze similarity-based retrieval experiments
""")

# Route to appropriate page
if page == "Overview":
    from dashboard.pages.overview import render
    render()
elif page == "Comparison":
    from dashboard.pages.comparison import render
    render()
elif page == "Sample Explorer":
    from dashboard.pages.sample_explorer import render
    render()
elif page == "Similarity Analysis":
    from dashboard.pages.similarity import render
    render()

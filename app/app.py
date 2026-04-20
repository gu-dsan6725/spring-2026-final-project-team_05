"""
Streamlit app for contract clause analysis demo.
Upload or paste a contract to analyze clauses, risk scores, and benchmarks.
"""

import streamlit as st
import json
import sys
import os
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator_agent import run_pipeline


st.set_page_config(
    page_title="Contract Clause Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 4px solid #d32f2f;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 4px solid #f57c00;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 4px solid #388e3c;
    }
    .risk-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.875rem;
    }
    .high-risk-badge {
        background-color: #d32f2f;
        color: white;
    }
    .medium-risk-badge {
        background-color: #f57c00;
        color: white;
    }
    .low-risk-badge {
        background-color: #388e3c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Contract Clause Analyzer")
st.markdown("Automated risk assessment for commercial contracts using AI agents")

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None
if "contract_text" not in st.session_state:
    st.session_state.contract_text = ""


def get_risk_color(score):
    """Return color based on risk score."""
    if score > 0.6:
        return "High"
    elif score >= 0.3:
        return "Medium"
    else:
        return "Low"


def get_risk_class(score):
    """Return CSS class for risk score."""
    if score > 0.6:
        return "high-risk"
    elif score >= 0.3:
        return "medium-risk"
    else:
        return "low-risk"


def get_risk_badge_class(score):
    """Return CSS class for risk badge."""
    if score > 0.6:
        return "risk-badge high-risk-badge"
    elif score >= 0.3:
        return "risk-badge medium-risk-badge"
    else:
        return "risk-badge low-risk-badge"


# Sidebar for input
st.sidebar.header("Input Contract")

input_method = st.sidebar.radio(
    "Choose input method:",
    ["Upload File", "Paste Text"],
    key="input_method"
)

contract_text = ""

if input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a .txt contract file",
        type=["txt"],
        key="file_uploader"
    )
    if uploaded_file is not None:
        contract_text = uploaded_file.read().decode("utf-8")
        st.session_state.contract_text = contract_text
else:
    contract_text = st.sidebar.text_area(
        "Paste contract text here:",
        height=300,
        key="text_input"
    )
    st.session_state.contract_text = contract_text

# Analyze button
if st.sidebar.button("Analyze Contract", type="primary"):
    if not contract_text.strip():
        st.error("Please provide a contract (upload or paste)")
    else:
        with st.spinner("Analyzing contract... (this may take 30-60 seconds)"):
            try:
                result = run_pipeline(contract_text)
                report_json = json.loads(result["report"])
                st.session_state.results = report_json
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.session_state.results = None

# Display results
if st.session_state.results:
    report = st.session_state.results
    summary = report.get("summary", {})
    clauses = report.get("clauses", [])

    # Summary section
    st.header("Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Clauses",
            value=summary.get("total_clauses", 0)
        )

    # Count risk levels
    high_risk_count = sum(1 for c in clauses if c.get("risk_score", 0) > 0.6)
    medium_risk_count = sum(1 for c in clauses if 0.3 <= c.get("risk_score", 0) <= 0.6)
    low_risk_count = sum(1 for c in clauses if c.get("risk_score", 0) < 0.3)

    with col2:
        st.metric(
            label="High Risk",
            value=high_risk_count,
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label="Medium Risk",
            value=medium_risk_count,
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="Low Risk",
            value=low_risk_count
        )

    # Clauses section
    st.header("Clause Analysis")

    if clauses:
        # Sort options
        sort_by = st.selectbox(
            "Sort by:",
            ["Risk Score (High to Low)", "Risk Score (Low to High)", "Clause Type", "Benchmark Score (Low to High)"],
            key="sort_select"
        )

        if sort_by == "Risk Score (High to Low)":
            clauses = sorted(clauses, key=lambda c: c.get("risk_score", 0), reverse=True)
        elif sort_by == "Risk Score (Low to High)":
            clauses = sorted(clauses, key=lambda c: c.get("risk_score", 0))
        elif sort_by == "Clause Type":
            clauses = sorted(clauses, key=lambda c: c.get("clause_type", "Unknown"))
        elif sort_by == "Benchmark Score (Low to High)":
            clauses = sorted(clauses, key=lambda c: c.get("benchmark_similarity", 0))

        # Display each clause in expandable format
        for idx, clause in enumerate(clauses):
            risk_score = clause.get("risk_score", 0)
            risk_label = get_risk_color(risk_score)

            # Create expander header with key info
            header_text = f"{idx + 1}. {clause.get('clause_type', 'Unknown')} — {risk_label}"

            with st.expander(header_text, expanded=(idx == 0)):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Clause Type", clause.get("clause_type", "Unknown"))

                with col2:
                    st.metric("Risk Score", f"{risk_score:.2f}", help="0.0 = Low Risk, 1.0 = High Risk")

                with col3:
                    st.metric("Confidence", f"{clause.get('confidence', 0):.2%}", help="Classification confidence")

                with col4:
                    st.metric("Benchmark Score", f"{clause.get('benchmark_similarity', 0):.2f}", help="0.0 = Unusual, 1.0 = Standard")

                st.divider()

                # Section info
                st.markdown(f"**Section:** {clause.get('section', 'Unknown')}")

                # Clause text preview
                st.markdown("**Clause Text (Preview):**")
                st.markdown(f"> {clause.get('text', 'No text available')}")

                # Risk factors
                st.markdown("**Risk Factors:**")
                risk_factors = clause.get("risk_factors", [])
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.markdown("_No significant risk factors identified._")

                # Benchmark info
                st.markdown("**Benchmark Analysis:**")
                st.markdown(f"- **Similarity to Industry Standard:** {clause.get('benchmark_similarity', 0):.2%}")
                st.markdown(f"- **Source:** {clause.get('benchmark_source', 'Unknown')}")

                if clause.get('benchmark_similarity', 0) < 0.7:
                    st.warning("This clause deviates significantly from industry standard language.")
    else:
        st.info("No clauses found in the contract.")

else:
    # Welcome section
    st.markdown("""
    ---
    ## Welcome!

    This tool analyzes commercial contracts clause-by-clause using AI agents. It provides:

    - **Clause Classification**: Identifies clause types from the CUAD taxonomy (41 clause types)
    - **Risk Scoring**: Evaluates risk factors and ambiguous language (0.0–1.0 scale)
    - **Benchmark Comparison**: Compares clauses against industry standard language

    **To get started:**
    1. Upload a `.txt` contract file or paste contract text in the sidebar
    2. Click "Analyze Contract"
    3. Explore the clause-by-clause analysis, risk scores, and benchmarks

    **Supported file types:** Plain text `.txt` files

    **Example contracts** are available in this project's data directory.
    """)

# Footer
st.markdown("""
---
**Part of:** Multi-Agent Contract Analysis Project (DSAN 6725)
[GitHub Repository](https://github.com/satomiito/spring-2026-final-project-team_05)
""")

import streamlit as st
import pandas as pd
import altair as alt

def load_mock_metrics():
    return {
        "GNN_our": {"Recall@10": 0.25, "Recall@25": 0.35, "MAP@10": 0.19},
        "BPRMF": {"Recall@10": 0.20, "Recall@25": 0.31, "MAP@10": 0.15},
        "LightGCN": {"Recall@10": 0.23, "Recall@25": 0.33, "MAP@10": 0.17},
    }

def render():
    st.title("Model Comparison")
    metrics = load_mock_metrics()

    selected_models = st.multiselect("Select models to compare", list(metrics.keys()), default=list(metrics.keys()))
    metric_options = list(next(iter(metrics.values())).keys())
    selected_metric = st.selectbox("Select metric", metric_options)

    # Prepare data for chart
    data = [{"Model": model, "Score": metrics[model][selected_metric]} for model in selected_models]
    df = pd.DataFrame(data)

    # Show bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Model", sort="-y"),
        y="Score",
        tooltip=["Model", "Score"]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df.set_index("Model"))

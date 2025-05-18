import streamlit as st
from app import compare, graph_viz, embeddings, recommend

st.set_page_config(page_title="GNN Recommender Dashboard", layout="wide")

PAGES = {
    "Home": "Welcome",
    "Compare Models": compare.render,
    "Model Visualization": graph_viz.render,
    "Embeddings": embeddings.render,
    "Recommendations": recommend.render
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if selection == "Home":
    st.title("Graph-based Recommender Systems with Explicit Negative Feedback")
    st.markdown("""
        This dashboard demonstrates the performance and internals of our GNN-based recommender model.

        **Project Description:**
        Classic recommender systems often underutilize negative feedback. Our model incorporates negative items directly via a graph-based architecture to improve recommendation relevance.

        - **Start date**: October 15, 2024
        - **Expected end date**: June 2025

        Use the sidebar to explore different aspects:
        - Compare model performance across metrics
        - Visualize user-item graphs and embeddings
        - Try out recommendation output with mock users
    """)
else:
    PAGES[selection]()

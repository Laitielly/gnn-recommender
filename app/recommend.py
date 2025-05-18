import streamlit as st
import numpy as np
import pandas as pd

def generate_mock_recommendations(user_id, num_items=100, top_k=10):
    np.random.seed(228)
    item_ids = [f"item_{i}" for i in range(num_items)]
    scores = np.random.rand(num_items)
    ranked_items = sorted(zip(item_ids, scores), key=lambda x: -x[1])[:top_k]
    return ranked_items

def render():
    st.title("Recommendations")

    user_list = [f"user_{i}" for i in range(20)]
    selected_user = st.selectbox("Выберите пользователя", user_list)
    top_k = st.slider("Top-K", 5, 20, 10)

    recommendations = generate_mock_recommendations(selected_user, top_k=top_k)

    df = pd.DataFrame(recommendations, columns=["Item ID", "Score"])
    st.write(f"Топ-{top_k} рекомендаций для {selected_user}:")
    st.table(df)

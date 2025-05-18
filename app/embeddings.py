import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import altair as alt

def generate_mock_embeddings(n_users=100, n_items=100, dim=64):
    user_embs = np.random.normal(0, 1, size=(n_users, dim))
    item_embs = np.random.normal(0, 1, size=(n_items, dim))
    return user_embs, item_embs

def prepare_pca_data(user_embs, item_embs):
    pca = PCA(n_components=2)
    all_embs = np.vstack([user_embs, item_embs])
    reduced = pca.fit_transform(all_embs)

    labels = ['user'] * len(user_embs) + ['item'] * len(item_embs)
    ids = [f"user_{i}" for i in range(len(user_embs))] + [f"item_{j}" for j in range(len(item_embs))]

    df = pd.DataFrame(reduced, columns=["x", "y"])
    df["type"] = labels
    df["id"] = ids
    return df

def render():
    st.title("Embedding Visualization")
    st.write("PCA проекция случайных эмбеддингов пользователей и айтемов (заглушка)")

    n_users = st.slider("Количество пользователей", 10, 300, 100, step=10)
    n_items = st.slider("Количество айтемов", 10, 300, 100, step=10)
    dim = st.selectbox("Размерность эмбеддингов", [16, 32, 64, 128], index=2)

    user_embs, item_embs = generate_mock_embeddings(n_users, n_items, dim)
    df = prepare_pca_data(user_embs, item_embs)

    chart = alt.Chart(df).mark_circle(size=60).encode(
        x="x", y="y",
        color="type",
        tooltip=["id", "type"]
    ).interactive().properties(height=500)

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df.head(10))
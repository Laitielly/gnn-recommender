import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import random

random.seed(228)

def generate_mock_user_item_graph(num_users=20, num_items=30, edge_prob=0.1):
    G = nx.Graph()
    users = [f"user_{i}" for i in range(num_users)]
    items = [f"item_{j}" for j in range(num_items)]

    G.add_nodes_from(users, bipartite=0)
    G.add_nodes_from(items, bipartite=1)

    for u in users:
        for i in items:
            if random.uniform(0, 1) < edge_prob:
                G.add_edge(u, i)
    return G

def render_graph(G):
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=650, scrolling=True)

    os.remove(html_path)

def render():
    st.header("User-Item Graph Visualization")
    st.write("Пример случайного user-item графа (заглушка)")

    num_users = st.slider("Количество пользователей", 10, 100, 20)
    num_items = st.slider("Количество айтемов", 10, 100, 30)
    edge_prob = st.slider("Вероятность связи", 0.01, 0.5, 0.1)

    G = generate_mock_user_item_graph(num_users, num_items, edge_prob)
    render_graph(G)

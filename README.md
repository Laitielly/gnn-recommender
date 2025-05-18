
# GNN-Based Recommender Dashboard

This is a Streamlit-based frontend for visualizing and comparing graph-based recommender models, specifically designed to work with systems that incorporate **explicit negative feedback**. The dashboard allows interactive exploration of model performance, graph structures, learned embeddings, and recommendation outputs.

---

## Features

-  **Model Comparison**: Visual comparison of models across Recall, MAP, and NDCG at different cutoff thresholds (10, 25, 50).
-  **User-Item Graph**: Interactive visualization of the bipartite user-item graph.
-  **Embeddings**: Dimensionality reduction and scatter plotting of user and item embeddings.
-  **Recommendations**: View top-N recommendations for mock users.
-  Fully containerized (Docker), suitable for isolated demo or deployment environments.

---

## Quick Start (Docker)

### 1. Build the image

```bash
docker build -t gnn-dashboard .
```

### 2. Run the container

```bash
docker run -p 8501:8501 gnn-dashboard
```

### 3. Open in browser

Go to: [http://localhost:8501](http://localhost:8501)

---

##  Local Run (without Docker)

> Requires Python 3.11+ and `virtualenv` or `poetry`

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run main.py
```

---

##  Project Structure

```
frontend/
├── app/                # Modular Streamlit pages
│   ├── compare.py
│   ├── embeddings.py
│   ├── graph_viz.py
│   ├── recommend.py
│   └── utils.py
├── data/               # Placeholder for future model outputs
├── main.py             # Streamlit entry point
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔍 Notes on Mock Data

This project currently uses synthetic (mock) data in place of real model outputs:

| Module              | Mocked? | Description                                              |
|---------------------|---------|----------------------------------------------------------|
| `compare.py`        | ✅ Yes  | Hardcoded performance metrics for 3 models              |
| `graph_viz.py`      | ✅ Yes  | Randomly generated bipartite graph                      |
| `embeddings.py`     | ✅ Yes  | Random user/item vectors projected via PCA              |
| `recommend.py`      | ✅ Yes  | Random item ranking based on fixed-seed scores          |

Once model checkpoints and outputs are available, these components can be updated to load real data from `.csv`, `.json`, or serialized `torch`/`pickle` formats.

---

##  Planned Extensions

- Real-time recommendation inference from trained GNN models
- Log and loss visualizations from training runs
- Model and dataset selector (if multiple runs/checkpoints are available)

---

## 🗓 Project Timeline

- **Start**: October 15, 2024 
- **Expected Completion**: June 2025

---

##  Related Work

Based on internal research into graph-based recommender systems with explicit negative feedback. See `docs/` and `research_documents/` for references and the system design document.

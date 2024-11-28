# Graph-based recommender systems with explicit negative feedback encoding

## Description

Classic recommender systems usually handle negative feedback only in the loss for better classification, ranking, or rating prediction. The project aims to develop a graph-based recommender system that can directly handle negative items to improve recommendation performance.

**The project started:** October 15, 2024

**Expected end date:** June 2025

## How to run

The project has configured formatters and linters, and is equipped with a pre-commit.

How to set up an environment with poetry:

```
cd ./gnn-recommender-directory
poetry install
```

How to start a pre-commit manually before pushing changes:

```
git add changes
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## What we have now

- A design document is available for review and study in [./docs/ml_system_design_doc.md](https://github.com/Laitielly/gnn-recommender/blob/main/docs/ml_system_design_doc.md).
- Completed research on 30 articles and presented the findings in a presentation. The research is available for review in [./research_documents/AIRI_research.pdf](https://github.com/Laitielly/gnn-recommender/blob/main/research_documents/AIRI_research.pdf).

# KGPolicy-CUDA

## 📌 Installation Guide

### **1.Install dependencies with Poetry**
First, install all standard dependencies using Poetry:
```sh
poetry install
```
This will create a virtual environment (`venv`) and install all required dependencies from `pyproject.toml`.
---

### **2.Install `torch-geometric` manually**
Since `torch-geometric` and its dependencies (`torch-scatter`, `torch-sparse`, `torch-cluster`) are **not available in PyPI**, you must install them separately using `pip`:

```sh
pip install torch-geometric \
  torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html \
  torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html \
  torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```
---

### **3.Verify installation**
To check if `torch-geometric` is installed correctly, run:
```sh
python -c "import torch_geometric; print(torch_geometric.__version__)"
```
If the version number appears, `torch-geometric` is successfully installed! 🚀

---

### **Notes**
- `torch-geometric` **cannot be installed through Poetry** because it requires binary dependencies from the PyG repository.
- Ensure that all team members follow these **manual installation steps** after running `poetry install`.

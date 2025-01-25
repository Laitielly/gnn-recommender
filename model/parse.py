import os
import torch
import numpy as np
import random
from pydantic_settings import BaseSettings


os.environ["OMP_NUM_THREADS"] = "20"


# Класс конфигурации
class Config(BaseSettings):
    data_dir: str = "database/"
    data: str = "users_interactions.txt"
    offset: float = 1.0
    topks: int = 10
    device: str = 'cpu'
    lambda_reg: float = 1e-4
    seed: int = 1234
    n_layers: int = 3
    hidden_dim: int = 64
    model: str = "eig+path"
    alpha: float = 0.2
    beta: float = 1.0
    eigs_dim: int = 64
    sample_hop: int = 3

    class Config:
        env_prefix = "APP_"


config = Config()
device = config.device

if config.seed != -1:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

if "eig" not in config.model:
    config.eigs_dim = 0

# Вывод конфигурации
print("Using", device)
print("Model Setting")
print(f"    hidden dim: {config.hidden_dim:d}")
print(f"    layers: {config.n_layers:d}")
print(f"    alpha: {config.alpha:f}")
print(f"    beta: {config.beta:f}")
print(f"    eigs dim: {config.eigs_dim:d}")
print(f"    sample hop: {config.sample_hop:d}")
print(f"    model: {config.model:s}")

print("Data Setting")
data_file = os.path.join(config.data_dir, config.data)
print(f"    data: {data_file:s}")
print(f"    offset: {config.offset:.1f}")

print("---------------------------")

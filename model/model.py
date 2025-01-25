from model.parse import config
import torchsparsegradutils
import torch
from torch import nn
import torch.nn.functional as F


def sum_norm(indices, values, n):
    s = torch.zeros(n, device=values.device).scatter_add(0, indices[0], values)
    s[s == 0.0] = 1.0
    return values / s[indices[0]]


def sparse_softmax(indices, values, n):
    return sum_norm(indices, torch.clamp(torch.exp(values), min=-5, max=5), n)

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.lambda0 = nn.Parameter(torch.zeros(1))
        self.path_emb = nn.Embedding(2 ** (config.sample_hop + 1) - 2, 1)
        nn.init.zeros_(self.path_emb.weight)
        self.sqrt_dim = 1.0 / torch.sqrt(torch.tensor(config.hidden_dim))
        self.sqrt_eig = 1.0 / torch.sqrt(torch.tensor(config.eigs_dim))
        self.my_parameters = [
            {"params": self.lambda0, "weight_decay": 1e-2},
            {"params": self.path_emb.parameters()},
        ]

    def forward(self, q, k, v, indices, eigs, path_type):
        ni, nx, ny, nz = [], [], [], []
        for i, pt in zip(indices, path_type):
            x = torch.mul(q[i[0]], k[i[1]]).sum(dim=-1) * self.sqrt_dim
            nx.append(x)
            if "eig" in config.model:
                if config.eigs_dim == 0:
                    y = torch.zeros(i.shape[1]).to(config.device)
                else:
                    y = torch.mul(eigs[i[0]], eigs[i[1]]).sum(dim=-1)
                ny.append(y)
            if "path" in config.model:
                z = self.path_emb(pt).view(-1)
                nz.append(z)
            ni.append(i)
        i = torch.concat(ni, dim=-1)
        s = []
        s.append(torch.concat(nx, dim=-1))
        if "eig" in config.model:
            s[0] = s[0] + torch.exp(self.lambda0) * torch.concat(ny, dim=-1)
        if "path" in config.model:
            s.append(torch.concat(nz, dim=-1))
        s = [sparse_softmax(i, _, q.shape[0]) for _ in s]
        s = torch.stack(s, dim=1).mean(dim=1)
        return torchsparsegradutils.sparse_mm(
            torch.sparse_coo_tensor(i, s, torch.Size([q.shape[0], k.shape[0]])), v
        )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.self_attention = Attention()
        self.my_parameters = self.self_attention.my_parameters

    def forward(self, x, indices, eigs, path_type):
        y = F.layer_norm(x, normalized_shape=(config.hidden_dim,))
        y = self.self_attention(y, y, y, indices, eigs, path_type)
        return y


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.dataset = dataset
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.embedding_user = nn.Embedding(self.dataset.num_users, self.hidden_dim)
        self.embedding_item = nn.Embedding(self.dataset.num_items, self.hidden_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.my_parameters = [
            {"params": self.embedding_user.parameters()},
            {"params": self.embedding_item.parameters()},
        ]
        self.layers = []
        for i in range(config.n_layers):
            layer = Encoder().to(config.device)
            self.layers.append(layer)
            self.my_parameters.extend(layer.my_parameters)
        self._users, self._items = None, None

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for i in range(self.n_layers):
            indices, paths = self.dataset.sample()
            all_emb = self.layers[i](all_emb, indices, self.dataset.L_eigs, paths)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        self._users, self._items = torch.split(
            light_out, [self.dataset.num_users, self.dataset.num_items]
        )

    def predict_user(self, user_idx):
        self.eval()
        user_emb, item_emb = self._users, self._items
        max_K = config.topks
        with torch.no_grad():
            user_e = user_emb[user_idx]
            rating = torch.mm(user_e.reshape(1, -1), item_emb.t())
            rating[0, self.dataset.pos_list[user_idx]] = -(1 << 10)
            rating[0, self.dataset.neg_list[user_idx]] = -(1 << 10)
            _, rating = torch.topk(rating, k=max_K)
        return rating

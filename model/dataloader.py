import torch
from torch.utils.data import Dataset
import pandas as pd
import scipy.sparse as sp
import torch.nn.functional as F
from model.parse import config
import os


class MyDataset(Dataset):
    def __init__(self, database, device):
        self.device = device

        data = pd.read_table(database, header=None, sep=" ")
        pos_data = data[data[2] >= config.offset]
        neg_data = data[data[2] < config.offset]
        self.data = torch.from_numpy(data.values).to(self.device)
        self.pos_user = torch.from_numpy(pos_data[0].values).to(self.device)
        self.pos_item = torch.from_numpy(pos_data[1].values).to(self.device)
        self.pos_unique_users = torch.unique(self.pos_user)
        self.pos_unique_items = torch.unique(self.pos_item)
        self.neg_user = torch.from_numpy(neg_data[0].values).to(self.device)
        self.neg_item = torch.from_numpy(neg_data[1].values).to(self.device)
        self.neg_unique_users = torch.unique(self.neg_user)
        self.neg_unique_items = torch.unique(self.neg_item)

        self.num_users = (
            max(
                [
                    self.pos_unique_users.max(),
                    self.neg_unique_users.max(),
                ]
            ).cpu()
            + 1
        )
        self.num_items = (
            max(
                [
                    self.pos_unique_items.max(),
                    self.neg_unique_items.max()
                ]
            ).cpu()
            + 1
        )
        self.num_nodes = self.num_users + self.num_items
        print("users: %d, items: %d." % (self.num_users, self.num_items))
        print(
            "train: %d pos + %d neg."
            % (self.pos_user.shape[0], self.neg_user.shape[0])
        )
        #
        self._neg_list = None
        self._pos_list = None
        self._A_pos = None
        self._A_neg = None
        self._degree_pos = None
        self._degree_neg = None
        self._tildeA = None
        self._tildeA_pos = None
        self._tildeA_neg = None
        self._indices = None
        self._paths = None
        self._values = None
        self._counts = None
        self._counts_sum = None
        self._L = None
        self._L_pos = None
        self._L_neg = None
        self._L_eigs = None

    @property
    def pos_list(self):
        if self._pos_list is None:
            self._pos_list = [
                list(self.pos_item[self.pos_user == u].cpu().numpy())
                for u in range(self.num_users)
            ]
        return self._pos_list

    @property
    def neg_list(self):
        if self._neg_list is None:
            self._neg_list = [
                list(self.neg_item[self.neg_user == u].cpu().numpy())
                for u in range(self.num_users)
            ]
        return self._neg_list

    @property
    def A_pos(self):
        if self._A_pos is None:
            self._A_pos = torch.sparse_coo_tensor(
                torch.cat(
                    [
                        torch.stack([self.pos_user, self.pos_item + self.num_users]),
                        torch.stack([self.pos_item + self.num_users, self.pos_user]),
                    ],
                    dim=1,
                ),
                torch.ones(self.pos_user.shape[0] * 2).to(self.device),
                torch.Size([self.num_nodes, self.num_nodes]),
            )
        return self._A_pos

    @property
    def degree_pos(self):
        if self._degree_pos is None:
            self._degree_pos = self.A_pos.sum(dim=1).to_dense()
        return self._degree_pos

    @property
    def tildeA_pos(self):
        if self._tildeA_pos is None:
            D = self.degree_pos.float()
            D[D == 0.0] = 1.0
            D1 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1),
                D ** (-1 / 2),
                torch.Size([self.num_nodes, self.num_nodes]),
            )
            D2 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1),
                D ** (-1 / 2),
                torch.Size([self.num_nodes, self.num_nodes]),
            )
            self._tildeA_pos = torch.sparse.mm(torch.sparse.mm(D1, self.A_pos), D2)
        return self._tildeA_pos

    @property
    def L_pos(self):
        if self._L_pos is None:
            D = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1),
                torch.ones(self.num_nodes, device=self.device),
                torch.Size([self.num_nodes, self.num_nodes]),
            )
            self._L_pos = D - self.tildeA_pos
        return self._L_pos

    @property
    def A_neg(self):
        if self._A_neg is None:
            self._A_neg = torch.sparse_coo_tensor(
                torch.cat(
                    [
                        torch.stack([self.neg_user, self.neg_item + self.num_users]),
                        torch.stack([self.neg_item + self.num_users, self.neg_user]),
                    ],
                    dim=1,
                ),
                torch.ones(self.neg_user.shape[0] * 2).to(self.device),
                torch.Size([self.num_nodes, self.num_nodes]),
            )
        return self._A_neg

    @property
    def degree_neg(self):
        if self._degree_neg is None:
            self._degree_neg = self.A_neg.sum(dim=1).to_dense()
        return self._degree_neg

    @property
    def tildeA_neg(self):
        if self._tildeA_neg is None:
            D = self.degree_neg.float()
            D[D == 0.0] = 1.0
            D1 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1),
                D ** (-1 / 2),
                torch.Size([self.num_nodes, self.num_nodes]),
            )
            D2 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1),
                D ** (-1 / 2),
                torch.Size([self.num_nodes, self.num_nodes]),
            )
            self._tildeA_neg = torch.sparse.mm(torch.sparse.mm(D1, self.A_neg), D2)
        return self._tildeA_neg

    @property
    def L_neg(self):
        if self._L_neg is None:
            D = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1),
                torch.ones(self.num_nodes, device=self.device),
                torch.Size([self.num_nodes, self.num_nodes]),
            )
            self._L_neg = D - self.tildeA_neg
        return self._L_neg

    @property
    def L(self):
        if self._L is None:
            self._L = (self.L_pos + config.alpha * self.L_neg) / (1 + config.alpha)
        return self._L

    @property
    def L_eigs(self):
        if self._L_eigs is None:
            if config.eigs_dim == 0:
                self._L_eigs = torch.tensor([]).to(self.device)
            else:
                _, self._L_eigs = sp.linalg.eigs(
                    sp.csr_matrix(
                        (self.L._values().cpu(), self.L._indices().cpu()),
                        (self.num_nodes, self.num_nodes),
                    ),
                    k=config.eigs_dim,
                    which="SR",
                )
                self._L_eigs = torch.tensor(self._L_eigs.real).to(self.device)
                self._L_eigs = F.layer_norm(self._L_eigs, normalized_shape=(config.eigs_dim,))
        return self._L_eigs

    def sample(self):
        if self._indices is None:
            self._indices = torch.cat(
                [
                    torch.stack([self.pos_user, self.pos_item + self.num_users]),
                    torch.stack([self.pos_item + self.num_users, self.pos_user]),
                    torch.stack([self.neg_user, self.neg_item + self.num_users]),
                    torch.stack([self.neg_item + self.num_users, self.neg_user]),
                ],
                dim=1,
            )
            self._paths = (
                torch.cat(
                    [
                        torch.ones(self.pos_user.shape).repeat(2),
                        torch.zeros(self.neg_user.shape).repeat(2),
                    ],
                    dim=0,
                )
                .long()
                .to(self.device)
            )
            sorted_indices = torch.argsort(self._indices[0, :])
            self._indices = self._indices[:, sorted_indices]
            self._paths = self._paths[sorted_indices]
            self._counts = torch.bincount(self._indices[0], minlength=self.num_nodes)
            self._counts_sum = torch.cumsum(self._counts, dim=0)
            d = torch.sqrt(self._counts)
            d[d == 0.0] = 1.0
            d = 1.0 / d
            self._values = (
                torch.ones(self._indices.shape[1]).to(self.device)
                * d[self._indices[0]]
                * d[self._indices[1]]
            )
        res_X, res_Y = [], []
        record_X = []
        (
            X,
            Y,
        ) = (
            self._indices,
            torch.ones_like(self._paths).long() * 2 + self._paths,
        )
        loop_indices = torch.zeros_like(Y).bool()
        for hop in range(config.sample_hop):
            loop_indices = loop_indices | (X[0] == X[1])
            for i in range(hop % 2, hop, 2):
                loop_indices = loop_indices | (record_X[i][1] == X[1])
            record_X.append(X)
            res_X.append(X[:, ~loop_indices])
            res_Y.append(Y[~loop_indices] - 2)
            next_indices = (
                self._counts_sum[X[1]]
                - (torch.rand(X.shape[1]).to(self.device) * self._counts[X[1]]).long()
                - 1
            )
            X = torch.stack([X[0], self._indices[1, next_indices]], dim=0)
            Y = Y * 2 + self._paths[next_indices]
        return res_X, res_Y

dataset = MyDataset(os.path.join(config.data_dir, config.data), config.device)

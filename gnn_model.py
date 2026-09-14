import torch
import torch.nn as nn


class BipartiteGNN(nn.Module):
    """Bipartite Graph Neural Network for decoding quantum error correction syndromes.

    Uses an alternating detector-fault message passing architecture with GRU updates.
    Optimized for large codes using CSR Sparse Matrix Multiplication (SpMM).
    """

    def __init__(self, n_det, n_fault, edges_det, edges_fault, h_dim=32, num_layers=10):
        super().__init__()
        self.n_det, self.n_fault = n_det, n_fault
        self.num_layers, self.h_dim = num_layers, h_dim

        self.register_buffer("edges_det", torch.tensor(edges_det, dtype=torch.long))
        self.register_buffer("edges_fault", torch.tensor(edges_fault, dtype=torch.long))

        self.det_emb = nn.Embedding(2, h_dim)
        self.fault_emb = nn.Parameter(torch.randn(1, n_fault, h_dim) * 0.1)
        
        self.msg_d2f = nn.Sequential(nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim))
        self.msg_f2d = nn.Sequential(nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim))
        
        self.gru_det = nn.GRUCell(h_dim, h_dim)
        self.gru_fault = nn.GRUCell(h_dim, h_dim)
        self.out_layer = nn.Linear(h_dim, 1)

        self._sparse_mats = None

    def _build_sparse_matrices(self, device):
        """Builds cached CSR adjacency matrices for fast SpMM inference."""
        edge_det_ids = self.edges_det.detach().cpu()
        edge_fault_ids = self.edges_fault.detach().cpu()

        indices = torch.stack([edge_det_ids, edge_fault_ids], dim=0)
        vals = torch.ones(edge_det_ids.numel(), dtype=torch.float32)
        
        A = torch.sparse_coo_tensor(indices, vals, size=(self.n_det, self.n_fault)).coalesce()
        A = A.to_sparse_csr().to(device)

        indices_t = torch.stack([indices[1], indices[0]], dim=0)
        AT = torch.sparse_coo_tensor(indices_t, vals, size=(self.n_fault, self.n_det)).coalesce()
        AT = AT.to_sparse_csr().to(device)

        self._sparse_mats = (A, AT)

    def _ensure_sparse(self, device):
        if self._sparse_mats is None:
            self._build_sparse_matrices(device)
        return self._sparse_mats

    def forward(self, syndrome):
        B = syndrome.shape[0]
        A, AT = self._ensure_sparse(syndrome.device)
        h = self.h_dim

        h_d = self.det_emb(syndrome.long()).permute(1, 0, 2).contiguous()
        h_f = self.fault_emb.expand(B, -1, -1).permute(1, 0, 2).contiguous()

        for _ in range(self.num_layers):
            m_f2 = self.msg_f2d(h_f).reshape(self.n_fault, B * h)
            aggr_d = torch.sparse.mm(A, m_f2).view(self.n_det, B, h)

            m_d2 = self.msg_d2f(h_d).reshape(self.n_det, B * h)
            aggr_f = torch.sparse.mm(AT, m_d2).view(self.n_fault, B, h)

            h_d = self.gru_det(aggr_d.reshape(-1, h), h_d.reshape(-1, h)).view(self.n_det, B, h)
            h_f = self.gru_fault(aggr_f.reshape(-1, h), h_f.reshape(-1, h)).view(self.n_fault, B, h)

        return self.out_layer(h_f).squeeze(-1).permute(1, 0)
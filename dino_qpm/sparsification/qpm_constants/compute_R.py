import torch


def _cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def compute_cos_sim_matrix(feat_class_matrix):
    feat_class_matrix = torch.tensor(feat_class_matrix)
    return _cosine_sim_matrix(feat_class_matrix, feat_class_matrix)

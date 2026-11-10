"""Evaluation metrics (Sec. 9.5)."""
from __future__ import annotations

import torch


def nearest_value_acc(reads: torch.Tensor, true_idx: torch.Tensor, candidates: torch.Tensor) -> float:
    """Recall: fraction of reads whose nearest candidate value (cosine) is the true one."""
    r = torch.nn.functional.normalize(reads, dim=-1)
    c = torch.nn.functional.normalize(candidates, dim=-1)
    pred = (r @ c.T).argmax(-1)
    return float((pred == true_idx).float().mean())


def benign_recall(mem, keys: torch.Tensor, values: torch.Tensor, extra_candidates=None,
                  mask=None, src=None, high_risk=True) -> float:
    cands = values if extra_candidates is None else torch.cat([values, extra_candidates])
    idx = torch.arange(len(keys), device=keys.device)
    if mask is not None:
        keys, idx = keys[mask], idx[mask]
        if src is not None:
            src = src[mask]
    if len(keys) == 0:
        return float("nan")
    reads = mem.read(keys, src=src, high_risk=high_risk)
    return nearest_value_acc(reads, idx, cands)


def targeted_success(mem, k_star, v_star, v_true=None, thresh=0.5) -> bool:
    """Flip: read closer (cosine) to v* than to v_true.  Insertion: cos(read, v*) > thresh."""
    y = mem.read(k_star.unsqueeze(0)).squeeze(0)
    cs = torch.nn.functional.cosine_similarity(y, v_star, dim=0)
    if v_true is None:
        return bool(cs > thresh)
    ct = torch.nn.functional.cosine_similarity(y, v_true, dim=0)
    return bool(cs > ct)


def target_cos(mem, k_star, v_star) -> float:
    y = mem.read(k_star.unsqueeze(0)).squeeze(0)
    return float(torch.nn.functional.cosine_similarity(y, v_star, dim=0))


def l2_recall(mem, keys, values, tol=0.5, src=None, high_risk=True) -> float:
    """Magnitude-aware recall: ||read - v|| < tol * ||v||.  Unlike nearest-cosine
    recall this counts a shrunken (diluted) or erased association as lost."""
    if len(keys) == 0:
        return float("nan")
    y = mem.read(keys, src=src, high_risk=high_risk)
    return float(((y - values).norm(dim=-1) < tol * values.norm(dim=-1)).float().mean())


def l2_flip(mem, k_star, v_star, v_true) -> bool:
    """Flip in L2: the read is closer to v* than to v_true."""
    y = mem.read(k_star.unsqueeze(0)).squeeze(0)
    return bool((y - v_star).norm() < (y - v_true).norm())


def amnesia(mem, k_star, v_true, tol=0.5) -> bool:
    """The true association is lost: ||read - v_true|| >= tol ||v_true||."""
    y = mem.read(k_star.unsqueeze(0)).squeeze(0)
    return bool((y - v_true).norm() >= tol * v_true.norm())

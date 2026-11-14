"""Tier D substrate: a Titans-style linear memory retrofitted on a FROZEN pretrained LM.

    logits(prompt) = W_U ( x_out + gain * M k )

x_out = the LM's final hidden state at the last prompt token (the input of its
unembedding, i.e. after the final norm), k = the memory key computed from an
internal hidden state of the same token, and M the test-time memory.  The LM
is never updated; only M is written at test time.  Because the read enters
AFTER the final norm, the logits are exactly linear in the read, so the pairwise
action certificate (Thm 3a) holds exactly with W = gain * W_U.

Keys: k = normalize( P (h_layer - mu) ), with (mu, P) a PCA-whitening map fitted
on a disjoint calibration set of prompts (clean, no attack involved).  Whitening
removes the anisotropy of LM hidden states so that different prompts get
near-orthogonal keys.
Values: the unit-norm unembedding direction of the asserted next token.
"""
from __future__ import annotations

from pathlib import Path

import torch


class LMRetrofit:
    def __init__(self, name: str, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.device = name, device
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.lm = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.W_U = self.lm.get_output_embeddings().weight.detach().float()        # (V, d)
        self.n_layers = self.lm.config.num_hidden_layers
        self.E_hat = None

    # ------------------------------------------------------------ features
    @torch.no_grad()
    def features(self, prompts: list[str], layers: list[int], batch: int = 32):
        """Returns x_out (N, d) and {layer: (N, d)} at the last token of every prompt."""
        grabbed = {}
        hook = self.lm.get_output_embeddings().register_forward_hook(
            lambda _m, inp, _out: grabbed.__setitem__("x", inp[0]))
        outs, hid = [], {ly: [] for ly in layers}
        try:
            for i in range(0, len(prompts), batch):
                enc = self.tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(self.device)
                o = self.lm(**enc, output_hidden_states=True)
                last = enc["attention_mask"].sum(1) - 1
                ar = torch.arange(len(last), device=self.device)
                outs.append(grabbed["x"][ar, last].float().cpu())
                for ly in layers:
                    hid[ly].append(o.hidden_states[ly][ar, last].float().cpu())
        finally:
            hook.remove()
        return torch.cat(outs), {ly: torch.cat(v) for ly, v in hid.items()}

    def first_token(self, word: str) -> int:
        return self.tok.encode(" " + word.strip())[0]

    def value_vectors(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Unit-norm unembedding directions (centred over the vocabulary)."""
        if self.E_hat is None:
            E = self.W_U - self.W_U.mean(0, keepdim=True)
            self.E_hat = E / E.norm(dim=1, keepdim=True)
        return self.E_hat[token_ids.to(self.W_U.device)]

    def logits(self, x_out: torch.Tensor, y: torch.Tensor, gain: float) -> torch.Tensor:
        return (x_out + gain * y) @ self.W_U.T


class KeyMap:
    """PCA-whitening key map fitted on calibration features."""

    def __init__(self, calib: torch.Tensor, dk: int):
        self.mu = calib.mean(0)
        X = (calib - self.mu).double()
        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        dk = min(dk, int((S > 1e-6 * S[0]).sum()))
        sd = S[:dk] / (X.shape[0] - 1) ** 0.5
        self.P = (Vh[:dk] / sd.unsqueeze(1)).float()                               # (dk, d)
        self.dk = dk

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        k = (h - self.mu.to(h)) @ self.P.to(h).T
        return k / k.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def cached_features(lm_name: str, prompts: list[str], layers: list[int], device: str, cache: Path):
    """Features of the fact pool, cached on disk (the LM is frozen, so they never change)."""
    if cache.exists():
        d = torch.load(cache, weights_only=False)
        if d["n"] == len(prompts) and set(layers) <= set(d["hid"]):
            return d["x_out"], d["hid"], d["W_U"], d["meta"]
    lm = LMRetrofit(lm_name, device)
    x_out, hid = lm.features(prompts, layers)
    meta = {"n_layers": lm.n_layers}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"n": len(prompts), "x_out": x_out, "hid": hid, "W_U": lm.W_U.cpu(), "meta": meta}, cache)
    return x_out, hid, lm.W_U.cpu(), meta

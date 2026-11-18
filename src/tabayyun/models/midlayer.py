"""Tier M substrate: a test-time memory read INSIDE a frozen pretrained LM (mid-layer residual stream).

    h_l(prompt)  <-  h_l(prompt) + R * c          (last prompt token, after block l)

and the remaining L - l blocks, the final norm and the unembedding run unchanged.  Each memory
value c is a residual-stream edit: for a (prompt, object) pair, c = delta*/R where delta* is
the vector that, added at layer l, makes the frozen LM emit the object as the next token
(gradient search with a norm cap R; the same construction as the value vector of locate-and-edit
methods).  R is a fixed scale, the median norm of h_l on calibration prompts.

Certified snapping.  The memory read y is snapped to the nearest stored value,
a = argmax_c <y, c> - |c|^2 / 2, which is linear in y, so the pairwise action certificate
(Theorem 3a with W = C) decides whether the snap is certified.  Only a certified snap is
injected.  Because the injected vector is then exactly the vector the attack-free reference would
inject, the LM's output equals the reference output whatever the depth and nonlinearity of
the remaining blocks: the certificate needs no Lipschitz constant of the network.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


class MidLayerLM:
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
        base = getattr(self.lm, "transformer", None) or getattr(self.lm, "model")
        self.blocks = base.h if hasattr(base, "h") else base.layers
        self.n_layers = len(self.blocks)
        self._delta = None          # (B, d) added at the last position of each row, or None
        self._last = None

    def first_token(self, word: str) -> int:
        return self.tok.encode(" " + word.strip())[0]

    @contextmanager
    def inject(self, layer: int, last: torch.Tensor, delta: torch.Tensor):
        """Add delta[b] to the residual stream after block ``layer`` - 1 at position last[b]."""
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            ar = torch.arange(h.shape[0], device=h.device)
            h = h.clone()
            h[ar, last] = h[ar, last] + delta.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hd = self.blocks[layer - 1].register_forward_hook(hook)
        try:
            yield
        finally:
            hd.remove()

    def encode(self, prompts: list[str]):
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        return enc, enc["attention_mask"].sum(1) - 1

    @torch.no_grad()
    def hidden(self, prompts: list[str], layer: int, batch: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(prompts), batch):
            enc, last = self.encode(prompts[i:i + batch])
            o = self.lm(**enc, output_hidden_states=True)
            out.append(o.hidden_states[layer][torch.arange(len(last), device=self.device), last].float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def next_token(self, prompts: list[str], layer: int, deltas: torch.Tensor | None, batch: int = 64) -> torch.Tensor:
        """Greedy next token of every prompt with deltas[i] injected at ``layer`` (None: no injection)."""
        out = []
        for i in range(0, len(prompts), batch):
            enc, last = self.encode(prompts[i:i + batch])
            ar = torch.arange(len(last), device=self.device)
            if deltas is None:
                logits = self.lm(**enc).logits
            else:
                with self.inject(layer, last, deltas[i:i + batch].to(self.device)):
                    logits = self.lm(**enc).logits
            out.append(logits[ar, last].argmax(-1).cpu())
        return torch.cat(out)

    def value_vectors(self, prompts: list[str], tokens: list[int], layer: int, R: float,
                      steps: int = 30, lr: float = 0.1, batch: int = 32) -> torch.Tensor:
        """delta* (N, d) with |delta*| <= R that makes the LM emit tokens[i] after prompts[i]."""
        out = []
        for i in range(0, len(prompts), batch):
            enc, last = self.encode(prompts[i:i + batch])
            tgt = torch.tensor(tokens[i:i + batch], device=self.device)
            ar = torch.arange(len(last), device=self.device)
            d = self.lm.config.hidden_size
            delta = torch.zeros(len(last), d, device=self.device, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=lr * R / d ** 0.5)
            with torch.enable_grad():
                for _ in range(steps):
                    with self.inject(layer, last, delta):
                        logits = self.lm(**enc).logits[ar, last].float()
                    loss = torch.nn.functional.cross_entropy(logits, tgt)
                    opt.zero_grad(); loss.backward(); opt.step()
                    with torch.no_grad():
                        n = delta.norm(dim=-1, keepdim=True)
                        delta.mul_(torch.clamp(R / n.clamp(min=1e-9), max=1.0))
            out.append(delta.detach().float().cpu())
        return torch.cat(out)

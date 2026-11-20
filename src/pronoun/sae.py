"""Minimal Gemma Scope (JumpReLU) SAE loader — avoids the sae_lens/TransformerLens version coupling.

Residual-stream SAEs in google/gemma-scope-2b-pt-res are trained on the output of block `layer`
(= TransformerLens `blocks.{layer}.hook_resid_post`, unprocessed weights).
We pick, per layer, the width-16k SAE whose average L0 is closest to `target_l0` (default 70, close to
the "canonical" choice used by Neuronpedia / sae_lens).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import torch

REPO = "google/gemma-scope-2b-pt-res"


class JumpReLUSAE(torch.nn.Module):
    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold, layer, l0):
        super().__init__()
        self.W_enc = torch.nn.Parameter(W_enc, requires_grad=False)
        self.W_dec = torch.nn.Parameter(W_dec, requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False)
        self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)
        self.threshold = torch.nn.Parameter(threshold, requires_grad=False)
        self.layer, self.l0 = layer, l0
        self.hook_name = f"blocks.{layer}.hook_resid_post"

    def encode(self, x):
        pre = x.float() @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))


@lru_cache(maxsize=32)
def available_l0s(layer: int, width: str = "16k") -> list[int]:
    import glob
    import os

    try:
        from huggingface_hub import HfApi

        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise RuntimeError("offline")
        paths = [f.path for f in HfApi().list_repo_tree(REPO, path_in_repo=f"layer_{layer}/width_{width}")]
    except Exception:  # offline compute node: use whatever is in the local HF cache
        from huggingface_hub.constants import HF_HUB_CACHE

        paths = glob.glob(os.path.join(HF_HUB_CACHE, "models--google--gemma-scope-2b-pt-res", "snapshots", "*",
                                       f"layer_{layer}", f"width_{width}", "average_l0_*", "params.npz"))
        if not paths:
            raise FileNotFoundError(f"no cached Gemma Scope SAE for layer {layer}")
    return sorted({int(re.search(r"average_l0_(\d+)", p).group(1)) for p in paths})


def load_sae(layer: int, width: str = "16k", target_l0: int = 70, l0: int | None = None, device="cpu"):
    from huggingface_hub import hf_hub_download

    if l0 is None:
        l0s = available_l0s(layer, width)
        l0 = min(l0s, key=lambda v: abs(v - target_l0))
    path = hf_hub_download(REPO, f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz")
    p = np.load(path)
    t = {k: torch.tensor(p[k]).float() for k in ["W_enc", "W_dec", "b_enc", "b_dec", "threshold"]}
    return JumpReLUSAE(**t, layer=layer, l0=l0).to(device)


@torch.no_grad()
def variance_explained(sae, x):
    """Reconstruction quality on activations x [N, d].

    Returns (1 - normalized MSE, mean L0), with normalized MSE = sum||x - x_hat||^2 / sum||x||^2.
    (Variance explained around the *sample* mean is misleading here: our prompts are near-identical at a
    given position, so that variance is tiny and the ratio can go strongly negative.)
    """
    x = x.float()
    a = sae.encode(x)
    r = sae.decode(a)
    fve = 1 - ((x - r) ** 2).sum() / (x ** 2).sum()
    return float(fve), float((a > 0).float().sum(-1).mean())

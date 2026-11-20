"""Model loading for Gemma-2-2B (base) in TransformerLens 2.x.

We always use `from_pretrained_no_processing` so that activations match what the
Gemma Scope SAEs were trained on (no LayerNorm folding / centering / logit centering).
If the official `google/gemma-2-2b` repo is not accessible (gated licence not accepted),
we fall back to the byte-identical `unsloth/gemma-2-2b` mirror.
"""
from __future__ import annotations

import os
from functools import lru_cache

import torch

OFFICIAL = "google/gemma-2-2b"
MIRROR = "unsloth/gemma-2-2b"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hf_source() -> str:
    src = os.environ.get("GEMMA_REPO")
    if src:
        return src
    try:
        from huggingface_hub import HfApi

        HfApi().auth_check(OFFICIAL)
        return OFFICIAL
    except Exception:
        return MIRROR


@lru_cache(maxsize=1)
def load_model(device: str | None = None, dtype: str | None = None):
    """Return a HookedTransformer for Gemma-2-2B.

    dtype defaults to float32 (PRONOUN_DTYPE to override); PRONOUN_NDEVICES=k splits the layers over k GPUs.
    """
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or os.environ.get("PRONOUN_DEVICE") or default_device()
    # Default fp32 everywhere: Gemma-2 overflows in fp16, and Turing/Pascal GPUs (2080 Ti / 1080 Ti) have no
    # native bf16. On 11 GB cards split the fp32 model over 2 GPUs with PRONOUN_NDEVICES=2.
    dtype = dtype or os.environ.get("PRONOUN_DTYPE") or "float32"
    n_devices = int(os.environ.get("PRONOUN_NDEVICES", "1"))
    torch_dtype = getattr(torch, dtype)
    src = _hf_source()
    print(f"[load_model] weights from {src} on {device} x{n_devices} ({dtype})")

    tok = AutoTokenizer.from_pretrained(src)
    hf = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    model = HookedTransformer.from_pretrained_no_processing(
        "gemma-2-2b",
        hf_model=hf,
        tokenizer=tok,
        device=device,
        n_devices=n_devices,
        dtype=torch_dtype,
    )
    del hf
    model.eval()
    # No weight gradients anywhere in this project (attribution patching only needs activation grads);
    # otherwise a backward pass allocates another 10.5 GB for weight grads.
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_grad_enabled(False)
    return model

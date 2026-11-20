"""Linear probes with control tasks (Hewitt & Liang 2019) and cross-lingual transfer."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ----------------------------------------------------------------------------- activation collection

@torch.no_grad()
def collect_resid(model, seqs, positions, batch_size=16, layers=None):
    """resid_post at given positions. seqs: list of token-id lists; positions: list of lists of ints.

    Returns X: [n_vectors, n_layers+1, d_model] float16 where layer 0 = resid_pre of block 0 (embeddings),
    and owner: [n_vectors] index into seqs. Sequences are right-padded; causal attention means padding
    after a position cannot influence it.
    """
    L = model.cfg.n_layers
    layers = list(range(L)) if layers is None else layers
    names = ["blocks.0.hook_resid_pre"] + [f"blocks.{l}.hook_resid_post" for l in layers]
    pad = model.tokenizer.pad_token_id or 0
    order = np.argsort([len(s) for s in seqs])
    feats, owner = [], []
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        T = max(len(seqs[i]) for i in idx)
        toks = torch.full((len(idx), T), pad, dtype=torch.long)
        for j, i in enumerate(idx):
            toks[j, : len(seqs[i])] = torch.tensor(seqs[i])
        _, cache = model.run_with_cache(toks.to(model.cfg.device), names_filter=lambda n: n in names,
                                        stop_at_layer=max(layers) + 1)
        stack = torch.stack([cache[n].to("cpu", torch.float16) for n in names], 2)  # [B, T, n_layers+1, d]
        for j, i in enumerate(idx):
            for p in positions[i]:
                feats.append(stack[j, p])
                owner.append(i)
    if not feats:
        return np.zeros((0, len(names), model.cfg.d_model), np.float16), np.zeros(0, int)
    X = torch.stack(feats).numpy()
    owner = np.array(owner)
    # restore original order of sequences
    srt = np.argsort(owner, kind="stable")
    return X[srt], owner[srt]


# ----------------------------------------------------------------------------- probes

class TorchLogReg:
    """Standardize + multinomial L2 logistic regression fitted by full-batch L-BFGS on GPU.

    Same objective as sklearn's LogisticRegression(C): C * sum_i CE_i + 0.5 * ||W||^2 (bias unpenalized),
    divided by n*C for conditioning. ~100x faster than sklearn on CPU for 10k x 2304 features.
    """

    def __init__(self, C=0.1, max_iter=500, device=None):
        self.C, self.max_iter = C, max_iter
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def _x(self, X):
        X = torch.as_tensor(np.asarray(X, np.float32), device=self.device)
        return (X - self.mu) / self.sd

    def fit(self, X, y):
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        X0 = torch.as_tensor(np.asarray(X, np.float32), device=self.device)
        self.mu, self.sd = X0.mean(0), X0.std(0, unbiased=False).clamp_min(1e-8)
        Xs, t = (X0 - self.mu) / self.sd, torch.as_tensor(yi, device=self.device)
        n, d, k = Xs.shape[0], Xs.shape[1], len(self.classes_)
        with torch.enable_grad():
            W = torch.zeros(d, k, device=self.device, requires_grad=True)
            b = torch.zeros(k, device=self.device, requires_grad=True)
            opt = torch.optim.LBFGS([W, b], lr=1, max_iter=self.max_iter, tolerance_grad=1e-6,
                                    tolerance_change=1e-9, history_size=20, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(Xs @ W + b, t) + (W ** 2).sum() / (2 * self.C * n)
                loss.backward()
                return loss

            opt.step(closure)
        self.W, self.b = W.detach(), b.detach()
        return self

    @torch.no_grad()
    def decision_function(self, X):
        return (self._x(X) @ self.W + self.b).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(1)]

    @property
    def coef_(self):  # in standardized space, like the sklearn pipeline's LogisticRegression step
        return self.W.T.cpu().numpy()


def make_probe(C=0.1):
    """sklearn pipeline by default; PRONOUN_PROBE=torch uses the GPU L-BFGS equivalent (TorchLogReg)."""
    import os

    if os.environ.get("PRONOUN_PROBE") == "torch":
        return TorchLogReg(C)
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def cv_probe(X, y, groups, n_splits=5, C=0.1, seed=0):
    """Grouped CV accuracy (groups = word types, so test words are never seen in training)."""
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    accs = []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_probe(C).fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)) if accs else np.nan


def control_labels(y, groups, seed=0):
    """Hewitt & Liang control task: each word type gets a random label drawn from the label distribution."""
    rng = np.random.default_rng(seed)
    types = np.unique(groups)
    vals, counts = np.unique(y, return_counts=True)
    m = {t: rng.choice(vals, p=counts / counts.sum()) for t in types}
    return np.array([m[g] for g in groups])


def majority_baseline(y):
    _, c = np.unique(y, return_counts=True)
    return float(c.max() / c.sum())


def probe_direction(X, y, C=0.1):
    """Unit-norm probe weight vector in the original (unscaled) activation space, for binary y."""
    clf = make_probe(C).fit(X, y)
    sc, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    w = lr.coef_[0] / sc.scale_
    return w / np.linalg.norm(w), clf


def bootstrap_acc(pred, y, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    c = (pred == y).astype(float)
    b = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(n_boot)]
    return float(c.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

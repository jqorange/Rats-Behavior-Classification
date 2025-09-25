"""
Very simple (but fast) SLDS (Switching Linear Dynamical System)
using EM-like iterations + vectorized Viterbi decoding.

Only depends on numpy (compute) and torch (I/O for .pt).

Usage:
    python slds_simplified.py --session F3D5_outdoor --repr-dir representations \
        --K 20 --num-iters 30
"""

import argparse, os, csv
# Avoid OpenMP runtime conflicts on Windows (e.g. MKL vs LLVM OMP).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch


# -------------------------
# IO helpers
# -------------------------
def load_representation(session, root):
    path = os.path.join(root, f"{session}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Representation file not found: {path}")
    payload = torch.load(path, map_location="cpu")
    feats = payload["features"].detach().cpu().numpy().astype(np.float32)  # (T, D)
    centers = payload.get("centers", None)
    return feats, centers


def slice_and_standardise(feats, start, end, standardise):
    obs = feats[slice(start, end)]
    if obs.ndim != 2 or obs.size == 0:
        raise ValueError(f"Invalid obs shape {obs.shape}")
    if standardise:
        m = obs.mean(0, keepdims=True).astype(np.float32)
        s = obs.std(0, keepdims=True).astype(np.float32)
        s[s == 0] = 1.0
        obs = (obs - m) / s
    return obs.astype(np.float32)


def write_states_csv(path, states, centers, start, end):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    centers_slice = None
    if centers is not None:
        centers_slice = list(centers)[slice(start, end)]
        if len(centers_slice) != len(states):
            centers_slice = None
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["center", "state"])
        for i, z in enumerate(states):
            c = centers_slice[i] if centers_slice is not None else i
            w.writerow([int(c), int(z)])


# -------------------------
# SLDS core (simplified but vectorized)
# -------------------------
class SimpleSLDS:
    """
    Fixed-K SLDS (approximate EM):
      - Hidden discrete state z_t in {0..K-1}
      - Continuous x_t (not explicitly inferred here; we use a crude proxy b_k)
      - Emission y_t ~ N(C x_t + d, R), but we approximate with y_t ~ N(C b_{z_t} + d, R)
      - M-step updates:
          * P via normalized transition counts from Viterbi path
          * b_k via least-squares:  C b_k ≈ mean_y_k - d
    Notes:
      - This is a lightweight, fast heuristic; not a full SLDS EM.
      - Works well as an unsupervised temporal segmentation baseline.
    """
    def __init__(self, K, x_dim, y_dim, seed=0):
        rng = np.random.RandomState(seed)
        self.K = int(K)
        self.x_dim, self.y_dim = int(x_dim), int(y_dim)

        # Discrete transition matrix P (row-stochastic)
        self.P = np.full((K, K), 1.0 / K, dtype=np.float32)

        # Continuous proxy per-state (x mean)
        self.b = rng.randn(K, x_dim).astype(np.float32) * 0.1

        # Shared emission C, d  (y ≈ C x + d)
        # C: (D, x_dim), d: (D,)
        self.C = (rng.randn(y_dim, x_dim).astype(np.float32) * 0.1)
        self.d = np.zeros(y_dim, dtype=np.float32)

        # Diagonal emission noise (only used as a weight if你想加权)
        self.R_diag = np.full(y_dim, 0.1, dtype=np.float32)

        # Cache
        self._logP = None  # log of P

    # ---------- utilities ----------
    def _refresh_caches(self):
        # logP cache
        self._logP = np.log(self.P + 1e-12).astype(np.float32)

    def _emission_means(self):
        """
        mean_y[k] = C @ b[k] + d  -> shape (K, D)
        """
        # (K, x_dim) @ (x_dim, D)^T  => (K, D)
        # but C is (D, x_dim), so compute (C @ b.T).T
        mean_y = (self.C @ self.b.T).T + self.d[None, :]
        return mean_y.astype(np.float32)

    # ---------- Viterbi (vectorized) ----------
    def viterbi_decode(self, Y):
        """
        Vectorized Viterbi for additive (transition + emission) scores.
        Emission score for state k at time t:
            emis[t,k] = -0.5 * || Y[t] - mean_y[k] ||^2
        """
        T, D = Y.shape
        K = self.K

        if self._logP is None:
            self._refresh_caches()

        # Precompute emission means for all states
        mean_y = self._emission_means()            # (K, D)
        mean_y_sq = np.sum(mean_y * mean_y, axis=1)  # (K,)

        # We’ll compute emis[t, :] on the fly using the identity:
        # ||y - m||^2 = ||y||^2 + ||m||^2 - 2 y·m
        # Precompute ||y||^2
        y_sq = np.sum(Y * Y, axis=1)  # (T,)

        log_delta = np.empty((T, K), dtype=np.float32)
        psi = np.empty((T, K), dtype=np.int32)

        # t = 0
        # dot = Y[0] dot mean_y (for all k)
        dot0 = Y[0][None, :] @ mean_y.T           # (1, K)
        emis0 = -0.5 * (y_sq[0] + mean_y_sq - 2.0 * dot0.ravel())  # (K,)
        log_delta[0, :] = emis0
        psi[0, :] = 0

        # DP recursion
        for t in range(1, T):
            # Transition: for each next-state k, we need max over prev-state i of:
            #   log_delta[t-1, i] + logP[i, k]
            trans = log_delta[t - 1][:, None] + self._logP  # (K, K)
            psi[t, :] = np.argmax(trans, axis=0)            # (K,)
            best_prev = np.max(trans, axis=0)               # (K,)

            # Emission at time t for all k
            # dot[t, k] = Y[t] dot mean_y[k]
            dot_t = Y[t][None, :] @ mean_y.T                # (1, K)
            emis_t = -0.5 * (y_sq[t] + mean_y_sq - 2.0 * dot_t.ravel())

            log_delta[t, :] = best_prev + emis_t

        # Backtrack
        z = np.empty(T, dtype=np.int32)
        z[-1] = int(np.argmax(log_delta[-1, :]))
        for t in range(T - 2, -1, -1):
            z[t] = psi[t + 1, z[t + 1]]
        return z

    # ---------- M-step updates ----------
    def _update_transitions(self, z):
        """Update P from a single state path z (add-1 smoothing)."""
        K = self.K
        counts = np.zeros((K, K), dtype=np.float32)
        # vectorized transitions counting
        np.add.at(counts, (z[:-1], z[1:]), 1.0)
        self.P = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + K)
        self._refresh_caches()

    def _update_b_leastsq(self, Y, z):
        """
        For each state k, set b_k by solving least-squares:
            C b_k ≈ mean_y_k - d
        where mean_y_k = mean of Y over frames with z_t = k.
        """
        # Precompute pseudo-inverse of C:  (x_dim,D)  <- (D, x_dim)
        # Use lstsq (numerically stable) instead of explicit pinv if you prefer.
        # Here we use lstsq per state as D and x_dim may not be huge.
        for k in range(self.K):
            idx = np.where(z == k)[0]
            if idx.size == 0:
                continue
            target = Y[idx].mean(axis=0) - self.d           # (D,)
            # Solve C b_k ≈ target
            # lstsq returns (x, residuals, rank, s)
            b_k, *_ = np.linalg.lstsq(self.C, target, rcond=None)
            self.b[k] = b_k.astype(np.float32)

    # ---------- main fit ----------
    def fit(self, Y, num_iters=20, verbose=True):
        Y = np.asarray(Y, dtype=np.float32)
        T, D = Y.shape
        assert D == self.y_dim

        self._refresh_caches()

        for it in range(1, num_iters + 1):
            # E-step (Viterbi path under current params)
            z = self.viterbi_decode(Y)

            # M-step
            self._update_transitions(z)
            self._update_b_leastsq(Y, z)

            if verbose:
                used = len(np.unique(z))
                print(f"[Iter {it:02d}] states used = {used}")

        return z


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="F3D5_outdoor")
    p.add_argument("--repr-dir", default="representations")
    p.add_argument("--output", default=None)
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--num-iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-standardise", action="store_true")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    feats, centers = load_representation(args.session, args.repr_dir)
    obs = slice_and_standardise(feats, args.start, args.end, not args.no_standardise)

    y_dim = obs.shape[1]
    # 为了简化，这里令 x_dim = y_dim；如果你想更小的 x_dim，可以改 C 的初始化形状。
    x_dim = y_dim

    model = SimpleSLDS(K=args.K, x_dim=x_dim, y_dim=y_dim, seed=args.seed)
    states = model.fit(obs, num_iters=args.num_iters, verbose=True)

    out_path = args.output or os.path.join(args.repr_dir, f"{args.session}_slds.csv")
    write_states_csv(out_path, states, centers, args.start, args.end)
    print(f"[OK] Saved SLDS states to {out_path}")


if __name__ == "__main__":
    main()

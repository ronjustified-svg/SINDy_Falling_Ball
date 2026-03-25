"""

SINDy (Sparse Identification of Nonlinear Dynamics) implementation
for the falling-ball system.

Library
-------
Polynomial terms in (h, v) up to degree 3:
    [1, h, v, h², hv, v², h³, h²v, hv², v³]   — 10 terms

Algorithm
---------
1. Build library matrix Θ(h, v)
2. Normalise columns to unit L²-norm
3. Initial least-squares fit:  ξ = Θ_norm \ acc
4. Sequential Thresholded Least Squares (STLSQ):
      - zero out |ξⱼ| < δ
      - refit on surviving columns
      - repeat for max_iter iterations
5. Rescale coefficients back to original data scale
6. Rank terms by L²-norm contribution: ‖Θⱼ · ξⱼ‖₂
"""

import numpy as np


# ---------------------------------------------------------------------------
# Library construction
# ---------------------------------------------------------------------------
LIBRARY_NAMES = ["1", "h", "v", "h²", "h·v", "v²", "h³", "h²·v", "h·v²", "v³"]


def build_library(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Construct the 10-column polynomial candidate library.

    Parameters
    ----------
    h : ndarray   height signal  (N,)
    v : ndarray   velocity signal (N,)

    Returns
    -------
    Theta : ndarray  shape (N, 10)
    """
    return np.column_stack([
        np.ones_like(h),   # 1
        h,                 # h
        v,                 # v
        h ** 2,            # h²
        h * v,             # h·v
        v ** 2,            # v²
        h ** 3,            # h³
        h ** 2 * v,        # h²·v
        h * v ** 2,        # h·v²
        v ** 3,            # v³
    ])


# ---------------------------------------------------------------------------
# STLSQ
# ---------------------------------------------------------------------------
def stlsq(Theta: np.ndarray, acc: np.ndarray,
          delta: float = 0.05, max_iter: int = 10):
    """
    Sequential Thresholded Least Squares on a normalised library.

    Parameters
    ----------
    Theta    : ndarray  (N, P)  candidate library (unnormalised)
    acc      : ndarray  (N,)    regression target (acceleration)
    delta    : float            sparsity threshold on normalised coefficients
    max_iter : int              maximum STLSQ iterations

    Returns
    -------
    xi          : ndarray  (P,)   sparse coefficients in original data scale
    xi_norm     : ndarray  (P,)   sparse coefficients in normalised scale
    col_norm    : ndarray  (P,)   L²-norm of each library column
    active_mask : ndarray  (P,)   bool mask of surviving terms
    """
    P = Theta.shape[1]

    # Column normalisation
    col_norm = np.sqrt(np.sum(Theta ** 2, axis=0))
    col_norm[col_norm == 0] = 1.0
    Theta_norm = Theta / col_norm

    # Initial least-squares fit on normalised library
    xi_norm, _, _, _ = np.linalg.lstsq(Theta_norm, acc, rcond=None)

    # Iterative thresholding
    for _ in range(max_iter):
        small = np.abs(xi_norm) < delta
        big   = ~small

        if not np.any(big):
            xi_norm[:] = 0.0
            break

        xi_norm[small] = 0.0

        # Refit on surviving (normalised) columns
        xi_norm[big], _, _, _ = np.linalg.lstsq(
            Theta_norm[:, big], acc, rcond=None)

    # Rescale back to original data scale
    xi = xi_norm / col_norm

    return xi, xi_norm, col_norm, ~small


# ---------------------------------------------------------------------------
# Contribution analysis
# ---------------------------------------------------------------------------
def contribution_analysis(Theta: np.ndarray, xi: np.ndarray):
    """
    Rank library terms by their L²-norm contribution over the trajectory.

        contrib_j = norm(Theta_j(t) * xi_j)

    Parameters
    ----------
    Theta : ndarray  (N, P)   unnormalised library
    xi    : ndarray  (P,)     rescaled coefficients

    Returns
    -------
    contrib  : ndarray  (P,)   contribution magnitudes
    sort_idx : ndarray  (P,)   indices sorted descending by contribution
    """
    contrib  = np.linalg.norm(Theta * xi, axis=0)   # element-wise, then norm
    sort_idx = np.argsort(contrib)[::-1]
    return contrib, sort_idx


def print_contributions(contrib, sort_idx, xi, names=None):
    """Print ranked term contributions."""
    if names is None:
        names = LIBRARY_NAMES
    print("\nSINDy term contributions (STLSQ, ranked):")
    print(f"{'Term':>8}   {'Contribution':>14}   {'Coefficient':>14}")
    print("-" * 42)
    for i in sort_idx:
        print(f"{names[i]:>8}   {contrib[i]:>14.6f}   {xi[i]:>+14.6f}")


# ---------------------------------------------------------------------------
# Convenience: full SINDy run
# ---------------------------------------------------------------------------
def run_sindy(h: np.ndarray, v: np.ndarray, acc: np.ndarray,
              delta: float = 0.05, max_iter: int = 10):
    """
    Build library, run STLSQ, compute contributions.

    Parameters
    ----------
    h, v, acc  : ndarray   smoothed signals from simulate.py
    delta      : float     sparsity threshold
    max_iter   : int       STLSQ iterations

    Returns
    -------
    dict with keys:
        Theta, xi, contrib, sort_idx, active_mask, col_norm
    """
    Theta = build_library(h, v)
    xi, xi_norm, col_norm, active_mask = stlsq(Theta, acc, delta, max_iter)
    contrib, sort_idx = contribution_analysis(Theta, xi)

    return {
        "Theta":       Theta,
        "xi":          xi,
        "xi_norm":     xi_norm,
        "col_norm":    col_norm,
        "contrib":     contrib,
        "sort_idx":    sort_idx,
        "active_mask": active_mask,
    }


def print_model(xi, names=None):
    """Print the identified sparse model equation."""
    if names is None:
        names = LIBRARY_NAMES
    terms = [(names[i], xi[i]) for i in range(len(xi)) if xi[i] != 0.0]
    expr  = " + ".join(f"({c:.4f})·{n}" for n, c in terms)
    print(f"\nIdentified model:  v̇ = {expr}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from simulate import run_pipeline

    # Noise-free run
    data   = run_pipeline(noise_level=0.0)
    result = run_sindy(data["h"], data["v"], data["acc"], delta=0.05)
    print_model(result["xi"])
    print_contributions(result["contrib"], result["sort_idx"], result["xi"])

    # Noisy run
    data_n   = run_pipeline(noise_level=0.03, seed=42)
    result_n = run_sindy(data_n["h"], data_n["v"], data_n["acc"], delta=0.08)
    print("\n--- Noisy run ---")
    print_model(result_n["xi"])
    print_contributions(result_n["contrib"], result_n["sort_idx"], result_n["xi"])

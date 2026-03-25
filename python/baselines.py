"""
baselines.py
------------
Fits three classical model-based acceleration models to estimated
acceleration data via Ordinary Least Squares, then simulates each
model forward in time and computes RMSE against the reference trajectory.

Models
------
  Constant:  a = a0
  Linear:    a = a0 + b1 * v
  Quadratic: a = a0 + b2 * v|v|
"""

import numpy as np


# ---------------------------------------------------------------------------
# OLS fitting
# ---------------------------------------------------------------------------
def fit_baselines(v: np.ndarray, acc: np.ndarray):
    """
    Fit all three drag baselines to (v, acc) via OLS.

    Parameters
    ----------
    v   : ndarray   velocity signal (downward-positive, m/s)
    acc : ndarray   acceleration signal (downward-positive, m/s²)

    Returns
    -------
    params : dict with keys 'constant', 'linear', 'quadratic'
        Each entry is itself a dict with the fitted scalar coefficients.
    """
    N   = len(v)
    one = np.ones(N)

    # 1) Constant: a = a0
    a0_const = np.linalg.lstsq(one[:, None], acc, rcond=None)[0][0]

    # 2) Linear: a = a0 + b1*v
    Phi_lin            = np.column_stack([one, v])
    a0_lin, b1_lin     = np.linalg.lstsq(Phi_lin, acc, rcond=None)[0]

    # 3) Quadratic: a = a0 + b2*v|v|
    Phi_quad           = np.column_stack([one, v * np.abs(v)])
    a0_quad, b2_quad   = np.linalg.lstsq(Phi_quad, acc, rcond=None)[0]

    return {
        "constant":  {"a0": a0_const},
        "linear":    {"a0": a0_lin,  "b1": b1_lin},
        "quadratic": {"a0": a0_quad, "b2": b2_quad},
    }


def print_baselines(params: dict):
    """Pretty-print the fitted model equations."""
    p = params
    print("\nFitted model-based accelerations (downward-positive):")
    print(f"  Constant:  a = {p['constant']['a0']:.4f}")
    print(f"  Linear:    a = {p['linear']['a0']:.4f} + "
          f"({p['linear']['b1']:.4f}) v")
    print(f"  Quadratic: a = {p['quadratic']['a0']:.4f} + "
          f"({p['quadratic']['b2']:.4f}) v|v|")


# ---------------------------------------------------------------------------
# Forward simulation of each fitted model
# ---------------------------------------------------------------------------
def simulate_baselines(params: dict, t: np.ndarray, H0: float = 40.0):
    """
    Simulate the three fitted models forward from rest using explicit Euler
    on the same time grid as the reference data.

    Parameters
    ----------
    params : dict   output of fit_baselines()
    t      : ndarray   time vector (camera-rate)
    H0     : float     initial height (m)

    Returns
    -------
    trajectories : dict with keys 'constant', 'linear', 'quadratic'
        Each entry is a dict with 'h' and 'v' arrays.
    """
    N      = len(t)
    dt_cam = t[1] - t[0]

    results = {}

    for name, coeff in params.items():
        h_sim = np.zeros(N)
        v_sim = np.zeros(N)
        h_sim[0] = H0
        v_sim[0] = 0.0

        a0 = coeff["a0"]

        for k in range(N - 1):
            v_k = v_sim[k]

            if name == "constant":
                a_k = a0
            elif name == "linear":
                a_k = a0 + coeff["b1"] * v_k
            elif name == "quadratic":
                a_k = a0 + coeff["b2"] * v_k * abs(v_k)

            v_sim[k + 1] = v_k + a_k * dt_cam
            h_sim[k + 1] = h_sim[k] - v_k * dt_cam

        results[name] = {"h": h_sim, "v": v_sim}

    return results


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------
def compute_rmse(trajectories: dict, h_ref: np.ndarray, v_ref: np.ndarray):
    """
    Compute height and velocity RMSE for each model against the reference.

    Parameters
    ----------
    trajectories : dict   output of simulate_baselines()
    h_ref        : ndarray   reference height
    v_ref        : ndarray   reference velocity

    Returns
    -------
    rmse : dict  {model_name: {"h": float, "v": float}}
    """
    rmse = {}
    for name, traj in trajectories.items():
        rmse[name] = {
            "h": float(np.sqrt(np.mean((traj["h"] - h_ref) ** 2))),
            "v": float(np.sqrt(np.mean((traj["v"] - v_ref) ** 2))),
        }
    return rmse


def print_rmse(rmse: dict):
    """Pretty-print RMSE table."""
    print("\n{:<12}  {:>16}  {:>18}".format(
        "Model", "RMSE height (m)", "RMSE velocity (m/s)"))
    print("-" * 50)
    for name, vals in rmse.items():
        print(f"{name:<12}  {vals['h']:>16.4f}  {vals['v']:>18.4f}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from simulate import run_pipeline

    data   = run_pipeline(noise_level=0.0)
    params = fit_baselines(data["v"], data["acc"])
    print_baselines(params)

    trajs = simulate_baselines(params, data["t"])
    rmse  = compute_rmse(trajs, data["h"], data["v"])
    print_rmse(rmse)

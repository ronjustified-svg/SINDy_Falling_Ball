"""
Simulates a tennis ball dropped from rest under Reynolds-dependent drag
(Brown & Lawler 2003), subsamples at camera rate, optionally adds noise,
smooths with Savitzky-Golay, and estimates velocity and acceleration by
centred finite differences.

Sign convention (matches MATLAB):
    v > 0  downward
    a > 0  downward
    v = -dh/dt,   a = -d²h/dt²
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# Physical / fluid constants
# ---------------------------------------------------------------------------
G   = 9.81       # m/s², gravitational acceleration
RHO = 1.211      # kg/m³, air density
MU  = 1.81e-5    # Pa·s, dynamic viscosity

# Ball properties (tennis ball)
R = 0.033        # radius (m)
M = 0.0567       # mass (kg)
D = 2 * R        # diameter (m)
A = np.pi * R**2 # cross-sectional area (m²)
H0 = 40.0        # initial height (m)


# ---------------------------------------------------------------------------
# Drag coefficient — Brown & Lawler (2003)
# ---------------------------------------------------------------------------
def cd_from_re(re: np.ndarray) -> np.ndarray:
    """
    Brown & Lawler (2003) empirical drag correlation for a sphere:
        CD = (24/Re)(1 + 0.150 Re^0.681) + 0.407 / (1 + 8710/Re)

    Parameters
    ----------
    re : array-like  Reynolds number (scalar or array)

    Returns
    -------
    CD : ndarray
    """
    re = np.atleast_1d(np.asarray(re, dtype=float))
    re = np.clip(re, 1e-6, None)
    cd = (24.0 / re) * (1.0 + 0.150 * re**0.681) + 0.407 / (1.0 + 8710.0 / re)
    return cd


# ---------------------------------------------------------------------------
# Dense ODE integration (explicit Euler)
# ---------------------------------------------------------------------------
def integrate_dense(dt_dense: float = 1e-4, t_max: float = 5.0):
    """
    Integrate the equation of motion with Re-dependent drag at high
    temporal resolution until the ball reaches the ground (h = 0).

    Parameters
    ----------
    dt_dense : float   integration time step (s), default 1e-4
    t_max    : float   safety upper bound on simulation time (s)

    Returns
    -------
    t : ndarray   time vector (s)
    h : ndarray   height (m)
    v : ndarray   velocity, downward-positive (m/s)
    """
    n_max = int(t_max / dt_dense) + 1

    t = np.zeros(n_max)
    h = np.zeros(n_max)
    v = np.zeros(n_max)

    h[0] = H0
    v[0] = 0.0

    k = 0
    for k in range(n_max - 1):
        t[k + 1] = t[k] + dt_dense

        h_k = h[k]
        v_k = v[k]

        if h_k <= 0.0:
            break

        re_k = RHO * abs(v_k) * D / MU
        re_k = max(re_k, 1e-12)

        cd_k = cd_from_re(re_k)[0]
        drag_acc = (0.5 * RHO * cd_k * A / M) * v_k * abs(v_k)

        a_k = G - drag_acc          # dv/dt

        v[k + 1] = v_k + a_k * dt_dense
        h[k + 1] = h_k - v_k * dt_dense   # h decreases when v > 0

    # Trim to the step where ball hit the ground
    t = t[:k + 1]
    h = h[:k + 1]
    v = v[:k + 1]

    return t, h, v


# ---------------------------------------------------------------------------
# Camera subsampling at fps Hz via PCHIP interpolation
# ---------------------------------------------------------------------------
def camera_sample(t_dense, h_dense, v_dense, fps: float = 15.0):
    """
    Subsample the dense trajectory at camera rate using PCHIP interpolation.

    Parameters
    ----------
    t_dense, h_dense, v_dense : ndarray   dense trajectory arrays
    fps : float                           camera frame rate (Hz)

    Returns
    -------
    t_cam, h_cam, v_cam : ndarray
    """
    dt_cam = 1.0 / fps
    t_cam  = np.arange(0.0, t_dense[-1], dt_cam)

    h_cam = PchipInterpolator(t_dense, h_dense)(t_cam)
    v_cam = PchipInterpolator(t_dense, v_dense)(t_cam)

    return t_cam, h_cam, v_cam


# ---------------------------------------------------------------------------
# Add Gaussian measurement noise to height
# ---------------------------------------------------------------------------
def add_noise(h_cam: np.ndarray, noise_level: float = 0.0,
              seed: int = None) -> np.ndarray:
    """
    Corrupt height measurements with zero-mean Gaussian noise.

    Parameters
    ----------
    h_cam       : ndarray   clean height signal
    noise_level : float     standard deviation in metres (0 = noise-free)
    seed        : int       random seed for reproducibility

    Returns
    -------
    h_noisy : ndarray
    """
    if noise_level == 0.0:
        return h_cam.copy()
    rng = np.random.default_rng(seed)
    return h_cam + noise_level * rng.standard_normal(len(h_cam))


# ---------------------------------------------------------------------------
# Smoothing and centred finite-difference derivatives
# ---------------------------------------------------------------------------
def smooth_and_differentiate(t_cam: np.ndarray, h_noisy: np.ndarray,
                              poly_order: int = 3, max_window: int = 35):
    """
    Apply Savitzky-Golay smoothing then estimate velocity and acceleration
    using centred finite differences.  Boundary values are copied from
    their nearest interior neighbours (same as MATLAB code).

    Sign convention: v = -dh/dt,  a = -d²h/dt²  (downward positive).

    Parameters
    ----------
    t_cam      : ndarray   camera-rate time vector
    h_noisy    : ndarray   (possibly noisy) height signal
    poly_order : int       S-G polynomial order (default 3)
    max_window : int       maximum S-G window length (default 35)

    Returns
    -------
    h_smooth : ndarray   smoothed height
    v_est    : ndarray   estimated velocity  (downward +)
    a_est    : ndarray   estimated acceleration (downward +)
    """
    N  = len(h_noisy)
    dt = t_cam[1] - t_cam[0]

    # Window must be odd and <= N
    win = min(max_window, N if N % 2 == 1 else N - 1)
    win = max(win, poly_order + 2 if (poly_order + 2) % 2 == 1
              else poly_order + 3)

    try:
        h_smooth = savgol_filter(h_noisy, window_length=win,
                                 polyorder=poly_order)
    except ValueError:
        h_smooth = h_noisy.copy()

    # Centred finite differences
    v_est = np.zeros(N)
    v_est[1:-1] = -(h_smooth[2:] - h_smooth[:-2]) / (2 * dt)
    v_est[0]    = v_est[1]
    v_est[-1]   = v_est[-2]

    a_est = np.zeros(N)
    a_est[1:-1] = -(h_smooth[2:] - 2 * h_smooth[1:-1] + h_smooth[:-2]) / dt**2
    a_est[0]    = a_est[1]
    a_est[-1]   = a_est[-2]

    return h_smooth, v_est, a_est


# ---------------------------------------------------------------------------
# Convenience: run full pipeline in one call
# ---------------------------------------------------------------------------
def run_pipeline(noise_level: float = 0.0, fps: float = 15.0,
                 dt_dense: float = 1e-4, seed: int = 0):
    """
    Run the complete simulation pipeline.

    Parameters
    ----------
    noise_level : float   height noise std (m); 0 = noise-free
    fps         : float   camera frame rate (Hz)
    dt_dense    : float   dense integration step (s)
    seed        : int     random seed for noise

    Returns
    -------
    dict with keys:
        t, h, v, acc   — smoothed signals ready for SINDy / baselines
        t_dense, h_dense, v_dense  — full high-resolution trajectory
        Re, CD         — Reynolds number and drag coefficient at camera times
    """
    t_dense, h_dense, v_dense = integrate_dense(dt_dense)
    t_cam, h_cam, v_cam       = camera_sample(t_dense, h_dense, v_dense, fps)
    h_noisy                   = add_noise(h_cam, noise_level, seed)
    h_smooth, v_est, a_est    = smooth_and_differentiate(t_cam, h_noisy)

    Re = RHO * np.abs(v_est) * D / MU
    CD = cd_from_re(Re)

    return {
        "t": t_cam, "h": h_smooth, "v": v_est, "acc": a_est,
        "t_dense": t_dense, "h_dense": h_dense, "v_dense": v_dense,
        "Re": Re, "CD": CD,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = run_pipeline(noise_level=0.0)
    print(f"Noise-free run:  {len(data['t'])} camera samples, "
          f"t_end = {data['t'][-1]:.2f} s, "
          f"max Re = {data['Re'].max():.2e}")

    data_noisy = run_pipeline(noise_level=0.03, seed=42)
    print(f"Noisy run:       {len(data_noisy['t'])} camera samples, "
          f"t_end = {data_noisy['t'][-1]:.2f} s")

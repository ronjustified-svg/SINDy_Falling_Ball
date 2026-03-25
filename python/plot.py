"""
Generates all figures:

    1.  CD vs Re (Brown-Lawler curve)
    2.  Height trajectories: true vs. model-based (noise-free + noisy)
    3.  Velocity trajectories: true vs. model-based (noise-free + noisy)
    4.  SINDy candidate function contributions (noise-free + noisy)

Figures are saved to figures/ and also displayed if running interactively.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from simulate import run_pipeline
from baselines import fit_baselines, simulate_baselines
from sindy import run_sindy, LIBRARY_NAMES

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 1. CD vs Re
# ---------------------------------------------------------------------------
def plot_cd_re(data):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(data["Re"], data["CD"], "-o", lw=2, ms=4, color="#e05c2a",
              markerfacecolor="white", label="Brown–Lawler (2003)")
    ax.set_xlabel("Reynolds number, Re")
    ax.set_ylabel("Drag coefficient, $C_D$")
    ax.set_title("$C_D$ vs Re (Brown–Lawler, simulated trajectory)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    _save(fig, "CD_Re_Curve.png")
    return fig


# ---------------------------------------------------------------------------
# 2 & 3.  Height and velocity: true vs. baselines
# ---------------------------------------------------------------------------
def plot_trajectories(data, noise_label, suffix):
    params = fit_baselines(data["v"], data["acc"])
    trajs  = simulate_baselines(params, data["t"])

    styles = {
        "constant":  ("--",  "#e05c2a", "Const accel"),
        "linear":    ("-.",  "#2a7fe0", "Linear drag"),
        "quadratic": (":",   "#2ac462", "Quadratic drag"),
    }

    for var, ylabel, fname in [
        ("h", "Height (m)",         f"Ht_{suffix}.png"),
        ("v", "Velocity (m/s, ↓+)", f"Vt_{suffix}.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(data["t"], data[var], "k", lw=2, label="True (Re-dependent)")
        for name, (ls, clr, lbl) in styles.items():
            ax.plot(data["t"], trajs[name][var], ls=ls, lw=1.4,
                    color=clr, label=lbl)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        title_var = "Height" if var == "h" else "Velocity"
        ax.set_title(f"{title_var}: true vs. simplified model-based "
                     f"trajectories\n({noise_label})")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        _save(fig, fname)


# ---------------------------------------------------------------------------
# 4.  SINDy candidate function contributions
# ---------------------------------------------------------------------------
def plot_contributions(data, delta, noise_label, suffix):
    result = run_sindy(data["h"], data["v"], data["acc"], delta=delta)
    ci     = result["contrib"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(ci)), ci, color="#3399cc", edgecolor="none")
    ax.set_xticks(range(len(ci)))
    ax.set_xticklabels(LIBRARY_NAMES, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Contribution (L² norm)")
    ax.set_title(f"Candidate Function Contributions  "
                 f"(δ = {delta:.3f}, {noise_label})")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, f"Dominant_Candidate_Functions_{suffix}.png")
    return fig


# ---------------------------------------------------------------------------
# Main: generate all figures
# ---------------------------------------------------------------------------
def generate_all(show: bool = False):
    print("Running noise-free pipeline ...")
    data_clean = run_pipeline(noise_level=0.0)

    print("Running noisy pipeline (σ = 0.03 m) ...")
    data_noisy = run_pipeline(noise_level=0.03, seed=42)

    print("\nGenerating figures ...")

    # 1. CD vs Re (noise-free is fine — Re comes from estimated v)
    plot_cd_re(data_clean)

    # 2 & 3. Trajectories
    plot_trajectories(data_clean, noise_label="noise-free",      suffix="NoNoise")
    plot_trajectories(data_noisy, noise_label="noise σ = 0.03 m", suffix="Noise")

    # 4. SINDy contributions
    plot_contributions(data_clean, delta=0.05,
                       noise_label="noise-free",       suffix="No_Noise")
    plot_contributions(data_noisy, delta=0.08,
                       noise_label="noise σ = 0.03 m", suffix="with_Noise")

    print("\nDone. All figures saved to python/figures/")

    if show:
        plt.show()


if __name__ == "__main__":
    generate_all(show=True)

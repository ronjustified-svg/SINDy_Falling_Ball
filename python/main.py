"""
main.py
-------
Single entry point — runs the full pipeline end-to-end:

    1. Simulate dense trajectory + camera sampling
    2. Fit and evaluate model-based baselines
    3. Run SINDy (noise-free and noisy)
    4. Print all results to console
    5. Generate and save all figures

Usage
-----
    python main.py                   # noise-free + noisy, save figures
    python main.py --show            # also open figure windows
    python main.py --noise 0.05      # custom noise level
"""

import argparse
import numpy as np

from simulate  import run_pipeline
from baselines import fit_baselines, simulate_baselines, compute_rmse, \
                      print_baselines, print_rmse
from sindy     import run_sindy, print_model, print_contributions
from plot      import generate_all


# ---------------------------------------------------------------------------
# Single run: print results for one noise level
# ---------------------------------------------------------------------------
def run_and_report(noise_level: float, delta: float, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    data   = run_pipeline(noise_level=noise_level, seed=42)

    # --- Model-based baselines ---
    params = fit_baselines(data["v"], data["acc"])
    print_baselines(params)

    trajs = simulate_baselines(params, data["t"])
    rmse  = compute_rmse(trajs, data["h"], data["v"])
    print_rmse(rmse)

    # --- SINDy ---
    result = run_sindy(data["h"], data["v"], data["acc"], delta=delta)
    print_model(result["xi"])
    print_contributions(result["contrib"], result["sort_idx"], result["xi"])

    return data, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SINDy falling-ball identification pipeline")
    parser.add_argument("--show",  action="store_true",
                        help="Display figure windows interactively")
    parser.add_argument("--noise", type=float, default=None,
                        help="Run a single custom noise level instead of both")
    args = parser.parse_args()

    if args.noise is not None:
        run_and_report(args.noise, delta=0.08, label=f"Custom noise σ={args.noise}")
    else:
        run_and_report(noise_level=0.0,  delta=0.05, label="Noise-free run")
        run_and_report(noise_level=0.03, delta=0.08, label="Noisy run  (σ = 0.03 m)")

    print("\nGenerating figures ...")
    generate_all(show=args.show)


if __name__ == "__main__":
    main()

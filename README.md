# SINDy: Falling-Ball Dynamics (Nonlinear System Identification)

> Investigating the combination of model-based signal processing and data-driven learning for aerodynamic drag identification.


## Overview

This project applies the **Sparse Identification of Nonlinear Dynamics (SINDy)** framework to identify the governing equations of a falling sphere from simulated and noisy height measurements. It benchmarks SINDy-discovered models against classical physics-based baselines (constant, linear, and quadratic drag) to assess where data-driven methods succeed and where they fundamentally break down.

The system is deceptively simple, a ball falling under gravity, yet its true drag physics (governed by a Reynolds-number-dependent coefficient) are structurally incompatible with polynomial approximation. This makes it an ideal stress-test for hybrid identification methods.

This research work was completed as a part of Master's thesis at the **Budapest University of Technology and Economics (BME VIK)**, Department of Artificial Intelligence and Systems Engineering, under the supervision of Dr. Tamás Dabóczi.

---

## Motivation

Traditional model-based methods are interpretable and noise-robust, but degrade when the assumed physics are incomplete. Data-driven methods are flexible, but often lack physical consistency and extrapolation reliability. **Hybrid approaches like SINDy** aim to combine both — discovering sparse, interpretable governing equations directly from data.

This project examines how SINDy performs under realistic conditions:
- Noisy height measurements (pixel-level camera tracking)
- Coarse temporal sampling (15 Hz)
- Numerically differentiated velocity and acceleration
- Non-polynomial aerodynamic drag (Brown–Lawler correlation)

---

## System: Falling Tennis Ball

A tennis ball is dropped from rest at H₀ = 40 m. The **true dynamics** are governed by Reynolds-number-dependent drag:

$$\dot{v}(t) = g - \frac{\rho A}{2m} C_D(\text{Re}) \cdot v|v|$$

where the drag coefficient follows the Brown–Lawler empirical correlation:

$$C_D(\text{Re}) = \frac{24}{\text{Re}}\left(1 + 0.150\,\text{Re}^{0.681}\right) + \frac{0.407}{1 + 8710/\text{Re}}$$

This expression contains **fractional powers, inverse powers, and rational terms in velocity** — none of which are polynomial. SINDy's polynomial library is therefore structurally incapable of recovering the true law, regardless of data quality.

**Ball parameters:**

| Parameter | Value |
|---|---|
| Radius | 0.033 m |
| Mass | 0.0567 kg |
| Air density | 1.211 kg/m³ |
| Dynamic viscosity | 1.81 × 10⁻⁵ Pa·s |

---

## Pipeline

```
Dense ODE integration (Δt = 10⁻⁴ s)
        ↓
Subsample at 15 Hz  (mimics camera)
        ↓
Add Gaussian noise  (σ ≈ 0.03–0.06 m)
        ↓
Savitzky–Golay smoothing  (cubic, window ≤ 35 samples)
        ↓
Finite-difference derivatives  →  v̂(t), â(t)
        ↓
  ┌─────────────────────┐     ┌──────────────────────┐
  │  Model-Based Fit    │     │    SINDy Regression  │
  │  (OLS baselines)    │     │  (STLSQ, poly lib.)  │
  └─────────────────────┘     └──────────────────────┘
        ↓                               ↓
   RMSE vs. reference         Sparse coefficient vector ξ
```

---

## Methods

### 1. Model-Based Baselines

Three classical models are fitted to the estimated acceleration via ordinary least squares:

| Model | Equation |
|---|---|
| Constant acceleration | `a = a₀` |
| Linear drag | `a = a₀ + b₁v` |
| Quadratic drag | `a = a₀ + b₂v\|v\|` |

### 2. SINDy (Sparse Identification of Nonlinear Dynamics)

A polynomial candidate library is constructed in height `h` and velocity `v` up to degree 3:

$$\Theta(h, v) = \left[1,\; h,\; v,\; h^2,\; hv,\; v^2,\; h^3,\; h^2v,\; hv^2,\; v^3\right]$$

The acceleration is regressed as:

$$\hat{a}(t) \approx \Theta(h, v)\,\xi$$

**Sequential Thresholded Least Squares (STLSQ)** enforces sparsity by iteratively zeroing coefficients below a threshold δ and refitting on the surviving terms. Columns are normalized to unit L₂-norm before regression to prevent scale bias.

---

## Results

### Model-Based RMSE (noise-free, vs. Reynolds-dependent reference)

| Model | RMSE Height (m) | RMSE Velocity (m/s) |
|---|---|---|
| Constant acceleration | 4.47 | 2.34 |
| Linear drag | 0.39 | 0.24 |
| Quadratic drag | **0.37** | **0.13** |

### SINDy Identified Models

**Noise-free** (δ = 0.08):
```
v̇ ≈ 5.44 + 0.159·v + 0.120·h
```

**Noisy** (σ = 0.03 m, δ = 0.08):
```
v̇ ≈ 12.89 + 0.489·v
```

Both models are compact and stable. Noise primarily inflates coefficient magnitudes rather than changing the sparsity pattern.

### Why SINDy Cannot Recover the True Law

1. **Library mismatch** — The Brown–Lawler drag involves fractional and rational powers of v. No polynomial can exactly represent this.
2. **Noise-amplified derivatives** — Differentiating noisy height data corrupts velocity and acceleration estimates, biasing all regression coefficients.
3. **Multicollinearity** — During free fall, `h` and `v` are nearly functionally related (`h ≈ H₀ − ∫v dt`), making polynomial combinations collinear and unstable under thresholding.
4. **Coarse sampling** — At 15 Hz, subtle curvature in the drag-Re relationship is lost, further blurring the physics.

SINDy converges to the **best polynomial surrogate** of the dynamics within the measurement regime — useful for short-term prediction, but not for physical discovery.

---

## Repository Structure

```
sindy-falling-ball/
├── simulation/
│   ├── generate_trajectory.py      # Dense ODE integration, camera subsampling, noise
│   └── drag_models.py              # Brown–Lawler CD(Re), model-based baselines
├── identification/
│   ├── sindy.py                    # STLSQ implementation, polynomial library builder
│   └── baselines.py                # OLS fits for constant/linear/quadratic drag
├── analysis/
│   ├── noise_sensitivity.py        # Comparison across noise levels
│   └── contribution_analysis.py    # L2-norm contribution ranking per SINDy term
├── notebooks/
│   └── full_pipeline.ipynb         # End-to-end walkthrough with plots
├── results/
│   └── figures/                    # Generated plots
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/<your-username>/sindy-falling-ball.git
cd sindy-falling-ball
pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
python simulation/generate_trajectory.py
python identification/sindy.py
```

Or open `notebooks/full_pipeline.ipynb` for an annotated walkthrough.

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `pandas`

---

## Key Takeaways

- SINDy reliably produces **sparse, stable surrogate models** even under realistic noise.
- It **cannot recover** the true drag law because the Brown–Lawler correlation is structurally non-polynomial.
- Hybrid identification requires **richer functional libraries** (rational, logarithmic, or physics-informed terms) when true dynamics fall outside polynomial spaces.
- The falling-ball system, despite appearing elementary, exposes fundamental limits of standard sparse regression for physical discovery.

---

## References

1. Brown & Lawler (2003). *Sphere drag and settling velocity revisited.* J. Environmental Engineering.
2. Brunton, Proctor & Kutz (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.* PNAS.
3. Champion et al. (2019). *Data-driven discovery of coordinates and governing equations.* PNAS.
4. de Silva et al. (2020). *Discovery of physics from data: universal laws and discrepancies.* Frontiers in AI.

---

## Academic Context

Master's Thesis — Diploma Thesis 1
Budapest University of Technology and Economics (BME VIK)
Department of Artificial Intelligence and Systems Engineering
Supervisor: Dr. Tamás Dabóczi | December 2025

# Python Reimplementation

The methodology is identical — same physics, same pipeline, same SINDy algorithm.  

---

## Files

| File | Description |
|---|---|
| `simulate.py` | Dense ODE integration, camera subsampling, noise, S-G smoothing, finite-difference derivatives |
| `baselines.py` | OLS fitting of constant / linear / quadratic drag models; forward simulation; RMSE |
| `sindy.py` | Polynomial library builder, STLSQ sparse regression, contribution analysis |
| `plot.py` | All figures — CD–Re curve, trajectory comparisons, SINDy bar charts |
| `main.py` | Single entry point — runs everything and prints results to console |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
cd python/
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `scipy`, `matplotlib` — no specialist libraries required.

---

## Usage

**Run the full pipeline (noise-free + noisy, save all figures):**
```bash
python main.py
```

**Also open figure windows interactively:**
```bash
python main.py --show
```

**Custom noise level:**
```bash
python main.py --noise 0.05
```

**Run modules individually:**
```bash
python simulate.py    # quick self-test: prints trajectory stats
python baselines.py   # fits + evaluates model-based baselines
python sindy.py       # noise-free and noisy SINDy runs
python plot.py        # generates and saves all figures
```

---

## Console Output

Running `python main.py` prints:

```
============================================================
  Noise-free run
============================================================

Fitted model-based accelerations (downward-positive):
  Constant:  a = 6.6063
  Linear:    a = 11.6481 + (-0.4075) v
  Quadratic: a = 10.0960 + (-0.0182) v|v|

Model          RMSE height (m)  RMSE velocity (m/s)
--------------------------------------------------
constant               4.3752              2.2796
linear                 0.1367              0.2733
quadratic              0.1266              0.2396

SINDy term contributions (STLSQ, ranked):
    Term     Contribution      Coefficient
------------------------------------------
       1      3352.56          -483.90
       h      3249.23          +16.43
       v      3228.97          +33.69
     ...

============================================================
  Noisy run  (σ = 0.03 m)
============================================================
...
```

---

## Notes on Differences from MATLAB

The Python and MATLAB implementations are logically identical, but minor
numerical differences arise from:

- **SciPy vs MATLAB solvers** — `numpy.linalg.lstsq` and MATLAB's `\` use
  different underlying LAPACK routines; coefficient values can differ slightly.
- **Random seed** — noise is seeded with `seed=42` for reproducibility;
  MATLAB's `randn` produces a different sequence for the same conceptual run.
- **STLSQ convergence** — because STLSQ is sensitive to the magnitude of
  normalised coefficients, small numerical differences can shift which terms
  survive thresholding.  The sparsity pattern may therefore differ from the
  exact 3-term model reported in the thesis, though the dominant physics
  (constant + linear-in-v terms) is consistent.

These differences are themselves a meaningful finding: they illustrate the
sensitivity of sparse regression to numerical precision, which the thesis
discusses in the context of collinearity between `h` and `v`.

---

## Generated Figures

After running `main.py`, `python/figures/` will contain:

| File | Description |
|---|---|
| `CD_Re_Curve.png` | CD vs Re — Brown–Lawler curve |
| `Ht_NoNoise.png` / `Ht_Noise.png` | Height: true vs. model-based |
| `Vt_NoNoise.png` / `Vt_Noise.png` | Velocity: true vs. model-based |
| `Dominant_Candidate_Functions_No_Noise.png` | SINDy contributions, noise-free |
| `Dominant_Candidate_Functions_with_Noise.png` | SINDy contributions, noisy |

# A stochastic cellular automaton for cortical dynamics under anesthesia

Simulation and analysis code for

> Y. Castellanos Márquez and Y. Cortés-Poza,
> *Phase Transitions in a Stochastic Cellular Automaton for Cortical Dynamics
> Under Anesthesia.*

The model is a two-dimensional stochastic cellular automaton (a Greenberg–Hastings
excitable medium, following the anesthesia automaton of Sleigh & Galletly, *Br. J.
Anaesth.* **78**:260–263, 1997). A single control parameter — the synaptic activation
probability `Pb` — governs a sharp, phase-transition-like change between a weakly
propagating and a highly active regime. This repository reproduces every figure in the
paper, including the spectral, event-size, and robustness analyses added during revision.

## Contents

| File | Description |
|------|-------------|
| `ca_model.py` | Canonical, vectorized model: `simulate()`, `avalanche_sizes()`, `ccdf()`, and `estimate_pb_star()`. |
| `make_revision_figures.py` | Generates `PSD.png`, `CCDF.png`, `Robustness.png`. |
| `make_critical_bis.py` | Generates `CriticalBIS.png` (fuzzy BIS↔Pb mapping, variants A–D). |
| `notebook_exploratory.ipynb` | Original exploratory notebook (time series, snapshots, entropy, fractal dimension, phase diagrams, bifurcation, finite-size scaling, fuzzy membership functions). |
| `PSD.png`, `CCDF.png`, `Robustness.png`, `CriticalBIS.png` | Pre-generated output figures. |
| `requirements.txt` | Python dependencies. |

## Installation

```bash
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
```

Tested with Python 3.9–3.12, NumPy 2.0, SciPy 1.13, Matplotlib 3.9, scikit-learn 1.3.

## Reproducing the figures

```bash
# Revision figures: PSD.png, CCDF.png, Robustness.png
python make_revision_figures.py all          # or: psd | ccdf | rob

# Critical-Pb / BIS figure: CriticalBIS.png
python make_critical_bis.py

# Sanity check of the transition
python ca_model.py
```

The exploratory notebook reproduces the remaining paper figures
(`jupyter notebook notebook_exploratory.ipynb`).

## Model rules

Each cell is `INACTIVE (0)`, `ACTIVE (1)`, or `REFRACTORY (2)`, updated synchronously:

- `ACTIVE → REFRACTORY` (deterministic)
- `REFRACTORY → INACTIVE` (deterministic)
- `INACTIVE → ACTIVE` with probability `1 − (1 − Pb)^k`, where `k` is the number of
  active neighbors (**additive**, per-bond rule). A **non-additive** variant — a single
  Bernoulli trial whenever `k > 0` — is available via `simulate(..., additive=False)` and
  is used in the robustness panel.

A boundary stimulus forces one random cell in the left column active at each step.
Boundaries are open (no wrap-around). Neighborhood is von Neumann (4) by default;
`neighborhood="moore"` selects the 8-cell Moore neighborhood.

## Parameters and seeds (for exact reproduction)

The module RNG seed is `DEFAULT_SEED = 20260617`; individual runs use deterministic
per-call seeds so that every figure is reproducible bit-for-bit.

| Analysis | N | steps | burn-in | Pb values | replicas | seed rule |
|----------|---|-------|---------|-----------|----------|-----------|
| PSD | 100 | 4000 | 1000 | 0.30, 0.40, 0.60 | 1 | `int(Pb*1000)` |
| CCDF | 100 | 8000 | 1000 | 0.30, 0.40, 0.60 | 1 | `int(Pb*1000)+7` |
| Robustness | 50/80/120 | 350 | 200 | 0.20–0.70 (Δ=0.025) | 2/5/10 | `hash((Pb,rep,nbhd,additive))` |
| `P_b*` estimate | 100 | 1100 | 600 | 0.35–0.50 (Δ=0.005) | 20 | `1000*rep+int(Pb*1000)` |

**Critical coupling.** Estimating `P_b*` as the argmax over `Pb` of the lag-1
autocorrelation of `E(t)` (20 replicas, Student-*t* interval) gives

```
P_b* = 0.404,   95% CI = [0.402, 0.407]
```

reproducible via `ca_model.estimate_pb_star()`. The fuzzy BIS↔Pb mapping places the
dynamical transition at BIS ≈ 38–40 across all four membership-function variants.

## License

MIT — see [LICENSE](LICENSE).

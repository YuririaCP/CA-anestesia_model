"""
Generate the figures added in the major revision (Reviewer #2):

  PSD.png         power spectral density of E(t), sub/critical/super regimes
  CCDF.png        complementary cumulative distribution of event sizes
  Robustness.png  robustness of the transition to N, replicas, neighborhood,
                  and additive vs non-additive activation rule

Usage:
    python make_revision_figures.py [psd|ccdf|rob|all]

All figures are written to the current directory. The critical reference coupling is
Pb=0.40, consistent with the estimate P_b* = 0.404 (see README).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from ca_model import simulate, avalanche_sizes, ccdf

OUT = "."

REGIMES = [(0.30, "Subcritical $P_b=0.30$", "tab:blue"),
           (0.40, "Critical $P_b=0.40$", "tab:red"),
           (0.60, "Supercritical $P_b=0.60$", "tab:green")]


def fig_psd():
    print("[PSD] simulating long series...")
    N, steps, burn = 100, 4000, 1000
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    out = {}
    for Pb, label, color in REGIMES:
        E = simulate(Pb, N=N, steps=steps, burn_in=burn, seed=int(Pb * 1000))
        E = E - E.mean()
        f, P = welch(E, fs=1.0, nperseg=min(1024, len(E)))
        m = f > 0
        ax.loglog(f[m], P[m], color=color, label=label, lw=1.3)
        out[Pb] = (f[m], P[m])
    fc, Pc = out[0.40]
    band = (fc > 0.01) & (fc < 0.2)
    if band.sum() > 3:
        slope, intercept = np.polyfit(np.log10(fc[band]), np.log10(Pc[band]), 1)
        ax.loglog(fc[band], 10 ** intercept * fc[band] ** slope, "k--", lw=1.0,
                  label=f"Power-law fit (slope ${slope:.2f}$)")
        print(f"[PSD] critical slope in band 0.01-0.2: {slope:.3f}")
    ax.set_xlabel("Frequency $f$ (cycles/step)")
    ax.set_ylabel("Power spectral density $S_E(f)$")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Power spectral density of $E(t)$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "PSD.png"), dpi=200)
    print("[PSD] saved PSD.png")


def fig_ccdf():
    print("[CCDF] simulating...")
    N, steps, burn = 100, 8000, 1000
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    crit_sizes = None
    for Pb, label, color in REGIMES:
        E = simulate(Pb, N=N, steps=steps, burn_in=burn, seed=int(Pb * 1000) + 7)
        s = avalanche_sizes(E)
        if len(s) < 5:
            print(f"[CCDF] Pb={Pb}: few avalanches ({len(s)})")
            continue
        xs, c = ccdf(s)
        ax.loglog(xs, c, color=color, marker=".", ls="none", ms=4, label=label)
        if abs(Pb - 0.40) < 1e-9:
            crit_sizes = s
    if crit_sizes is not None and len(crit_sizes) > 20:
        smin = np.percentile(crit_sizes, 20)
        tail = crit_sizes[crit_sizes >= smin]
        alpha = 1.0 + len(tail) / np.sum(np.log(tail / smin))
        xs, c = ccdf(crit_sizes)
        ref = xs >= smin
        ax.loglog(xs[ref], c[ref][0] * (xs[ref] / smin) ** (-(alpha - 1)), "k--",
                  lw=1.0, label=fr"Power-law guide $\alpha\approx{alpha:.2f}$")
        print(f"[CCDF] critical exponent estimate alpha={alpha:.3f}")
    ax.set_xlabel("Avalanche size $S$")
    ax.set_ylabel("CCDF  $P(S' \\geq S)$")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Complementary cumulative distribution of event sizes")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "CCDF.png"), dpi=200)
    print("[CCDF] saved CCDF.png")


def _order_param(pbs, N=80, steps=350, burn=200, reps=3,
                 neighborhood="vonneumann", additive=True):
    means = []
    for Pb in pbs:
        vals = [simulate(Pb, N=N, steps=steps, burn_in=burn,
                         seed=hash((Pb, rep, neighborhood, additive)) % (2 ** 31),
                         neighborhood=neighborhood, additive=additive).mean() / (N * N)
                for rep in range(reps)]
        means.append(np.mean(vals))
    return np.asarray(means)


def fig_robustness():
    print("[ROB] robustness panel...")
    pbs = np.round(np.arange(0.20, 0.71, 0.025), 3)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axs[0, 0]
    for N in [50, 80, 120]:
        ax.plot(pbs, _order_param(pbs, N=N), marker="o", ms=3, label=f"$N={N}$")
    ax.set_title("(a) Lattice size $N$")

    ax = axs[0, 1]
    for reps in [2, 5, 10]:
        ax.plot(pbs, _order_param(pbs, reps=reps), marker="s", ms=3,
                label=f"$N_{{rep}}={reps}$")
    ax.set_title("(b) Number of replicas")

    ax = axs[1, 0]
    for nb, lab in [("vonneumann", "von Neumann (4)"), ("moore", "Moore (8)")]:
        ax.plot(pbs, _order_param(pbs, neighborhood=nb), marker="^", ms=3, label=lab)
    ax.set_title("(c) Neighborhood")

    ax = axs[1, 1]
    for add, lab in [(True, "Additive (per-bond)"), (False, "Non-additive (single draw)")]:
        ax.plot(pbs, _order_param(pbs, additive=add), marker="d", ms=3, label=lab)
    ax.set_title("(d) Activation rule")

    for ax in axs.ravel():
        ax.axvline(0.40, ls=":", color="k", lw=0.8)
        ax.set_xlabel("$P_b$")
        ax.set_ylabel(r"$\langle E\rangle / N^2$")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Robustness of the transition to modeling choices", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Robustness.png"), dpi=200)
    print("[ROB] saved Robustness.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "psd"):
        fig_psd()
    if which in ("all", "ccdf"):
        fig_ccdf()
    if which in ("all", "rob"):
        fig_robustness()
    print("DONE")

"""
Stochastic cellular automaton for cortical dynamics under anesthesia.

Canonical, vectorized implementation of the model described in

    Y. Castellanos Marquez and Y. Cortes-Poza,
    "Phase Transitions in a Stochastic Cellular Automaton for Cortical
     Dynamics Under Anesthesia".

The rules are those of a Greenberg-Hastings excitable medium, following the
anesthesia automaton of Sleigh & Galletly (Br. J. Anaesth. 78:260-263, 1997).
Each cell is INACTIVE (0), ACTIVE (1), or REFRACTORY (2):

    ACTIVE      -> REFRACTORY        (deterministic)
    REFRACTORY  -> INACTIVE          (deterministic)
    INACTIVE    -> ACTIVE            with probability 1 - (1 - Pb)^k,
                                     where k is the number of active neighbors
                                     (additive, per-bond rule). A non-additive
                                     variant (single Bernoulli trial when k>0)
                                     is available via additive=False.

A boundary stimulus forces one random cell in the left column active at each
step. Boundaries are open (no wrap-around).
"""
import numpy as np

INACTIVE, ACTIVE, REFRACTORY = 0, 1, 2

VON_NEUMANN = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOORE = VON_NEUMANN + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# Global RNG seed used across the figure scripts for reproducibility.
DEFAULT_SEED = 20260617


def active_neighbor_count(active, neigh):
    """Number of active neighbors per cell, open boundaries (no wrap)."""
    k = np.zeros(active.shape, dtype=np.int16)
    N = active.shape[0]
    for dx, dy in neigh:
        shifted = np.zeros_like(active)
        shifted[max(0, -dx):N - max(0, dx), max(0, -dy):N - max(0, dy)] = \
            active[max(0, dx):N - max(0, -dx), max(0, dy):N - max(0, -dy)]
        k += shifted.astype(np.int16)
    return k


def simulate(Pb, N=100, steps=1000, burn_in=500, seed=None,
             neighborhood="vonneumann", additive=True):
    """Run the CA and return the global-activity time series E(t) (number of
    active cells) for t >= burn_in.

    Parameters
    ----------
    Pb : float           synaptic activation probability (sole control parameter)
    N : int              lattice side
    steps : int          total time steps
    burn_in : int        steps discarded as transient
    seed : int or None   RNG seed (per call); if None uses a module RNG
    neighborhood : str   "vonneumann" (4) or "moore" (8)
    additive : bool      True: per-bond rule 1-(1-Pb)^k; False: single draw if k>0
    """
    r = np.random.default_rng(seed) if seed is not None \
        else np.random.default_rng(DEFAULT_SEED)
    neigh = VON_NEUMANN if neighborhood == "vonneumann" else MOORE
    grid = np.zeros((N, N), dtype=np.int8)
    E = []
    for t in range(steps):
        grid[r.integers(N), 0] = ACTIVE          # boundary stimulus
        if t >= burn_in:
            E.append(int(np.sum(grid == ACTIVE)))
        active = (grid == ACTIVE)
        inactive = (grid == INACTIVE)
        k = active_neighbor_count(active, neigh)
        if additive:
            p_act = 1.0 - (1.0 - Pb) ** k
        else:
            p_act = np.where(k > 0, Pb, 0.0)
        draw = r.random((N, N)) < p_act
        new = np.zeros_like(grid)
        new[active] = REFRACTORY
        new[inactive & draw] = ACTIVE
        grid = new
    return np.asarray(E, dtype=float)


def avalanche_sizes(E, theta=None):
    """Event sizes: contiguous excursions of E(t) above threshold theta
    (default: stationary median), size = integrated activity above theta."""
    if theta is None:
        theta = np.median(E)
    sizes, cur = [], 0.0
    for val in E:
        if val > theta:
            cur += (val - theta)
        elif cur > 0:
            sizes.append(cur)
            cur = 0.0
    if cur > 0:
        sizes.append(cur)
    return np.asarray(sizes)


def ccdf(x):
    """Empirical complementary cumulative distribution function."""
    xs = np.sort(x)
    c = 1.0 - np.arange(len(xs)) / len(xs)
    return xs, c


def estimate_pb_star(grid=None, n_rep=20, N=100, steps=1100, burn_in=600):
    """Estimate the critical coupling P_b* as the argmax over Pb of the lag-1
    Pearson autocorrelation of E(t), with a Student-t 95% confidence interval.

    Returns (pb_star, ci_low, ci_high).
    """
    from scipy import stats
    if grid is None:
        grid = np.round(np.arange(0.35, 0.501, 0.005), 4)
    argmaxes = []
    for rep in range(n_rep):
        rs = []
        for Pb in grid:
            E = simulate(Pb, N=N, steps=steps, burn_in=burn_in,
                         seed=1000 * rep + int(Pb * 1000))
            e0, e1 = E[:-1], E[1:]
            r = np.corrcoef(e0, e1)[0, 1] if e0.std() > 0 and e1.std() > 0 else 0.0
            rs.append(r)
        argmaxes.append(grid[int(np.argmax(rs))])
    argmaxes = np.asarray(argmaxes)
    m, s = argmaxes.mean(), argmaxes.std(ddof=1)
    half = stats.t.ppf(0.975, n_rep - 1) * s / np.sqrt(n_rep)
    return float(m), float(m - half), float(m + half)


if __name__ == "__main__":
    # Quick demonstration of the transition.
    for Pb in [0.25, 0.35, 0.40, 0.43, 0.50, 0.60]:
        E = simulate(Pb, N=80, steps=400, burn_in=250, seed=1)
        print(f"Pb={Pb:.2f}  <E>/N^2 = {E.mean() / 6400:.4f}")

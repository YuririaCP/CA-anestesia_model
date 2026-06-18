"""
Generate CriticalBIS.png: the Mamdani fuzzy mapping BIS -> Pb for four
membership-function variants (A-D), with the critical Pb band overlaid and the
BIS intersection points marked. The critical band is [0.402, 0.407], centered on
the estimate P_b* = 0.404 (see README).

Usage:
    python make_critical_bis.py
"""
import numpy as np
import matplotlib.pyplot as plt

BAND = (0.402, 0.407)


def make_gauss(a, b, c):
    s = (c - a) / 4.0
    return lambda x: np.exp(-0.5 * ((x - b) / s) ** 2)


def make_sigmoid(a, b, c, invert=False):
    al = 4.0 / (c - a)
    return (lambda x: 1 / (1 + np.exp(al * (x - b)))) if invert \
        else (lambda x: 1 / (1 + np.exp(-al * (x - b))))


bis_ranges = {'Awake': (90, 95, 100), 'Light hypnosis': (60, 75, 90),
              'Surgical range': (40, 50, 60), 'Deep hypnosis': (0, 20, 40)}
pb_sets = {
    'A': {'Very low': (0, .1, .3), 'Low': (.2, .3, .5), 'Medium': (.4, .6, .8), 'High': (.7, .85, 1)},
    'B': {'Very low': (0, .1, .2), 'Low': (.15, .25, .4), 'Medium': (.35, .55, .75), 'High': (.65, .9, 1)},
    'C': {'Very low': (0, .2, .4), 'Low': (.3, .45, .5), 'Medium': (.55, .75, .95), 'High': (.8, .92, 1)},
    'D': {'Very low': (0, .2, .4), 'Low': (.4, .5, .7), 'Medium': (.55, .7, .95), 'High': (.8, .92, 1)},
}
rules = [('Deep hypnosis', ['Very low']), ('Surgical range', ['Low', 'Medium']),
         ('Light hypnosis', ['Medium', 'High']), ('Awake', ['High'])]


def build(pr):
    bis = {'Awake': make_sigmoid(*bis_ranges['Awake']),
           'Light hypnosis': make_gauss(*bis_ranges['Light hypnosis']),
           'Surgical range': make_gauss(*bis_ranges['Surgical range']),
           'Deep hypnosis': make_sigmoid(*bis_ranges['Deep hypnosis'], invert=True)}
    pb = {'Very low': make_sigmoid(*pr['Very low'], invert=True), 'Low': make_gauss(*pr['Low']),
          'Medium': make_gauss(*pr['Medium']), 'High': make_sigmoid(*pr['High'])}
    return bis, pb


def mapper(bisv, bis, pb, res=4000):
    al = {l: f(bisv) for l, f in bis.items()}
    y = np.linspace(0, 1, res)
    agg = np.zeros_like(y)
    for bl, pls in rules:
        a = al[bl]
        for pl in pls:
            agg = np.maximum(agg, np.minimum(a, pb[pl](y)))
    return (y * agg).sum() / agg.sum() if agg.sum() > 0 else 0.0


colors = {'A': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green', 'D': 'tab:purple'}
bisv = np.linspace(20, 60, 401)


def cross(pv, target):
    idx = np.where(np.diff(np.sign(pv - target)))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return bisv[i] + (target - pv[i]) * (bisv[i + 1] - bisv[i]) / (pv[i + 1] - pv[i])


fig, ax = plt.subplots(figsize=(8, 6))
curves = {}
for name, pr in pb_sets.items():
    bis, pb = build(pr)
    curves[name] = np.array([mapper(b, bis, pb) for b in bisv])
    ax.plot(bisv, curves[name], color=colors[name], lw=2, label=f'Variant {name}')

ax.axhspan(BAND[0], BAND[1], color='violet', alpha=0.3,
           label=f'Critical $P_b$ band [{BAND[0]:.3f},{BAND[1]:.3f}]')

ann_y = {'D': 0.255, 'C': 0.225, 'B': 0.195, 'A': 0.165}
for name in ['A', 'B', 'C', 'D']:
    pv = curves[name]
    lo, hi, bc = cross(pv, BAND[0]), cross(pv, BAND[1]), cross(pv, np.mean(BAND))
    if bc is not None:
        ax.plot(bc, np.mean(BAND), marker='*', ms=18, color=colors[name],
                markeredgecolor='k', markeredgewidth=0.4, zorder=5)
    ax.text(46, ann_y[name], f'V{name}: [{lo:.1f}, {hi:.1f}]',
            color=colors[name], fontsize=11, fontweight='bold')

ax.set_xlabel('BIS index', fontsize=12)
ax.set_ylabel('$P_b$ (inferred)', fontsize=12)
ax.set_title('BIS ranges associated with the critical $P_b$ interval', fontsize=13)
ax.set_xlim(20, 60)
ax.grid(True, ls='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=10)
fig.tight_layout()
fig.savefig('CriticalBIS.png', dpi=200)
print('saved CriticalBIS.png with band', BAND)

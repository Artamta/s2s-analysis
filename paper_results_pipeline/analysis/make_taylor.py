"""
Taylor diagram (Taylor 2001) summarising forecast skill in one glance: radial
distance = ratio of forecast to observed spatial standard deviation, azimuthal
angle = Pearson correlation, and distance to the reference point (REF, on the
x-axis at radius 1) = centred RMS difference. One panel per variable (TP, Z500
anomaly, T2M), four systems, pooled over all 13 inits x 6 lead weeks x land
points. Reads analysis/scatter_points.npz -> paper/figs/fig14_taylor.{pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'
FIGDIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/paper/figs'
d = np.load(f'{ADIR}/scatter_points.npz')
VARS = ['TP', 'Z500', 'T2M']
VLAB = {'TP': '(a) Precipitation', 'Z500': '(b) Z500 anomaly', 'T2M': '(c) T2M'}
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
COL = {'SPIRE': '#D55E00', 'FuXi': '#0072B2', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}
MK = {'SPIRE': 's', 'FuXi': 'o', 'ECMWF': '^', 'NCEP': 'D'}
LAB = {'SPIRE': 'SPIRE', 'FuXi': 'FuXi-S2S', 'ECMWF': 'ECMWF', 'NCEP': 'NCEP'}
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans', 'savefig.dpi': 300})


def taylor_axes(fig, pos, rmax):
    ax = fig.add_subplot(pos, projection='polar')
    ax.set_thetalim(0, np.pi / 2)
    ax.set_rlim(0, rmax)
    # correlation grid (azimuth)
    corrs = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    ax.set_thetagrids(np.degrees(np.arccos(corrs)), labels=[f'{c:g}' for c in corrs])
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    # std-ratio grid (radius), dashed REF arc at r=1
    th = np.linspace(0, np.pi / 2, 100)
    ax.plot(th, np.ones_like(th), color='0.4', ls='--', lw=1)
    ax.text(np.deg2rad(45), rmax * 1.02, 'correlation', ha='center', va='bottom', fontsize=10, rotation=-45)
    ax.set_xlabel('standard-deviation ratio ($\\sigma_f/\\sigma_o$)', fontsize=10)
    ax.plot([0], [1], 'k*', ms=13, zorder=6)  # REF (obs) at angle 0, r=1
    ax.text(0.02, 1.0, ' REF', fontsize=9, va='bottom')
    return ax


fig = plt.figure(figsize=(16, 5.4))
for k, v in enumerate(VARS):
    o0 = d[f'{v}_SPIRE_obs']; so = o0.std()
    ratios = []
    pts = []
    for m in MODELS:
        f = d[f'{v}_{m}_fcst']; o = d[f'{v}_{m}_obs']
        R = np.corrcoef(f, o)[0, 1]
        ratio = f.std() / o.std()
        ratios.append(ratio); pts.append((m, R, ratio))
    rmax = max(1.3, min(2.2, max(ratios) * 1.15))
    ax = taylor_axes(fig, 131 + k, rmax)
    for m, R, ratio in pts:
        theta = np.arccos(np.clip(R, -1, 1))
        ax.plot(theta, ratio, marker=MK[m], color=COL[m], ms=11, mec='k', mew=0.6,
                ls='', label=LAB[m], zorder=5)
    ax.set_title(VLAB[v] + '\n', fontsize=12, fontweight='bold')
    if k == 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.12), fontsize=9, framealpha=0.9)
fig.suptitle('Taylor diagram: forecast vs ERA5 over India, all 13 inits and lead weeks pooled — JFM 2026',
             fontsize=13.5, fontweight='bold', y=1.02)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{FIGDIR}/fig14_taylor.{ext}', bbox_inches='tight')
print('WROTE fig14_taylor', flush=True)
# print the numbers
for v in VARS:
    print(f'\n{v}:')
    for m in MODELS:
        f = d[f'{v}_{m}_fcst']; o = d[f'{v}_{m}_obs']
        print(f'  {m:6s} R={np.corrcoef(f,o)[0,1]:.3f}  std_ratio={f.std()/o.std():.3f}')
print('TAYLOR_DONE', flush=True)

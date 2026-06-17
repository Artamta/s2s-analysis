"""Shared style config imported by all plot scripts."""
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE    = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.dirname(HERE)          # presentationv3/
ODIR    = os.path.join(HERE, 'plots_results_pres')
os.makedirs(ODIR, exist_ok=True)

MODEL_COLORS = {
    'SPIRE':       '#1a6faf',
    'FuXi':        '#e05c2a',
    'ECMWF':       '#2a9d54',
    'MME':         '#9b59b6',
    'Persistence': '#7f7f7f',
}
MODEL_MARKERS = {
    'SPIRE': 'o', 'FuXi': 's', 'ECMWF': '^', 'MME': 'D', 'Persistence': 'x',
}
VAR_UNITS = {'TP': 'mm day⁻¹', 'Z500': 'gpm'}
VAR_LONG  = {'TP': 'Total Precipitation', 'Z500': '500-hPa Geopotential Height'}
REGION_LABEL = {
    'All India':            'All India',
    'northwest_india':      'NW India',
    'central_india':        'Central India',
    'south_peninsula':      'South Peninsula',
    'east_northeast_india': 'East & NE India',
}
REGIONS  = ['All India', 'northwest_india', 'central_india',
            'south_peninsula', 'east_northeast_india']
WEEKS    = [1, 2, 3, 4, 5, 6]
WLABELS  = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
WLONG    = ['Week 1\n(d1–7)', 'Week 2\n(d8–14)', 'Week 3\n(d15–21)',
            'Week 4\n(d22–28)', 'Week 5\n(d29–35)', 'Week 6\n(d36–42)']

DET_MODELS  = ['SPIRE', 'FuXi', 'ECMWF', 'MME', 'Persistence']
PROB_MODELS = ['SPIRE', 'FuXi', 'ECMWF']


def set_style():
    mpl.rcParams.update({
        'font.size':        11,
        'axes.titlesize':   12,
        'axes.labelsize':   11,
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
        'legend.fontsize':  10,
        'figure.dpi':       150,
        'savefig.dpi':      200,
        'axes.spines.top':  False,
        'axes.spines.right':False,
        'axes.grid':        True,
        'grid.alpha':       0.3,
        'grid.linestyle':   '--',
        'lines.linewidth':  2.0,
        'lines.markersize': 6,
    })


def savefig(fig, name):
    path = os.path.join(ODIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


def model_legend_handles(models=DET_MODELS, ls='-'):
    return [plt.Line2D([0], [0],
                       color=MODEL_COLORS[m],
                       marker=MODEL_MARKERS[m],
                       lw=2, ms=7, ls=ls, label=m)
            for m in models if m in MODEL_COLORS]


def style_week_axis(ax, ylabel='', ylim=None, title=''):
    ax.set_xticks(WEEKS)
    ax.set_xticklabels(WLABELS, fontsize=9)
    ax.set_xlabel('Forecast Week', fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=11, pad=5)
    ax.axhline(0, color='#999', lw=0.8, ls='--', zorder=0)

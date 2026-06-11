"""Headline + MME + persistence + variance-ratio summary from skill_per_init_full.csv."""
import sys
import numpy as np
import pandas as pd
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline')
from utils.verification_extra import bootstrap_ci

CSV = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis/skill_per_init_full.csv'
df = pd.read_csv(CSV)
df['wk'] = df['week'].str.extract(r'(\d)').astype(int)
ALLM = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP', 'MME', 'Persistence']


def table(var, metric, models=ALLM, region='All India'):
    print(f"\n### {var} {metric.upper()} — {region} (mean) ###")
    print(f"{'Model':<12}" + "".join(f" Wk{w}" + " " * 4 for w in range(1, 7)))
    for m in models:
        cells = []
        for w in range(1, 7):
            vals = df[(df.variable == var) & (df.region == region) & (df.model == m) & (df.wk == w)][metric].values
            mean, _, _ = bootstrap_ci(vals)
            cells.append(f"{mean:5.2f}" if np.isfinite(mean) else "  -- ")
        print(f"{m:<12}" + " ".join(f"{c:>6}" for c in cells))


print("=" * 60, "\nFULL RESULTS (incl. MME, persistence, variance)\n", "=" * 60)
print(f"rows={len(df)}  models={sorted(df.model.unique())}")
table('TP', 'pcc')
table('Z500', 'pcc')
table('Z500', 'rmse')

print("\n### MME vs best single system (All India PCC) ###")
for var in ['TP', 'Z500']:
    print(f" {var}:")
    for w in range(1, 5):
        d = df[(df.variable == var) & (df.region == 'All India') & (df.wk == w)]
        singles = {m: d[d.model == m]['pcc'].mean() for m in ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']}
        mme = d[d.model == 'MME']['pcc'].mean()
        best = max(singles, key=singles.get)
        print(f"   Wk{w}: MME={mme:.2f}  best_single={best}({singles[best]:.2f})  delta={mme-singles[best]:+.2f}")

print("\n### Variance ratio fcst_std/obs_std (All India, ensemble smoothing) ###")
for m in ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']:
    row = []
    for w in range(1, 7):
        d = df[(df.variable == 'Z500') & (df.region == 'All India') & (df.model == m) & (df.wk == w)]
        if len(d):
            row.append(f"{(d['fcst_std']/d['obs_std']).mean():.2f}")
        else:
            row.append(" -- ")
    print(f"   {m:<8} " + " ".join(f"Wk{i+1}={row[i]}" for i in range(6)))

print("\n### Persistence-beaten horizon (last week SPIRE PCC > Persistence PCC) ###")
for var in ['TP', 'Z500']:
    msg = []
    for w in range(1, 7):
        s = df[(df.variable == var) & (df.region == 'All India') & (df.model == 'SPIRE') & (df.wk == w)]['pcc'].mean()
        p = df[(df.variable == var) & (df.region == 'All India') & (df.model == 'Persistence') & (df.wk == w)]['pcc'].mean()
        if np.isfinite(s) and np.isfinite(p):
            msg.append(f"Wk{w}:SPIRE{s:.2f}/Pers{p:.2f}{'+' if s>p else '-'}")
    print(f"   {var}: " + "  ".join(msg))
print("\nDONE")

"""Print headline results from skill_per_init.csv (lightweight; no data loading)."""
import sys
import numpy as np
import pandas as pd
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/analysis-code')
from utils.verification_extra import bootstrap_ci, paired_bootstrap_diff

CSV = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis/skill_per_init.csv'
df = pd.read_csv(CSV)
df['wk'] = df['week'].str.extract(r'(\d)').astype(int)
MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']


def table(variable, metric, region='All India'):
    print(f"\n### {variable} {metric.upper()} — {region} (mean [95% CI]) ###")
    print(f"{'Model':<8}" + "".join(f"  Wk{w:<14}" for w in range(1, 7)))
    for m in MODELS:
        cells = []
        for w in range(1, 7):
            vals = df[(df.variable == variable) & (df.region == region) &
                      (df.model == m) & (df.wk == w)][metric].values
            mean, lo, hi = bootstrap_ci(vals)
            cells.append(f"{mean:.2f}[{lo:.2f},{hi:.2f}]" if np.isfinite(mean) else "  --  ")
        print(f"{m:<8}" + "".join(f"  {c:<15}" for c in cells))


def paired(variable, metric):
    print(f"\n### SPIRE - FuXi paired {variable} {metric.upper()} (All India) ###")
    for w in range(1, 7):
        a = df[(df.variable == variable) & (df.region == 'All India') & (df.model == 'SPIRE') & (df.wk == w)]
        b = df[(df.variable == variable) & (df.region == 'All India') & (df.model == 'FuXi') & (df.wk == w)]
        md, lo, hi, p = paired_bootstrap_diff(dict(zip(a.init_date, a[metric])), dict(zip(b.init_date, b[metric])))
        if np.isfinite(md):
            sig = '*' if p < 0.05 else ' '
            print(f"  Wk{w}: dPCC={md:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f} {sig}")


print("=" * 70, "\nHEADLINE RESULTS — JFM 2026 SPIRE benchmark (land, ensemble means)\n", "=" * 70)
print(f"rows={len(df)}  inits={df.init_date.nunique()}  models={sorted(df.model.unique())}")
table('TP', 'pcc')
table('Z500', 'pcc')
table('Z500', 'rmse')
paired('TP', 'pcc')
paired('Z500', 'pcc')
print("\n### Regional mean PCC, Weeks 1-4 ###")
for var in ['TP', 'Z500']:
    print(f"\n {var}:")
    for rg in ['northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']:
        row = [f"{m}={df[(df.variable==var)&(df.region==rg)&(df.model==m)&(df.wk<=4)]['pcc'].mean():.2f}" for m in MODELS]
        print(f"   {rg:<22} " + "  ".join(row))
print("\nDONE")

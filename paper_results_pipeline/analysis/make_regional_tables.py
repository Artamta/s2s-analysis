"""
Region-wise tables (13-init average) for the manuscript:
  * Table: RMSE per IMD region x model (weeks 1-4 mean), per variable
  * Table: PCC  per IMD region x model (weeks 1-4 mean), per variable
  * Table: lead-time error (PCC & RMSE vs week, All India)
Writes LaTeX snippets to analysis/tables_regional.tex and prints them.
"""
import sys
import numpy as np
import pandas as pd
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline')
ADIR = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'

z = pd.read_csv(f'{ADIR}/skill_per_init_full.csv'); z = z[z.variable == 'Z500']
tp = pd.read_csv(f'{ADIR}/skill_tp_corrected.csv'); tp = tp[tp.variable == 'TP']
frames = [tp, z]
try:
    t2 = pd.read_csv(f'{ADIR}/skill_t2m.csv'); frames.append(t2[t2.variable == 'T2M'])
except FileNotFoundError:
    pass
df = pd.concat(frames, ignore_index=True)
df['wk'] = df['week'].str.extract(r'(\d)').astype(int)
df['crmse'] = np.sqrt(np.clip(df['rmse'] ** 2 - df['bias'] ** 2, 0, None))

MODELS = ['SPIRE', 'FuXi', 'ECMWF', 'NCEP']
REG = ['northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
REGL = {'northwest_india': 'Northwest', 'central_india': 'Central', 'south_peninsula': 'S. Peninsula', 'east_northeast_india': 'East/NE'}
VARL = {'TP': 'Precipitation (mm\\,day$^{-1}$)', 'Z500': 'Z500 (m)', 'T2M': 'T2M (K)'}
FMT = {'TP': '{:.2f}', 'Z500': '{:.1f}', 'T2M': '{:.2f}'}


def regional_table(metric, lower_better):
    lines = []
    for var in [v for v in ['TP', 'Z500', 'T2M'] if v in df.variable.unique()]:
        # use bias-corrected RMSE for T2M (sub-daily cold-bias; see text)
        col = 'crmse' if (metric == 'rmse' and var == 'T2M') else metric
        fmt = '{:.2f}' if metric == 'pcc' else FMT[var]
        lines.append('\\midrule\n\\multicolumn{5}{l}{\\emph{' + VARL[var] + '}}\\\\')
        for rg in REG:
            vals = {m: df[(df.variable == var) & (df.region == rg) & (df.model == m) & (df.wk <= 4)][col].mean() for m in MODELS}
            best = (min if lower_better else max)(vals, key=vals.get)
            cells = [(f'\\textbf{{{fmt.format(vals[m])}}}' if m == best else fmt.format(vals[m])) for m in MODELS]
            lines.append(f'{REGL[rg]:<13} & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)


hdr = 'Region & SPIRE & FuXi-S2S & ECMWF & NCEP \\\\'
rmse_tab = ('\\begin{table}[t]\\centering\n'
            '\\caption{Region-wise RMSE by IMD homogeneous region (weeks 1--4 mean, 13-init average). '
            'Best system per region in bold. Temperature uses the bias-corrected (centered) RMSE '
            '(Section~\\ref{sec:t2m_skill}).}\\label{tab:reg_rmse}\n'
            '\\begin{tabular}{lcccc}\\toprule\n' + hdr + '\n' + regional_table('rmse', True) + '\n\\bottomrule\\end{tabular}\\end{table}')
pcc_tab = ('\\begin{table}[t]\\centering\n'
           '\\caption{Region-wise pattern correlation (PCC) by IMD homogeneous region (weeks 1--4 mean, 13-init average). '
           'Best system per region in bold.}\\label{tab:reg_pcc}\n'
           '\\begin{tabular}{lcccc}\\toprule\n' + hdr + '\n' + regional_table('pcc', False) + '\n\\bottomrule\\end{tabular}\\end{table}')

with open(f'{ADIR}/tables_regional.tex', 'w') as fh:
    fh.write(rmse_tab + '\n\n' + pcc_tab + '\n')

print("===== REGION-WISE RMSE (weeks 1-4 mean) =====")
for var in [v for v in ['TP', 'Z500', 'T2M'] if v in df.variable.unique()]:
    print(f"\n {var}:")
    print(f"   {'Region':<14}" + "".join(f"{m:>9}" for m in MODELS))
    for rg in REG:
        print(f"   {REGL[rg]:<14}" + "".join(f"{df[(df.variable==var)&(df.region==rg)&(df.model==m)&(df.wk<=4)]['rmse'].mean():9.2f}" for m in MODELS))

print("\n===== LEAD-TIME ERROR, All India (RMSE | PCC) per week =====")
for var in [v for v in ['TP', 'Z500', 'T2M'] if v in df.variable.unique()]:
    print(f"\n {var}:")
    for m in MODELS:
        rm = [df[(df.variable == var) & (df.region == 'All India') & (df.model == m) & (df.wk == w)]['rmse'].mean() for w in range(1, 7)]
        pc = [df[(df.variable == var) & (df.region == 'All India') & (df.model == m) & (df.wk == w)]['pcc'].mean() for w in range(1, 7)]
        print(f"   {m:<6} RMSE " + " ".join(f"{x:6.2f}" for x in rm))
        print(f"   {'':<6} PCC  " + " ".join(f"{x:6.2f}" for x in pc))
print("\nWROTE tables_regional.tex")
print("DONE")

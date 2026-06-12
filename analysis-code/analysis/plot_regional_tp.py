import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the WMO regional TP data
df = pd.read_csv('/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis/tp_regional_wmo.csv')

# Filter for Week 1 and Week 2 (the most critical predictive window for precipitation)
df_sub = df[df['Week'].isin(['Week 1', 'Week 2'])]

# Aggregate across the initializations and weeks to get the mean score per region per model
agg_df = df_sub.groupby(['Region', 'Model'])[['ACC', 'RMSE', 'Bias']].mean().reset_index()

regions = ['northwest_india', 'central_india', 'south_peninsula', 'east_northeast_india']
region_labels = ['NW India', 'Central India', 'South Peninsula', 'NE India']
models = ['FuXi', 'Spire', 'ECMWF', 'NCEP']
colors = {'FuXi': '#0072B2', 'Spire': '#D55E00', 'ECMWF': '#009E73', 'NCEP': '#CC79A7'}

x = np.arange(len(regions))
width = 0.2

fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)

for i, metric in enumerate(['ACC', 'RMSE', 'Bias']):
    ax = axes[i]
    for j, model in enumerate(models):
        y = []
        for r in regions:
            val = agg_df[(agg_df['Region'] == r) & (agg_df['Model'] == model)][metric]
            y.append(val.values[0] if not val.empty else 0)
        
        ax.bar(x + (j - 1.5) * width, y, width, label=model, color=colors[model], edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, fontweight='bold')
    ax.set_title(f'Regional TP {metric} (Week 1-2 Avg)', fontsize=14, fontweight='bold')
    
    if metric == 'ACC':
        ax.set_ylabel('Pattern Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.axhline(0.6, color='black', linestyle='--')
        ax.set_ylim(0, 1.0)
    elif metric == 'RMSE':
        ax.set_ylabel('RMSE (mm/day)', fontsize=12, fontweight='bold')
    elif metric == 'Bias':
        ax.set_ylabel('Mean Bias (mm/day)', fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        
    if i == 0:
        ax.legend(loc='best', fontsize=11)

plt.tight_layout()
out_path = '/home/raj.ayush/s2s/s2s_anlysis/analysis-code/figures/verification/tp_regional_wmo_metrics.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"SUCCESS! Regional TP Metrics saved to {out_path}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

os.makedirs('/home/raj.ayush/s2s/s2s_anlysis/analysis-code/figures/verification', exist_ok=True)

def plot_scorecard(csv_path, var_name):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}. Please run the compute script first.")
        return

    df = pd.read_csv(csv_path)
    # Filter out All India, keep only sub-regions
    df = df[df['Region'] != 'All India']
    
    # Format region names for display
    df['Region'] = df['Region'].str.replace('_', ' ').str.title()
    
    metrics = ['ACC', 'RMSE', 'CRPS']
    models = ['FuXi', 'Spire', 'ECMWF', 'NCEP']
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True, dpi=300)
        fig.suptitle(f'{var_name} - {metric} Scorecard', fontsize=16, fontweight='bold', y=1.05)
        
        for i, model in enumerate(models):
            ax = axes[i]
            model_df = df[df['Model'] == model].copy()
            if model_df.empty:
                continue
                
            # Pivot table to make matrix: Index=Region, Columns=Week, Values=Metric
            pivot_df = model_df.pivot(index='Region', columns='Week', values=metric)
            
            # Determine colormap and limits based on metric
            if metric == 'ACC':
                cmap = 'RdYlGn'
                vmin, vmax = 0, 1
                fmt = ".2f"
            elif metric == 'RMSE':
                cmap = 'YlOrRd'
                vmin = df[metric].min()
                vmax = df[metric].max()
                fmt = ".1f" if var_name == 'TP' else ".0f"
            elif metric == 'CRPS':
                cmap = 'YlGnBu'
                vmin = df[metric].min()
                vmax = df[metric].max()
                fmt = ".2f" if var_name == 'TP' else ".1f"
                
            sns.heatmap(pivot_df, ax=ax, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax,
                        linewidths=.5, cbar=(i == 3), cbar_kws={'label': metric})
            
            ax.set_title(f'{model}', fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            
        plt.tight_layout()
        out_path = f'/home/raj.ayush/s2s/s2s_anlysis/analysis-code/figures/verification/scorecard_{var_name.lower()}_{metric.lower()}.png'
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved scorecard to {out_path}")

print("Generating Scorecards...")
plot_scorecard('/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis/tp_regional_wmo_with_crps.csv', 'TP')
plot_scorecard('/home/raj.ayush/s2s/s2s_anlysis/analysis-code/analysis/z500_regional_wmo_with_crps.csv', 'Z500')
print("Done!")

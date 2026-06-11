import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Metrics
df = pd.read_csv('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis/regional_metrics_week2.csv')

regions = df['IMD Region'].tolist()
fuxi_rmse = df['FuXi RMSE (°C)'].tolist()
spire_rmse = df['Spire RMSE (°C)'].tolist()
fuxi_bias = df['FuXi Mean Bias (°C)'].tolist()
spire_bias = df['Spire Mean Bias (°C)'].tolist()

x = np.arange(len(regions))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# RMSE Plot
rects1 = ax1.bar(x - width/2, fuxi_rmse, width, label='FuXi S2S', color='#1f77b4', edgecolor='black', zorder=3)
rects2 = ax1.bar(x + width/2, spire_rmse, width, label='Spire S2S', color='#ff7f0e', edgecolor='black', zorder=3)

ax1.set_ylabel('RMSE (°C)', fontsize=12, fontweight='bold')
ax1.set_title('Week 2 T2M RMSE by IMD Region', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(regions, fontsize=10, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)

# Bias Plot
rects3 = ax2.bar(x - width/2, fuxi_bias, width, label='FuXi S2S', color='#1f77b4', edgecolor='black', zorder=3)
rects4 = ax2.bar(x + width/2, spire_bias, width, label='Spire S2S', color='#ff7f0e', edgecolor='black', zorder=3)

ax2.set_ylabel('Mean Bias (°C)', fontsize=12, fontweight='bold')
ax2.set_title('Week 2 T2M Mean Bias by IMD Region', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(regions, fontsize=10, fontweight='bold')
ax2.axhline(0, color='black', linewidth=1.5, zorder=3)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6, zorder=0)

plt.tight_layout()
out_path = '/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/figures/verification/imd_regional_barchart.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"SUCCESS! Bar chart saved to {out_path}")

"""
13_main_figure.py — Consolidated paper main figure (Figure 1).

Reads weekly_anomalies_v2.nc (corrected, consistent-baseline) and builds a single
publication panel combining the headline result:
  Row 1: Spire forecast max-T2m anomaly, W1 / W3 / W6
  Row 2: ERA5 observed   max-T2m anomaly, W1 / W3 / W6
  Row 3: (g) India-mean ACC vs lead (all 4 fields); (h) RMSE vs lead;
         (i) grid-cell scatter Spire vs ERA5 (W1, max-T2m)

Output: figures/paper/Fig1_main.png  (+ Fig1_main.pdf)
"""
import os, string, warnings
import numpy as np, pandas as pd, xarray as xr
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs, cartopy.feature as cfeature
warnings.filterwarnings("ignore")

DATA = "weekly_anomalies_v2.nc"
OUT  = "figures/paper"; os.makedirs(OUT, exist_ok=True)
PROJ = ccrs.PlateCarree(); EXT = [55, 105, 0, 50]
C_SPIRE, C_ERA5, C_PREC, C_Z500, C_TMAX = "#D73027","#2166AC","#1a9850","#762a83","#F46D43"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.facecolor":"#FAFAFA"})

ds = xr.open_dataset(DATA)
lats, lons = ds.latitude.values, ds.longitude.values
it = pd.DatetimeIndex(ds.init_time.values); N = len(it)
RNG = f"{it[0]:%Y-%m-%d} to {it[-1]:%Y-%m-%d}"
let = list(string.ascii_lowercase)

SP_MAX = ds["spire_t2m_max_anom"].mean("init_time").values   # (week,lat,lon)
E5_MAX = ds["era5_t2m_max_anom"].mean("init_time").values

def imean(v): return np.nanmean(ds[v].values, axis=(-1,-2))   # (init,week)
def acc_rmse(spv, e5v):
    sp, e5 = ds[spv].values, ds[e5v].values
    nW = sp.shape[1]; acc=np.full(nW,np.nan); rmse=np.full(nW,np.nan)
    for w in range(nW):
        a=sp[:,w].reshape(N,-1); b=e5[:,w].reshape(N,-1)
        rs=[pearsonr(a[:,p],b[:,p])[0] for p in range(a.shape[1])
            if np.std(a[:,p])>1e-6 and np.std(b[:,p])>1e-6]
        acc[w]=np.nanmean(rs) if rs else np.nan
        rmse[w]=float(np.sqrt(np.nanmean((a-b)**2)))
    return acc, rmse

ACC_MEAN,RMSE_MEAN = acc_rmse("spire_t2m_mean_anom","era5_t2m_mean_anom")
ACC_MAX, RMSE_MAX  = acc_rmse("spire_t2m_max_anom", "era5_t2m_max_anom")
ACC_PREC,RMSE_PREC = acc_rmse("spire_precip_anom",  "era5_precip_anom")
ACC_Z500,RMSE_Z500 = acc_rmse("spire_z500_anom",    "era5_z500_anom")
x = np.arange(1,7)

def base_map(ax, data, norm, title="", letter=None, corner=None, ll=False, bl=False):
    im = ax.pcolormesh(lons,lats,data,cmap=plt.cm.RdBu_r,norm=norm,transform=PROJ,
                       rasterized=True,shading="auto")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),lw=0.7,edgecolor="k",zorder=4)
    ax.set_extent(EXT,crs=PROJ)
    gl=ax.gridlines(lw=0.2,color="gray",ls=":",alpha=0.5)
    gl.xlocator=mticker.FixedLocator([60,70,80,90,100]); gl.ylocator=mticker.FixedLocator([10,20,30,40])
    gl.top_labels=gl.right_labels=False; gl.left_labels=ll; gl.bottom_labels=bl
    gl.xlabel_style=gl.ylabel_style={"size":7,"color":"0.35"}
    if title: ax.set_title(title,fontsize=10,fontweight="bold",pad=3)
    if letter: ax.text(0.03,0.95,letter,transform=ax.transAxes,fontsize=10,fontweight="bold",
                       va="top",bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="0.5",alpha=0.9))
    if corner: ax.text(0.97,0.05,corner,transform=ax.transAxes,fontsize=8,ha="right",va="bottom",
                       bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="0.5",alpha=0.9))
    return im

fig = plt.figure(figsize=(15,15))
gs = gridspec.GridSpec(3,3,figure=fig,hspace=0.18,wspace=0.10,
                       left=0.07,right=0.92,bottom=0.05,top=0.93,
                       height_ratios=[1,1,0.95])
vmax=6; norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
cols=[0,2,5]; clab=["W1 (d1–7)","W3 (d15–21)","W6 (d36–42)"]

# rows 1-2 : maps
for j,(wi,cl) in enumerate(zip(cols,clab)):
    ax=fig.add_subplot(gs[0,j],projection=PROJ)
    base_map(ax,SP_MAX[wi],norm,title=cl,letter=f"({let[j]})",
             corner=f"μ={np.nanmean(SP_MAX[wi]):+.2f}",ll=(j==0))
    ax2=fig.add_subplot(gs[1,j],projection=PROJ)
    im=base_map(ax2,E5_MAX[wi],norm,letter=f"({let[j+3]})",
                corner=f"μ={np.nanmean(E5_MAX[wi]):+.2f}",ll=(j==0),bl=True)
fig.text(0.045,0.80,"Spire forecast\n(max T2m)",ha="center",va="center",rotation=90,
         fontsize=11,fontweight="bold")
fig.text(0.045,0.56,"ERA5 observed\n(max T2m)",ha="center",va="center",rotation=90,
         fontsize=11,fontweight="bold")
cax=fig.add_axes([0.93,0.43,0.013,0.49])
cb=fig.colorbar(im,cax=cax,extend="both",ticks=[-6,-4,-2,0,2,4,6])
cb.set_label("Max-T2m anomaly (K)",fontsize=10)

# row 3
WL=["W1","W2","W3","W4","W5","W6"]
ax_acc=fig.add_subplot(gs[2,0]); ax_rmse=fig.add_subplot(gs[2,1]); ax_sc=fig.add_subplot(gs[2,2])

for a,c,m,l in [(ACC_MEAN,C_SPIRE,"o","T2m-mean"),(ACC_MAX,C_TMAX,"s","T2m-max"),
                (ACC_PREC,C_PREC,"^","Precip"),(ACC_Z500,C_Z500,"D","Z500")]:
    ax_acc.plot(x,a,m+"-",color=c,lw=2,ms=7,label=l,mec="white",mew=0.5)
ax_acc.axhline(0.5,color="gray",ls="-.",lw=1,alpha=0.7); ax_acc.axhline(0,color="k",ls="--",lw=0.8,alpha=0.5)
ax_acc.fill_between([0.5,6.5],0.5,1,color="#4caf50",alpha=0.07)
ax_acc.set_xlim(0.5,6.5); ax_acc.set_ylim(-0.45,1); ax_acc.set_xticks(x); ax_acc.set_xticklabels(WL)
ax_acc.set_ylabel("India-mean ACC (r)",fontsize=10); ax_acc.set_xlabel("Lead week",fontsize=10)
ax_acc.set_title(f"({let[6]}) Anomaly correlation vs lead",fontsize=10.5,fontweight="bold")
ax_acc.legend(fontsize=8,loc="upper right"); ax_acc.grid(True,alpha=0.3)

ax_rmse.plot(x,RMSE_MEAN,"o-",color=C_SPIRE,lw=2,ms=7,label="T2m-mean",mec="white",mew=0.5)
ax_rmse.plot(x,RMSE_MAX,"s-",color=C_TMAX,lw=2,ms=7,label="T2m-max",mec="white",mew=0.5)
ax_rmse.set_xticks(x); ax_rmse.set_xticklabels(WL); ax_rmse.set_xlabel("Lead week",fontsize=10)
ax_rmse.set_ylabel("RMSE (K)",fontsize=10)
ax_rmse.set_title(f"({let[7]}) Temperature RMSE vs lead",fontsize=10.5,fontweight="bold")
ax_rmse.legend(fontsize=9); ax_rmse.grid(True,alpha=0.3)

spw=SP_MAX[0].ravel(); e5w=E5_MAX[0].ravel()
mk=np.isfinite(spw)&np.isfinite(e5w); spw,e5w=spw[mk],e5w[mk]
r,_=pearsonr(spw,e5w); rmse=np.sqrt(np.mean((spw-e5w)**2)); bias=np.mean(spw-e5w)
lo,hi=-8,8
ax_sc.scatter(e5w,spw,s=6,alpha=0.3,color=C_TMAX,rasterized=True,linewidths=0)
ax_sc.plot([lo,hi],[lo,hi],"k--",lw=1.2,alpha=0.6)
coef=np.polyfit(e5w,spw,1); xf=np.linspace(lo,hi,100)
ax_sc.plot(xf,np.polyval(coef,xf),"-",color=C_SPIRE,lw=1.8)
ax_sc.set_xlim(lo,hi); ax_sc.set_ylim(lo,hi); ax_sc.set_aspect("equal","box")
ax_sc.set_xlabel("ERA5 (K)",fontsize=10); ax_sc.set_ylabel("Spire (K)",fontsize=10)
ax_sc.set_title(f"({let[8]}) W1 grid-cell scatter (max T2m)",fontsize=10.5,fontweight="bold")
ax_sc.text(0.04,0.96,f"r={r:+.3f}  R²={r**2:.3f}\nRMSE={rmse:.2f} K\nbias={bias:+.2f} K\nk={coef[0]:.2f}",
           transform=ax_sc.transAxes,fontsize=8.5,va="top",family="monospace",
           bbox=dict(boxstyle="round,pad=0.3",fc="#FFFDE7",ec="0.55",alpha=0.95))
ax_sc.grid(True,alpha=0.3)

fig.suptitle(f"Spire JFM 2026 S2S hindcast vs ERA5  |  India  |  daily-max 2 m T  |  90 inits ({RNG})",
             fontsize=13,fontweight="bold",y=0.965)
fig.savefig(f"{OUT}/Fig1_main.png",dpi=300,bbox_inches="tight",facecolor="white")
fig.savefig(f"{OUT}/Fig1_main.pdf",bbox_inches="tight",facecolor="white")
plt.close(fig)
print(f"→ {OUT}/Fig1_main.png + .pdf")

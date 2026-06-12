"""
T2M (2-m temperature) verification as a third variable, JFM-2026 SPIRE benchmark.
Truth: ERA5 daily-mean t2m (ARCO; era5_daily_t2m.nc).
Models (all daily-mean): SPIRE air_temperature; FuXi 't2m' channel (11-mem mean);
ECMWF/NCEP (mx2t6+mn2t6)/2 PF-mean as a daily-mean proxy.

Output: analysis/skill_t2m.csv  (+ appends nothing else)
"""
import os, sys, warnings
import numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings('ignore')
sys.path.append('/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/manuscript_code')
from utils.verification_wmo import get_cosine_latitude_weights, calc_wmo_acc, calc_wmo_rmse, calc_wmo_bias
from utils.verification_extra import get_land_mask, mask_land

DATA='/storage/raj.ayush/s2s-forecast-data'; ADIR='/home/raj.ayush/s2s/s2s_anlysis/paper_results_pipeline/analysis'
OPEN=dict(engine='cfgrib', backend_kwargs={'indexpath':''})
init_dates=['2026-01-01','2026-01-08','2026-01-15','2026-01-22','2026-01-29','2026-02-05','2026-02-12','2026-02-19','2026-02-26','2026-03-05','2026-03-12','2026-03-19','2026-03-26']
weeks=[('Week 1',1,7),('Week 2',8,14),('Week 3',15,21),('Week 4',22,28),('Week 5',29,35),('Week 6',36,42)]
target_lat=np.arange(38,5,-1.5); target_lon=np.arange(65,100,1.5)
LAND=get_land_mask(target_lat,target_lon)
REGION_BOUNDS={'All India':(5.,38.,65.,100.),'northwest_india':(22.,38.,68.,82.),'central_india':(18.,28.,72.,89.),'south_peninsula':(8.,20.,72.,85.),'east_northeast_india':(20.,30.,85.,98.)}
MODELS=['SPIRE','FuXi','ECMWF','NCEP']

def to_grid(da):
    ren={}
    if 'latitude' in da.dims: ren['latitude']='lat'
    if 'longitude' in da.dims: ren['longitude']='lon'
    if ren: da=da.rename(ren)
    return mask_land(da.interp(lat=target_lat,lon=target_lon,method='linear').squeeze(),LAND)

daily=xr.open_dataset(f'{ADIR}/era5_daily_t2m.nc')['t2m']
clim=to_grid(daily.mean('time'))
def era_week(valid):
    try: return to_grid(daily.sel(time=slice(valid[0],valid[-1])).mean('time'))
    except Exception: return None

def fuxi_day(init_str,day):
    fs=[]
    for mem in range(11):
        p=f"{DATA}/fuxi/output/{init_str}/member/{mem:02d}/{day:02d}.nc"
        if not os.path.exists(p): continue
        da=xr.open_dataset(p)['__xarray_dataarray_variable__'].sel(channel='t2m')
        for d in list(da.dims):
            if d not in ('lat','lon','latitude','longitude'): da=da.mean(d)
        fs.append(da)
    return None if not fs else xr.concat(fs,'m').mean('m')

def load_op_t2m(model,init_str):
    base=f'{DATA}/{model}/data'
    try:
        mx=xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib',filter_by_keys={'shortName':'mx2t6'},**OPEN)['mx2t6']
        mn=xr.open_dataset(f'{base}/sfc_pf_{init_str}.grib',filter_by_keys={'shortName':'mn2t6'},**OPEN)['mn2t6']
        t=((mx+mn)/2.0)
        return t.mean('number') if 'number' in t.dims else t
    except Exception as e:
        print(f"  {model} t2m fail {init_str}: {e}",flush=True); return None

def regional(f,rg):
    a,b,c,d=REGION_BOUNDS[rg]; return f.sel(lat=slice(b,a),lon=slice(c,d))

rows=[]
for init in init_dates:
    init_str=pd.to_datetime(init).strftime('%Y%m%d'); print(f"=== INIT {init} ===",flush=True)
    try:
        sp=xr.open_zarr(f"{DATA}/spire/spire_hindcast_jfm.zarr",group='mean_stddev').sel(reference_time=init)['air_temperature']
    except Exception as e: sp=None; print("  SPIRE fail",e,flush=True)
    ec=load_op_t2m('ecmwf',init_str); nc=load_op_t2m('ncep',init_str)
    fx={d:fuxi_day(init_str,d) for d in range(1,43)}
    for wn,ds_,de in weeks:
        dts=pd.date_range(start=init,periods=42)[ds_-1:de]
        valid=[d.strftime('%Y-%m-%d') for d in dts if d.strftime('%Y-%m-%d')<='2026-05-10']
        if not valid: continue
        o=era_week(valid)
        if o is None or np.isnan(o).all(): continue
        f={}
        if sp is not None: f['SPIRE']=to_grid(sp.isel(step=slice(ds_-1,de)).mean('step'))
        fd=[fx[d] for d in range(ds_,de+1) if fx.get(d) is not None]
        if fd: f['FuXi']=to_grid(xr.concat(fd,'t').mean('t'))
        for nm,arr in [('ECMWF',ec),('NCEP',nc)]:
            if arr is not None and arr.sizes.get('step',0)>=de:
                f[nm]=to_grid(arr.isel(step=slice(ds_-1,de)).mean('step'))
        for rg in REGION_BOUNDS:
            for m in MODELS:
                if m not in f: continue
                fr,orr,cr=regional(f[m],rg),regional(o,rg),regional(clim,rg)
                w=get_cosine_latitude_weights(fr.lat.values)
                try:
                    rows.append(dict(variable='T2M',region=rg,week=wn,init_date=init,model=m,
                        pcc=calc_wmo_acc(fr,orr,cr,w),rmse=calc_wmo_rmse(fr,orr,w),bias=calc_wmo_bias(fr,orr,w),
                        fcst_mean=float(fr.weighted(w).mean(['lat','lon'])),obs_mean=float(orr.weighted(w).mean(['lat','lon']))))
                except Exception as e: print("  metric fail",m,rg,wn,e,flush=True)

pd.DataFrame(rows).to_csv(f'{ADIR}/skill_t2m.csv',index=False)
df=pd.DataFrame(rows); df['wk']=df['week'].str.extract(r'(\d)').astype(int)
print(f"\nWROTE skill_t2m.csv ({len(df)} rows)",flush=True)
print("### T2M PCC (All India) ###")
for m in MODELS: print(f"{m:<6}"+"".join(f"  {df[(df.region=='All India')&(df.model==m)&(df.wk==w)]['pcc'].mean():.2f}" for w in range(1,7)))
print("### T2M RMSE K (All India) ###")
for m in MODELS: print(f"{m:<6}"+"".join(f"  {df[(df.region=='All India')&(df.model==m)&(df.wk==w)]['rmse'].mean():.2f}" for w in range(1,7)))
print("COMPUTE_DONE",flush=True)

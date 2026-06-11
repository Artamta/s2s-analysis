#!/usr/bin/env bash
# fix_fcn3run_gpu1.sh  — run this ONCE directly on gpu1
# Fixes zarr version, patches earth2studio, removes physicsnemo
set -euo pipefail

ENV_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "[1] site-packages: ${ENV_SITE}"

# ── 1. Upgrade zarr to 3.1.x ─────────────────────────────────────────────
pip install "zarr==3.1.6" -q
echo "[2] zarr 3.1.6 installed"

# ── 2. Patch async_zarr.py (AsyncGroup moved in zarr 3.1.x) ─────────────
python - <<'PYEOF'
import pathlib, site

sp = pathlib.Path(site.getsitepackages()[0])
f  = sp / "earth2studio/io/async_zarr.py"
txt = f.read_text()

old = "from zarr import AsyncGroup"
new = ("try:\n"
       "    from zarr import AsyncGroup\n"
       "except ImportError:\n"
       "    from zarr.core.group import AsyncGroup")

if old in txt and new not in txt:
    f.write_text(txt.replace(old, new))
    print(f"[3] patched {f}")
else:
    print(f"[3] async_zarr.py already patched or pattern not found")

# clear pyc
for p in (sp / "earth2studio/io/__pycache__").glob("async_zarr*"):
    p.unlink()
PYEOF

# ── 3. Remove physicsnemo (incompatible with torch 2.4.x) ────────────────
python - <<'PYEOF'
import shutil, pathlib, site

sp   = pathlib.Path(site.getsitepackages()[0])
pkg  = sp / "physicsnemo"
dist = sp / "nvidia_physicsnemo-2.1.1.dist-info"

for p in [pkg, dist]:
    if p.exists():
        shutil.rmtree(p)
        print(f"[4] removed {p}")
    else:
        print(f"[4] {p.name} not present")
PYEOF

# ── 4. Patch dx/__init__.py (wrap all physicsnemo-dependent imports) ──────
python - <<'PYEOF'
import pathlib, site

sp = pathlib.Path(site.getsitepackages()[0])
f  = sp / "earth2studio/models/dx/__init__.py"

content = """\
# patched: wrap all optional model imports so broken deps don't block FCN3
from earth2studio.models.dx.base import DiagnosticModel

def _try(fn):
    try: return fn()
    except Exception: return None

CBottleInfill     = _try(lambda: __import__('earth2studio.models.dx.cbottle_infill',   fromlist=['CBottleInfill']).CBottleInfill)
CBottleSR         = _try(lambda: __import__('earth2studio.models.dx.cbottle_sr',        fromlist=['CBottleSR']).CBottleSR)
CBottleTCGuidance = _try(lambda: __import__('earth2studio.models.dx.cbottle_tc',        fromlist=['CBottleTCGuidance']).CBottleTCGuidance)
ClimateNet        = _try(lambda: __import__('earth2studio.models.dx.climatenet',         fromlist=['ClimateNet']).ClimateNet)
CorrDiff          = _try(lambda: __import__('earth2studio.models.dx.corrdiff',           fromlist=['CorrDiff']).CorrDiff)
CorrDiffTaiwan    = _try(lambda: __import__('earth2studio.models.dx.corrdiff',           fromlist=['CorrDiffTaiwan']).CorrDiffTaiwan)
CorrDiffCMIP6     = _try(lambda: __import__('earth2studio.models.dx.corrdiff_cmip6',    fromlist=['CorrDiffCMIP6']).CorrDiffCMIP6)
PrecipitationAFNO = _try(lambda: __import__('earth2studio.models.dx.precipitation_afno',fromlist=['PrecipitationAFNO']).PrecipitationAFNO)
PrecipitationAFNOv2=_try(lambda: __import__('earth2studio.models.dx.precipitation_afno_v2',fromlist=['PrecipitationAFNOv2']).PrecipitationAFNOv2)
Identity          = _try(lambda: __import__('earth2studio.models.dx.identity',           fromlist=['Identity']).Identity)
OrbitGlobalPrecip = _try(lambda: __import__('earth2studio.models.dx.orbit2_precip',      fromlist=['OrbitGlobalPrecip']).OrbitGlobalPrecip)
WindgustAFNO      = _try(lambda: __import__('earth2studio.models.dx.wind_gust',          fromlist=['WindgustAFNO']).WindgustAFNO)

try:
    from earth2studio.models.dx.derived import DerivedRH,DerivedRHDewpoint,DerivedSurfacePressure,DerivedTCWV,DerivedVPD,DerivedWS
except Exception:
    DerivedRH=DerivedRHDewpoint=DerivedSurfacePressure=DerivedTCWV=DerivedVPD=DerivedWS=None
try:
    from earth2studio.models.dx.tc_tracking import TCTrackerVitart,TCTrackerWuDuan
except Exception:
    TCTrackerVitart=TCTrackerWuDuan=None
try:
    from earth2studio.models.dx.solarradiation_afno import SolarRadiationAFNO1H,SolarRadiationAFNO6H
except Exception:
    SolarRadiationAFNO1H=SolarRadiationAFNO6H=None
"""

f.write_text(content)
for p in (sp / "earth2studio/models/dx/__pycache__").glob("*"):
    p.unlink(missing_ok=True)
for p in (sp / "earth2studio/models/px/__pycache__").glob("*"):
    p.unlink(missing_ok=True)
print("[5] dx/__init__.py patched and caches cleared")
PYEOF

# ── 5. Final import check ─────────────────────────────────────────────────
echo "[6] Running import check..."
python -c "
from earth2studio.models.px import FCN3
from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic as e2s_run
import torch
print()
print('  ✓ ALL IMPORTS OK')
print(f'  torch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
" 2>/dev/null

echo ""
echo "Done. Now run: bash launch_fcn3_weekly.sh"

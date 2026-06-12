# ECMWF S2S Benchmark — Setup & Plan

Goal: benchmark **Spire AI-S2S** against the **ECMWF operational extended-range (S2S)**
model over the *same* period (JFM 2026) — a clean, referee-credible head-to-head.

## 1. Get access (free, ~10 min, you must do this — I can't register for you)
1. Create an ECMWF account: https://www.ecmwf.int/ (Login → register).
2. Accept the **S2S data licence**: https://apps.ecmwf.int/datasets/data/s2s/ (click any
   field, agree to the terms once).
3. Get your API key: https://api.ecmwf.int/v1/key/ — copy the shown block into
   `~/.ecmwfapirc`:
   ```
   {
       "url"   : "https://api.ecmwf.int/v1",
       "key"   : "<your-key>",
       "email" : "<your-email>"
   }
   ```
4. Client is already installed (`ecmwf-api-client`).

## 2. Download (script ready)
```bash
cd spire-s2s-paper
conda run -n s2s-hind python -u scripts/download_ecmwf_s2s.py
```
- Pulls ECMWF extended-range for **Mon/Thu init dates in JFM 2026** (≈26 dates that
  pair with Spire's daily inits), control + 50 perturbed members, 46-day lead.
- Variables: 2t, daily **max/min** 2t (so we match Spire Tmean/Tmax without a
  diurnal-sampling bias), Z500, total precip. Native ~1.5°, India domain.
- Output: `data/ecmwf_s2s/<param>_<cf|pf>_<YYYYMMDD>.grib`
- **Expect a queue:** ECMWF MARS is a tape archive — requests may sit in queue for
  minutes–hours each. Run it in the background / overnight. (~26 dates × 5 params × 2
  types ≈ 260 requests; consider trimming PARAMS to 2t + mx2t24 first.)

## 3. Verification (I will build `verify_ecmwf_s2s.py` once data is down)
It will, on the **paired init dates only**:
- Regrid ECMWF 1.5° → Spire 0.5° (or both → a common grid).
- Compute weekly means W1–W6 for ensemble mean (deterministic) and members (probabilistic).
- Re-use the exact Spire metrics so it's apples-to-apples:
  - **Deterministic:** ACC, RMSE, bias, scatter vs ERA5 → overlay ECMWF vs Spire.
  - **Probabilistic:** tercile RPSS, reliability → ECMWF vs Spire.
- Output: `fig27_spire_vs_ecmwf_acc.png`, `fig28_spire_vs_ecmwf_rpss.png`, etc.

## Caveats to state in the paper
- **Cadence:** ECMWF S2S archive is twice-weekly; comparison is on the ~26 paired
  dates, not all 90 Spire inits. Report n.
- **Resolution:** ECMWF ~1.5° vs Spire 0.5°; both verified on a common grid. Coarser
  model is mildly favoured by smoothing — disclose.
- **Members:** ECMWF 51 vs Spire ensemble — note ensemble sizes when comparing spread.
- Same period (JFM 2026), same truth (ERA5) → the comparison itself is clean.

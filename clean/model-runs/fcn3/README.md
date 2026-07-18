# FCN3 native-T2M inference

The paper production experiment retains only FourCastNet 3's native `t2m`
channel. It uses the frozen 517-case 2020-2024 calendar, exact ERA5 00 UTC
initial states, three fixed native stochastic samples, 42 complete forecast days,
and exact node selection onto the common 1.5 degree 27 x 27 India grid.

The production contract is
`model-runs/configs/fcn3_t2m_2020_2024.json`. Outputs are written to:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/model-runs/fcn3/
  fcn3_v1_t2m_00z_2020_2024_ens3/
    forecasts/YYYY/YYYYMMDD.nc
    manifests/YYYY/YYYYMMDD.json
```

Each NetCDF contains `t2m(member, lead_day, latitude, longitude)` in `degC`.
Daily means use trapezoidal integration of the instantaneous 0, 6, 12, 18,
and 24 UTC boundaries. FCN3 still evolves its full global 72-channel state;
only native T2M is retained.

## Validation and production

Run the two-day/two-member smoke test first, then the full 42-day/three-member
acceptance case. Both use `GPU-AI_prio`:

```bash
sbatch model-runs/fcn3/slurm/pilot_fcn3_t2m_20200601.sbatch
sbatch model-runs/fcn3/slurm/pilot42d_ens3_fcn3_t2m_20200601.sbatch
sbatch model-runs/fcn3/slurm/plot_fcn3_t2m_pilot.sbatch
```

After the acceptance case and ERA5 plot pass, production uses `GPU-AI` with at
most four concurrent A100 jobs:

```bash
sbatch model-runs/fcn3/slurm/run_fcn3_t2m_2020_2024.sbatch
```

The runner is resumable: a forecast is skipped only when both its manifest and
NetCDF exist and the recorded SHA256 matches. Other partial or invalid states
stop for inspection instead of being overwritten.

## Why AFNOv2 TP is excluded

FCN3 has no native precipitation channel. The earlier experiment coupled FCN3
to NVIDIA's separate AFNOv2 precipitation diagnostic and is preserved under
`fcn3_v1_afnov2_tp_t2m_00z_2020_2024_ens10` for provenance.

The three-member ensemble mean is the primary FCN3 field for ACC and RMSE; the
individual members remain in each file. A single-member sensitivity can also be
reported from member 0 without another inference run.

AFNOv2 was also driven directly by exact ERA5 atmospheric states with native
ERA5 surface pressure for one day in each season. On the common India grid,
predicted/ERA5 mean precipitation ratios were 0.493 (JFM), 0.441 (MAM), 0.483
(JJAS), and 0.429 (OND). Spatial correlations were 0.928-0.962. Therefore its
spatial pattern is useful diagnostically, but raw rainfall amount is
systematically about 51-57% too low in these checks. This is intrinsic to the
diagnostic in this benchmark setup, rather than an FCN3, unit-conversion,
surface-pressure, or remapping error.

No multiplicative bias correction is used. The raw seasonal isolation NetCDFs,
manifests, and the completed two-member 42-day pilot remain in storage as an
auditable sensitivity experiment, not as primary FCN3 precipitation results.

The supplied `rat_test.py` remains unchanged as the original working reference;
it is not a production runner.

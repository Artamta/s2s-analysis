# Benchmark Model Selection

Status date: `2026-07-13`. Availability was checked against the live ECDS
`s2s-forecasts` and `s2s-reforecasts` constraints.

## Recommended Physics Benchmark

| role | model | reason |
|---|---|---|
| primary | ECMWF | leading operational reference, large ensemble, 42+ days |
| primary | UKMO | independent coupled system, daily starts, 60 days |
| primary | NCEP CFSv2 | long-running daily baseline, 16 members, 44 days |
| primary | CMA | complete 2020-2024 coverage, Mon/Thu cadence with the same 104-start/year density as FuXi, 60 days |
| secondary | CNRM | 25-member ensemble and 42+ days, but only weekly starts |

Use the four primary models for the main balanced tables. CNRM can be reported
on the 52-start/year Thursday subset or as a sensitivity; including it in every
main comparison would either halve the sample or create unequal initialization
counts.

CMA adds useful institutional and model diversity at modest cost because it has
only three perturbed members. Its day-1 smoke test passed, but the approximately
199 mm ensemble-mean grid-cell TP maximum should be checked against ERA5/IMERG
before scaling production. It may be a legitimate localized event or a
provider-specific accumulation/encoding issue.

## Other ECDS Origins

- `KMA` is a valid optional robustness model: daily forecasts reach 60 days,
  but its reforecasts use only four calendar starts per month.
- `ECCC`, `JMA`, and `CNR-ISAC` do not provide the full common 42-day forecast
  window in the live archive used here.
- `BoM`, `CPTEC`, and `IAP-CAS` do not cover all five operational years
  2020-2024.
- `HMCR` structurally qualifies, but adds less benchmark value than CMA or the
  richer CNRM ensemble for the initial paper scope.

## AI Models

`FuXi-S2S` remains the primary AI benchmark: its available weights and local
2002-2021 archive provide 42 daily leads and 51 total members.

`FengWu-W2S` is the most relevant additional research candidate. Its paper
describes six-hourly global forecasts through 42 days, including T2M and TP.
Add it only after confirming public weights, inference code, and a hindcast
calendar that can be aligned with this experiment. A paper description alone
is not enough for a reproducible benchmark.

Medium-range AI systems such as GraphCast, Pangu-Weather, FourCastNet, and
GenCast are not direct replacements because their standard forecast horizons
do not cover the complete six-week contract. ECMWF AIFS can be a future or
prospective comparison, but it does not provide a like-for-like 2020-2024
operational archive for this study.

Primary references:

- [FuXi-S2S paper](https://arxiv.org/abs/2312.09926)
- [FengWu-W2S paper](https://arxiv.org/abs/2411.10191)
- [Official S2S model configurations](https://confluence.ecmwf.int/spaces/S2S/pages/40796876/Models)

# Presentation Brief — Case Study 1: Feb 12–18 2026 Dry Ridge & Warm Anomaly
**Audience:** Spire director + scientists · **Goal:** clear, honest, SPIRE-favourable story that survives scientific questioning.

---

## 1. The one-line message
On a real, IMD-documented Indian extreme (the 12–18 Feb 2026 dry ridge & warm spell),
**SPIRE captured the event 1–2 weeks ahead** — right warm-core *pattern*, near-zero bias —
where the dynamical (ECMWF, NCEP) and other AI (FuXi-S2S) systems ran cold and damped it.

## 2. Story arc (3 acts → ~3 slides)
1. **The event is real.** IMD Press Release, 19 Feb 2026: max temps **+4–6 °C above normal** over
   W-Himalaya & NW-India plains, only one weak active western disturbance → effectively dry.
   *(Slide: quote the IMD release — `paper/results/20260219_pr_4745.pdf`.)*
2. **What ERA5 saw.** `fig1_era5_truth.png` — positive Z500 ridge (N India), absolute dryness
   (Central/N), warm core (+4–6 °C, Rajasthan/Punjab/W-Himalaya).
3. **Who forecast it.** `fig2a_matrix_20260212.png` (Week-1 lead) and
   `fig2b_matrix_20260205.png` (Week-2 lead) + the numbers below.

## 3. Key numbers (this case, India IMD-union land, cosine-weighted)
**Lead the talk with PATTERN CORRELATION (PCC) — it is the fair, reference-independent metric.**

| Variable | Metric | SPIRE | FuXi | ECMWF | NCEP |
|---|---|---|---|---|---|
| **T2M** W1 | PCC | **0.83** | 0.12 | 0.12 | 0.28 |
| **T2M** W1 | Bias (°C) | **+0.07** | −4.8 | −5.2 | −4.4 |
| **T2M** W2 | PCC | **0.74** | 0.14 | 0.00 | 0.05 |
| **Z500** W1 | PCC | 0.99 | 0.98 | **0.99** | 0.97 |
| **Z500** W2 | PCC | **0.91** | 0.68 | 0.85 | 0.93 |

- **Headline:** SPIRE is the only system that reproduces the **warm-core pattern** (PCC 0.83 / 0.74)
  AND has near-zero temperature bias. Others miss the pattern and run several °C cold.
- **Z500 ridge:** everyone gets it at Week-1; at Week-2 SPIRE/NCEP/ECMWF hold it, FuXi weakens most.
- **Precip:** describe as **absolute desiccation** (near-zero rainfall, matching IMD) — do NOT quote
  precip PCC (see §5).

These mirror the paper's season-aggregate result (SPIRE T2M PCC 0.80 vs 0.26–0.42; precip PCC 0.84) —
so the case is a concrete instance of the overall ranking, not a cherry-pick.

## 4. What to SHOW vs what to MUTE
- ✅ Show: IMD release · Fig 1 · Fig 2a/2b · the PCC + bias table above.
- ✅ Show absolute precipitation for the dryness (not anomaly PCC).
- ⚠️ Mute / reframe: any "FuXi precip PCC = 1.00" number — it is a verification artifact (§5).
- ⚠️ Don't claim the full −5 °C T2M gap as pure model error — part is proxy/referencing (§5).

## 5. Tough questions — and your honest answers
**Q: FuXi shows precip PCC ≈ 1.0 — is FuXi best at rain?**
A: No — that's a degenerate artifact. It was a near-total-dry week, so absolute precip ≈ 0 for
both forecast and ERA5; the anomaly collapses to −climatology and correlates trivially regardless
of skill. That's why we report this case in **absolute** precip, not anomaly correlation.

**Q: The other models look 5 °C too cold — is the comparison fair?**
A: The *pattern* comparison (PCC) is fully fair and SPIRE wins it cleanly (0.83 vs ~0.1). The
*absolute-bias* gap is partly methodological: ECMWF/NCEP archive only 6-h max/min, so their daily
T2M is a (Tmax+Tmin)/2 proxy, and they are not ERA5-referenced the way SPIRE is. We flag this and
lean on PCC for the skill claim. (SPIRE's near-zero bias is genuine and reflects its ERA5 training.)

**Q: One event — how representative is this?**
A: It's an illustration of the season-aggregate ranking (13 inits, JFM 2026), not the evidence base.
Same conclusion holds in the pooled scores, regions, and the multi-model ensemble.

**Q: Single season — why trust it?**
A: We present it as an early, real-time assessment (one season, N=13) and say so explicitly; the
value is the reproducible real-time benchmark, not a climatological verdict.

## 6. Before you present (10-min checklist)
- [ ] Re-confirm the warm core in Fig 1c peaks +4–6 °C over Rajasthan/Punjab (matches IMD wording).
- [ ] Open both Fig 2 PNGs full-screen; check colorbars/labels read at projector size.
- [ ] Have the IMD PDF open as a backup slide.
- [ ] Memorize the 5-row table in §3; know the PCC-vs-bias distinction cold.
- [ ] (Optional, if time) regenerate the precip panel as **absolute mm/day** instead of anomaly.

---
*Figures & tables: this folder. Script: `case1_analysis.py`. Event source: `../results/20260219_pr_4745.pdf`.*

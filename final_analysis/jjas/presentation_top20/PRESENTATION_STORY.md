# JJAS FuXi-S2S vs ECMWF-S2S Presentation Story

## Core Story

The Indian summer monsoon is a high-impact subseasonal forecasting problem: rainfall decisions depend on weeks 1-6, but forecast skill is hard because monsoon rainfall is spatially organized, intermittent, and tied to large-scale circulation. The motivation is to test whether a data-driven global model, FuXi-S2S, can provide useful monsoon-range guidance compared with a strong operational physics-based benchmark, ECMWF-S2S, when both are verified against the same ERA5 valid windows.

The key methodological point is fairness. FuXi and ECMWF hindcasts do not always share the same initialization calendar, so the comparison is paired by ERA5 valid date window rather than by nominal initialization date alone. Every statistical claim should come back to this: same event, same verifying dates, same regions, same metrics.

The 2019 JJAS evidence is nuanced and defensible. FuXi shows suggestive rainfall structure advantages, especially for TP pattern correlation and early/central-region windows, but those all-India TP PCC/RMSE gains are not statistically robust after correction in the present 2019 sample. ECMWF is significantly better for TP absolute bias, and ECMWF remains stronger or more stable for Z500 overall. The current message is not "FuXi wins"; it is "FuXi is competitive for monsoon rainfall structure, with bias and circulation limitations that must be tested across the full hindcast archive."

## Slide Order: Best 20 Plots

1. `plots/01_truth_jjas_rainfall_climatology.png`
   Use this to define the target: JJAS monsoon rainfall over India is spatially heterogeneous and high-impact.

2. `plots/02_monsoon_circulation_jjas.png`
   Establish that rainfall skill depends on large-scale monsoon circulation, not just local rain amounts.

3. `plots/03_active_break_climatology.png`
   Show why subseasonal forecasts matter: active and break spells recur every monsoon season and typically persist for several days.

4. `plots/04_matched_valid_window_calendar.png`
   Explain the fair comparison design: forecasts are matched by ERA5 valid windows.

5. `plots/05_case_tp_2019_06_28.png`
   First case study: compare FuXi, ECMWF, and truth for rainfall structure.

6. `plots/06_case_z500_2019_06_28.png`
   Companion circulation case: check whether model rainfall behavior is dynamically consistent.

7. `plots/07_lead_skill.png`
   Main skill overview: how TP and Z500 skill changes from week 1 to week 6.

8. `plots/08_regional_skill_scorecard.png`
   Move from all-India to monsoon regions; this prevents overclaiming from a single average.

9. `plots/09_meeting_claim_matrix.png`
   Put the defensible claims in one place: what is statistically supported, what is only suggestive.

10. `plots/10_forest_pcc_gain.png`
    Effect-size evidence for pattern-correlation gain with bootstrap confidence intervals.

11. `plots/11_forest_rmse_reduction.png`
    Effect-size evidence for RMSE reduction; useful to discuss magnitude, not just significance.

12. `plots/12_forest_abs_bias_reduction.png`
    Important limitation slide: ECMWF has a clear TP absolute-bias advantage in the 2019 sample.

13. `plots/13_fdr_heatmap_pcc_gain.png`
    Where and when FuXi helps: region-by-lead PCC gains and FDR markers.

14. `plots/14_paired_scatter_pcc_rmse.png`
    Visual proof of paired comparison: each point is a matched valid window.

15. `plots/15_delta_distribution_by_lead.png`
    Show spread and robustness of FuXi-minus-ECMWF deltas by lead week.

16. `plots/16_rainfall_intensity_conditioning.png`
    Ask whether performance changes in dry, normal, and wet monsoon windows.

17. `plots/17_error_phase_space.png`
    Summarize the trade-off between pattern skill, RMSE, and bias.

18. `plots/18_top_bottom_tp_windows.png`
    Use as qualitative examples: when FuXi performs best and worst.

19. `plots/19_probabilistic_crpss_ssr.png`
    If ensemble/probabilistic evaluation is discussed, use this to show CRPSS and spread-skill behavior.

20. `plots/20_crpss_by_week.png`
    Backup probabilistic lead-time slide: whether ensemble skill survives beyond early weeks.

## One-Minute Speaking Script

"The goal is to evaluate whether FuXi-S2S, a data-driven subseasonal model, can provide useful JJAS monsoon guidance relative to ECMWF-S2S. The monsoon is a difficult S2S target because rainfall is intermittent, regionally structured, and dynamically linked to circulation. To keep the comparison fair, I match FuXi and ECMWF forecasts by the same ERA5 valid date windows, then evaluate TP and Z500 across all-India and monsoon regions using PCC, RMSE, bias, and ensemble diagnostics. In the current 2019 matched-window sample, FuXi is competitive and shows suggestive TP pattern and RMSE advantages, especially in some early/central monsoon windows, but these are not robust all-India statistical wins after FDR correction. ECMWF is clearly better for TP absolute bias and remains stronger for Z500 overall. So the defensible conclusion is that FuXi has promising rainfall-structure skill but needs bias correction and full multi-year validation before claiming climatological superiority."

## Best Short Version

## 2019 Caveat

The active-break motivation plots can be multi-year ERA5 diagnostics. The FuXi-vs-ECMWF comparison is currently 2019 because the matched FuXi compact outputs available now cover 2019 only; the all-year extraction and matched-window aggregation jobs are still running. Once those finish, the same statistical plots should be regenerated for the full hindcast period and the claim matrix can be upgraded from "2019 case-study evidence" to multi-year evidence.

For a short meeting, show only plots 1, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, and 19. Keep the remaining plots as backup for questions.

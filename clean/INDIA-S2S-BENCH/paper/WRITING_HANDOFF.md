# Writing handoff

## One-sentence paper

On 35 common JJAS 2025 initializations, forecast-conditioned XGBoost mixing
improves an equal-weight seven-system rainfall ensemble over India, mainly at
Weeks 1--4, but does not robustly beat a per-lead validation-selected
individual baseline and fails in east/northeast India.

That bounded result is the paper. Do not turn it into “PiggyCast is best,”
“all models are bad,” or an operational-impact claim.

## Canonical files and versioning

- Edit paper/main.tex only.
- Build paper/main.pdf only.
- Use Git commits for milestones; do not create draft-v2, final-final, or
  alternative manuscript files.
- Generated numbers, tables, and figures come from make_paper.py. Do not change
  a generated score by hand.

## Four-page main-paper structure

1. Motivation and narrow question.
2. Common-date benchmark and pre-specified split.
3. Baselines, PiggyCast variants, metrics, and paired uncertainty.
4. National result and stronger comparator.
5. Lead boundary and regional failure.
6. Climate relevance, responsible use, limitations, and reproducibility.

Inventory, all seven systems by week, complete metrics, coverage, regional
details, bootstrap sensitivity, QC, hyperparameters, and neural exclusion
belong in the appendix.

## Headline assets

- Figure 1: common-grid/common-date evaluation pipeline.
- Figure 2: ACC and RMSE versus lead for only the five claim-relevant methods.
- Figure 3: one paired-effect forest plot containing national contrasts and
  four regions.
- Table 1: full and forecast-only PiggyCast, validation-weighted and
  equal-weight means, per-lead selected baseline, and ECMWF; columns are ACC,
  RMSE, MAE, and bias.
- Appendix Figure A1: all seven systems plus equal weighting and full
  PiggyCast, Weeks 1--6, as IMD-referenced ACC/RMSE heatmaps.
- Appendix Figure A2: full-minus-equal ACC by IMD region and lead.

If the additive bias baseline is run, add it to Table 1 with an explicit
exploratory marker. Put climatology in a companion error table with ACC shown
as undefined.

## Language discipline

Safe:

- “improves equal weighting on average”
- “uses forecast-state information”
- “pre-specified 2025 evaluation within the released workflow”
- “paired moving-block percentile interval”
- “forecast-quality evidence relevant to adaptation”

Avoid:

- “uniformly superior,” “state of the art,” or “best model”
- “proves multi-model complementarity” without leave-one-model-out evidence
- “fully reproducible” if raw provider data are not distributable
- “untouched 2025” as a claim about every prior project
- “data-efficient” without a scaling study
- operational benefit, avoided loss, or improved decisions without a
  decision-value evaluation

## Current official submission checks

The active official OpenReview form was checked on 2026-08-17:

- paper track: at most four content pages; references excluded;
- optional appendices must be in the same PDF, and should not be assumed exempt
  from limits unless the author verifies this manually;
- one anonymized PDF, at most 50 MB;
- double-blind metadata, PDF, files, and review-visible artifact links;
- official CCAI 2026 workshop template required;
- title, abstract, keywords, track, climate area, ML area, contribution type,
  prior-publication status, code/data status, and a 50--2,000-character
  climate-problem/contribution statement are portal fields;
- deadline: 2026-08-29 23:59 AoE.

Source: https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FTCCML

Important: the current CCAI author policy says CCAI website and template
materials must not be shared with an LLM during paper preparation. An author
must personally retrieve and apply the official template, inspect its exact
formatting rules, and complete the required policy confirmations. Do not paste
those materials back into this coding workflow.

## Portal contribution statement draft

Weekly rainfall guidance at two-to-six-week leads is relevant to agricultural,
water, and flood preparedness in India, yet forecast comparisons are often
confounded by different initialization dates, grids, and climatologies. We
construct a common-date JJAS benchmark of seven physics, AI, and hybrid systems
against IMD rainfall and test a lightweight forecast-conditioned XGBoost
mixer without retraining the forecasting models. On 35 common 2025
initializations, the mixer improves equal weighting in national ACC, RMSE, and
MAE, but its gain is uncertain against a validation-selected individual
baseline, disappears at Weeks 5--6, and reverses in east/northeast India. The
contribution is an auditable climate-relevant benchmark and a failure-aware
assessment of where inexpensive post-processing does and does not help.

Suggested portal categories: climate adaptation/resilience; weather or climate
forecasting; climate-relevant benchmark plus academic methodology. Select the
artifact-availability option that is true at upload time rather than promising
a release that is not accessible.

## Manual author tasks

- Replace Anonymous authors only after the review policy permits it.
- Apply the official template personally and confirm the main content fits four
  pages after reflow.
- Insert an anonymized artifact URL or choose the accurate portal
  code/data-availability status.
- Check every author has an OpenReview profile.
- Complete prior-publication and reciprocal-reviewer fields honestly.
- Recheck title, abstract, figures, accessibility, references, PDF size, and
  anonymity immediately before upload.
- Read the compiled PDF, not only the LaTeX source.

No drafting or experiment can guarantee a top-paper outcome. The strongest
submission strategy is a clear climate-relevant benchmark, a useful bounded
positive result, and unusually honest failure analysis.

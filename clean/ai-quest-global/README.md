# Global TP probability adapter

This is a small global rainfall-calibration experiment. It converts the FuXi
ensemble into five ERA5 climatological categories and lets a 111,909-parameter
U-Net make a limited spatial correction to those probabilities.

It predicts two seven-day totals:

- D19–25
- D26–32

The output for each period is `probability(5, 121, 240)` on the global
1.5-degree grid.

Storage stays simple: the 13 TB hindcast and the compact prepared cache are
Zarr stores, the selected weights are one PyTorch `.pt` file, evaluation
tables are CSV, figures are PNG, and each local forecast period is NetCDF.

## What the model does

For each period, every FuXi member is summed over seven days and placed into
one of five ERA5 rainfall categories. The member counts receive a small
Jeffreys correction, `(count + 0.5) / (members + 2.5)`, so every category has
non-zero probability. This gives the FuXi ensemble anchor `p0`. The U-Net
predicts only a correction:

```text
FuXi weekly ensemble ──> smoothed category anchor p0
                                      │
18 input maps ──> small U-Net ──> correction logits
                                      │
              softmax(log(p0) + correction) ──> 5 probabilities
```

The final layer starts at zero, so the new network initially gives exactly
the FuXi `p0` anchor. If training does not beat that anchor on 2019 validation
RPS, `train.py` keeps the zero-correction checkpoint.

The 18 inputs are five `log(p0)` maps, five `log1p` FuXi-member rainfall
quantiles, latitude and longitude sine/cosine, valid-date sine/cosine, a period
flag, and land fraction. There is no VQ-VAE and no extra atmospheric variable
in this first experiment.

## Data split

- Train: 2017–2018
- Validation/model selection: 2019
- Untouched test: 2020–2021

The archive contains 104 fixed calendar initialization dates per year. Their
weekdays drift between years, so the primary score uses every initialization;
weekday filtering would discard different dates in different years. Using
2017 onward also avoids evaluating on years included in the published FuXi
base-model training period, which ended in 2016.

ERA5 categories use 100 historical samples: the preceding 20 years and date
offsets `[-4, -2, 0, 2, 4]`. Equality with a boundary enters the upper
category. Cells with four identical precipitation boundaries are excluded.

## Environment

The existing environment with the compatible Zarr v2 reader is:

```bash
PYTHON=/home/raj.ayush/.conda/envs/fuxi/bin/python
```

For official ERA5 retrieval, install the current Quest package and ECBox
client if they are not already present:

```bash
$PYTHON -m pip install 'AI-WQ-package>=3.26'
$PYTHON -m pip install sites-toolkit \
  -i https://get.ecmwf.int/repository/pypi-all/simple
```

Do not put the ECBox token in a script. Load it for the current shell:

```bash
read -s AI_WQ_PASSWORD
export AI_WQ_PASSWORD
```

## Run

First verify the existing FuXi archive:

```bash
$PYTHON prepare_data.py inspect-fuxi
```

Download ERA5 weekly rainfall for the target years and the preceding
climatology years:

```bash
$PYTHON prepare_data.py download-era5
```

Build the compact cache. The job is resumable by initialization and is the
slow part because it reads the large source archive:

```bash
sbatch slurm/prepare.sbatch
```

Train one model on one A100:

```bash
sbatch slurm/train.sbatch
```

For a direct foreground run:

```bash
$PYTHON train.py --device cuda
```

`train.py` writes `best.pt`, `history.csv`, `figures/loss_curve.png`, and the
complete run configuration under:

```text
/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_global/runs/
```

Evaluate uniform climatology, the FuXi `p0` anchor and the selected checkpoint:

```bash
$PYTHON evaluate.py \
  --cases /storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_global/cache/fuxi_tp_2017_2021.zarr \
  --years 2020 2021 \
  --checkpoint RUN_DIRECTORY/best.pt \
  --output-dir RUN_DIRECTORY/test \
  --device cuda \
  --batch-size 8
```

Or submit the same evaluation as a GPU job:

```bash
sbatch slurm/evaluate.sbatch RUN_DIRECTORY/best.pt RUN_DIRECTORY/test
```

The test directory contains `evaluation_summary.csv`, a lead-wise RPS/RPSS
figure, pooled reliability curves, two spatial maps of `p0 RPS - model RPS`,
and `india_summary.csv` for the 5–40°N, 65–100°E India box. Positive values on
the spatial maps mean the neural correction improved on `p0`.

Create two local NetCDF files for one prepared initialization:

```bash
$PYTHON predict.py \
  --case /storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_global/cache/fuxi_tp_2017_2021.zarr \
  --init-date 2020-06-02 \
  --checkpoint RUN_DIRECTORY/best.pt \
  --output-dir RUN_DIRECTORY/local_forecast
```

For a new global FuXi run, first convert its raw member files into one small
prediction case using the training-only normalization saved in the cache:

```bash
$PYTHON prepare_data.py prepare-case \
  --raw-directory /path/to/global_run/raw \
  --init-date 2026-07-28 \
  --output /tmp/fuxi_2026-07-28_quest_case.npz

$PYTHON predict.py \
  --case /tmp/fuxi_2026-07-28_quest_case.npz \
  --checkpoint RUN_DIRECTORY/best.pt \
  --output-dir RUN_DIRECTORY/local_forecast
```

`prepare-case` reads only TP from leads 19–32, uses every available member,
and checks the run metadata date when it is present.

Run all synthetic checks without reading the large archive:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python train.py --smoke --device cpu --max-epochs 1 --run-dir /tmp/quest-tp-smoke
```

## Files

- `prepare_data.py`: ERA5 retrieval and FuXi-to-Zarr preprocessing
- `data.py`: weekly/category/feature rules
- `model.py`: the complete neural network
- `train.py`: RPS training and validation selection
- `metrics.py`: official-style RPS and RPSS
- `evaluate.py`: baseline comparison and plots
- `predict.py`: offline local NetCDF export
- `slurm/`: small preprocessing, training, and evaluation launch scripts

IMERG is not used for fitting this first model. Once the current global IMERG
download is complete, it can be added as an independently labelled sensitivity
test without changing the ERA5 category definition.

## Permission boundary

FuXi's published terms prohibit competition use without written permission
from the authors. All code here is therefore offline research code: it contains
no upload or submission function. Do not submit FuXi-derived probabilities or
weights until that written permission explicitly covers the competition.

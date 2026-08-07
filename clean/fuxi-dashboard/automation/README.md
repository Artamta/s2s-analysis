# FuXi forecast automation

This directory contains the scheduler layer for the existing FuXi operational
driver. It is intentionally independent by source:

- GFS issues are considered only on Wednesday and Saturday. The controller
  polls every 15 minutes from 00:15 through 12:00 UTC and submits once all 16
  inputs are present.
- ERA5 is a delayed reference. A daily watcher revisits pending Wednesday and
  Saturday dates and submits only after all 768 ARCO objects are present.
- One source never waits for the other. A delayed ERA5 issue can be published
  later and paired with an already-published GFS issue by the publisher.

No timer or crontab is installed by this repository. The examples in `cron/`
and `systemd/` become active only after an operator reviews and installs one of
them.

## Safety boundary and automatic publication

The example config uses `submission_mode: inference_only`. In this mode the
controller submits only the existing staging and inference Slurm scripts. It
does **not** schedule the legacy publication job into the scientist's working
checkout. After a run manifest validates, the state record first contains this
auditable hand-off:

```json
{
  "next_action": {
    "kind": "export_and_publish_from_clean_clone",
    "source": "gfs",
    "issue_date": "20260808",
    "config": "/private/path/to/fuxi_gfs_20260808_ens100.json",
    "private_forecast": "/private/path/to/forecasts/annual2026/20260808.nc",
    "private_manifest": "/private/path/to/manifests/annual2026/20260808.json"
  }
}
```

The example configuration consumes that action automatically during the next
`reconcile`. `export_and_publish_operational_issue.py` creates a disposable
worktree from the dedicated publishing clone, points the existing exporter at
that isolated dashboard, and then hands its public package to the allow-listed
publisher. A second disposable worktree validates, builds, commits, and pushes
without force. The persistent clone and research checkout remain clean on
failure. The receipt advances `validated → exported → pushed`; later reconcile
runs use the deployed-site smoke test to advance `pushed → deployed`.

Removing `publication.command` retains the explicit hand-off for sites that
want a separate queue worker. `legacy_driver` remains a manual compatibility
mode; it is not the production default.

The submission guard is written before the first `sbatch` call. If `sbatch`
times out or returns an ambiguous result, the issue becomes `failed` and is not
automatically resubmitted. This prevents an expensive duplicate 100-member run.

## One-time private setup

1. Copy `config.example.json` to
   `/home/raj.ayush/.config/s2s-fuxi/scheduler.json` and restrict it to the
   service account. Keep runtime configuration and the state directory outside
   Git.
2. Set `schedule.start_date` to the first issue the automation should own.
   Dates before this boundary are never discovered, which prevents an
   accidental historical backfill.
3. Keep `dashboard_root` pointed at a read-only local dashboard for adopting
   already-exported issues. New exports use disposable publishing worktrees.
4. Create `/home/raj.ayush/.config/s2s-fuxi/automation.env` from `env.example`
   and add the private alert address. Do not commit the filled file. The cron
   launcher fails closed unless both private files are owned by the service
   user, are regular non-symlink files with no group/other permissions, and the
   alert address is populated.
5. Add the repository-scoped write deploy key, institution-owned Git author,
   correct Pages URL, and private alert address to the private environment.
6. Confirm the configured Python, operational driver, Slurm scripts, state
   storage, `sbatch`, and local `sendmail` paths, then install the reviewed root
   Pages workflow and exactly one cron/systemd schedule.

The controller itself uses only the Python standard library.

## Cron on the production host

The production host uses `Asia/Kolkata` and Debian cron does not honor a
per-user `CRON_TZ`. The tracked cron example therefore runs a lightweight GFS
heartbeat every 15 minutes and leaves Wednesday/Saturday 00:15-12:00 UTC
eligibility to the controller. ERA5 runs at 18:35 IST (13:05 UTC), while
reconcile runs every ten minutes at a two-minute offset so it never starts on
the same minute as the GFS heartbeat.

Cron calls `automation/run_scheduler.py`, not the controller directly. The
launcher safely parses the allow-listed private environment without sourcing
or evaluating shell text, validates the private scheduler config, removes
`PYTHONHOME`/`PYTHONPATH`, supplies a deterministic non-interactive `PATH`, and
then replaces itself with the controller. If Node is not installed system-wide,
set `PATH` in the private environment to a stable Node installation; a `/tmp`
installation is not suitable for unattended runs. Output is sent to the host
logger so cron does not generate mail for every no-op heartbeat.

No crontab is installed by the repository. After the private files, mail
transport, deploy key, clean clone, stable Node runtime, and dry runs are
verified, review and install `automation/cron/fuxi-automation.crontab.example`
as the service user.

## Rehearsal

Dry runs do not probe, submit, or write scheduler state:

```bash
/usr/bin/python3 automation/fuxi_scheduler.py \
  --config /home/raj.ayush/.config/s2s-fuxi/scheduler.json \
  --dry-run --now 2026-08-08T00:15:00Z gfs-poll

/usr/bin/python3 automation/fuxi_scheduler.py \
  --config /home/raj.ayush/.config/s2s-fuxi/scheduler.json \
  --dry-run --now 2026-08-10T13:00:00Z era5-watch
```

To rehearse the exact cron path, use the launcher after filling both private
files:

```bash
/usr/bin/python3 automation/run_scheduler.py \
  --dry-run --now 2026-08-08T00:15:00Z gfs-poll
```

Use `--date YYYYMMDD` only for a deliberate Wednesday/Saturday recovery. An
explicit date bypasses the GFS clock window but not the cadence or configured
start boundary.

For the requested no-inference replay, use a temporary config with a disposable
state root and `start_date: 2026-07-29`, keep `--dry-run`, and pass `--date`
explicitly for GFS 20260805 and ERA5 20260730. The existing GFS 100-member
export passes the adoption checks. ERA5 20260730 is both a Thursday and a
five-member historical experiment, so the production scheduler reports a
cadence skip. No probe, Slurm job, or state write occurs in either dry run. Do
not reuse the temporary earlier start date in the enabled timer config.

## Commands and state

```bash
python3 automation/fuxi_scheduler.py --config PRIVATE.json gfs-poll
python3 automation/fuxi_scheduler.py --config PRIVATE.json era5-watch
python3 automation/fuxi_scheduler.py --config PRIVATE.json reconcile
python3 automation/fuxi_scheduler.py --config PRIVATE.json status
```

Every source/date has an atomic JSON record under
`STATE_ROOT/issues/{gfs,era5}/YYYYMMDD.json`. Normal progression is:

```text
pending -> submitted -> validated -> exported -> pushed -> deployed
                  \-> failed
```

`reconcile` audits submitted manifests, publishes validated runs when the clean
worker is configured, and confirms the Pages deployment. For compute-only
submissions it also checks `sacct` and immediately fails/alerts on cancelled,
timed-out, out-of-memory, node-failed, or otherwise failed stage/inference
jobs. The `mark` commands remain available for manual recovery or an external
worker:

```bash
# Export the validated private run and publish from disposable worktrees:
python3 scripts/export_and_publish_operational_issue.py \
  --source gfs --issue 20260808 --config /private/path/case.json \
  --receipt /private/state/publish-gfs-20260808.json

python3 automation/fuxi_scheduler.py --config PRIVATE.json mark \
  --source gfs --date 20260808 --stage exported

python3 automation/fuxi_scheduler.py --config PRIVATE.json mark \
  --source gfs --date 20260808 --stage pushed --commit-sha COMMIT_SHA

python3 automation/fuxi_scheduler.py --config PRIVATE.json mark \
  --source gfs --date 20260808 --stage deployed --commit-sha COMMIT_SHA
```

The publisher reads its dedicated clone, repository URL, branch, dashboard
subdirectory, deploy-key command, and institutional Git author identity from
`S2S_PUBLISH_CLONE`, `S2S_REPOSITORY_URL`, `S2S_PUBLISH_BRANCH`,
`S2S_DASHBOARD_SUBDIR`, `GIT_SSH_COMMAND`, `S2S_GIT_AUTHOR_NAME`, and
`S2S_GIT_AUTHOR_EMAIL`. Its atomic private receipt contains the source/issue,
base and commit SHAs, public forecast checksum, pushed flag, and changed-path
allowlist. Run it with `--dry-run` before the first real push.

Stage changes are monotonic and idempotent. A `pushed` or `deployed` mark needs
a 40-64 character Git SHA. To recover a failed non-ambiguous probe, use
`resume`. If submission started but its result was ambiguous, first inspect the
Slurm queue and state receipt; only then use
`--acknowledge-safe-to-resubmit`.

## Alerts and retention

Terminal failures are sent through local `sendmail` only when
`S2S_ALERT_EMAIL` is present and syntactically safe. The address is never read
from a tracked file and is not written to public status data. When mail is not
configured, the failure remains visible in the private state record with
`last_alert.sent: false`.

Website archive retention (12 months for interactive forecast data, eight
weeks for PDFs, metadata indefinitely) belongs to the clean publisher. The
scheduler keeps its private operational state indefinitely unless the storage
operator applies a separate reviewed retention policy.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/automation
```

The focused suite covers cadence, duplicate prevention, delayed ERA5 backfill,
retry/backoff, hard timeouts, safe compute-only submission, clean publication,
deployment confirmation, reconciliation, cron cadence, private-file
permissions, non-evaluating environment parsing, and dry-run behavior.

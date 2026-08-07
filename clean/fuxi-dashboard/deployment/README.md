# Operational dashboard release

The live-data publisher uses a dedicated Git clone, separate from scientific
model outputs and developer worktrees. Each release runs in a disposable
detached worktree, copies a small public allowlist, regenerates the complete
checksum manifest, validates every cataloged issue, builds the TypeScript site,
and performs a normal fast-forward push. It never force-pushes and refuses a
dirty publishing clone. A failed gate removes only that disposable worktree,
so the next retry starts from a clean `origin/main` again.

## One-time setup

1. Create a GitHub deploy key with write access for this repository only. Keep
   its private half outside all Git checkouts with mode `0600`.
2. Copy `operational-publish.env.example` to private configuration storage,
   replace its key path, public institution-owned Git identity, and private
   institutional alert address, then load it in the scheduler service
   environment. The publisher refuses the `.example` Git author placeholder so
   an automated commit cannot silently expose a developer's personal identity.
3. Install the repository-root workflow after reviewing the template:

   ```bash
   python clean/fuxi-dashboard/scripts/manage_root_workflow.py --install
   ```

   GitHub does not discover the older workflow nested inside the dashboard.
   CI can subsequently assert the installation with `--check`.

## Rehearsal and publication

Run a no-Git-mutation rehearsal against an already exported forecast:

```bash
python scripts/publish_operational_site.py \
  --source gfs --issue 20260805 --dry-run
```

The live command omits `--dry-run` and receives `S2S_PUBLISH_CLONE` and the SSH
command from private service configuration. It validates that the selected
package contains 100 members. Pass `--expected-forecast-sha256` from the private
pipeline state to bind publication to the validated model artifact. A JSON
receipt can be written to private state with `--receipt`; it records the base
and publication commits without exposing credentials.

For end-to-end scheduled operation, the scheduler invokes
`scripts/export_and_publish_operational_issue.py`. That bridge targets the
existing private-run exporter at a disposable worktree before calling this
publisher, so neither the research checkout nor the persistent publishing
clone receives generated web files.

The GitHub Pages workflow independently stamps the deployed commit, audits the
current GFS issue, delayed ERA5 reference, entire retained archive, and global
demo, then type-checks and builds. Validation or build failures occur before
deployment and preserve the previously successful site. After deployment the
workflow fetches the live catalog and both current forecast packages, checking
their SHA-256 values and expected commit. A post-deploy smoke failure makes the
workflow visibly red for alerting and requires operator review; it does not
claim an automatic Pages rollback.

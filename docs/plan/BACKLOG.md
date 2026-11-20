# Backlog
Status: In progress

The one living record of open work. Rows move to Done with the commit that closed them and are
never deleted. Evidence is file + symbol, never a line number. `Verified` is the date the row was
last checked against the tree — not when it was opened.

## Open

| ID | Opened | Verified | Pri | Finding | Evidence | Effort | Acceptance |
|---|---|---|---|---|---|---|---|
| B1 | 2026-11-20 | 2026-11-20 | P3 | Repro script sits at repo root instead of `test/` | `test_motion_semantics.py` | S | File lives under `test/` (or in `docs/debug/2026-01-24-motion-detection-semantics/`) and still runs |
| B2 | 2026-11-20 | 2026-11-20 | P3 | `setup.py` version lags `pyproject.toml` | `setup.py` `version="0.4.1"` vs `pyproject.toml` `version = "0.5.0"` | S | One version source, or both agree |
| B3 | 2026-11-20 | 2026-11-20 | P3 | `## [0.2.0] - Previous Version` has no date; motion fix released in v0.4.0 is filed under `[0.5.0]` | `CHANGELOG.md` | S | Decide whether released sections may be corrected; otherwise move to Done as won't-fix |

## Pending verification

| Item | Closed by | Verifies when | Fallback |
|---|---|---|---|
| `README.MD` → `README.md` rename fixes `readme = "README.md"` on case-sensitive CI | this commit | next `publish.yml` run on `ubuntu-latest` | check sdist contains `README.md` |

## Awaiting external

| Item | Blocked on | Ready when |
|---|---|---|

## Watching

| Item | Why not now | Trigger | Re-read |
|---|---|---|---|
| `docs/reference/ffmpeg-help.md` (920 KB `ffmpeg -h full` dump) | Kept for option lookup; regenerable | FFmpeg major upgrade or repo size concern → regenerate or drop | 2026-11-20 |

## Done

| ID | Finding | Closed by |
|---|---|---|

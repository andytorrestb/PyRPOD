# PyRPOD Operational Logging

PyRPOD has a formal operational logging system built entirely on the Python
standard library (`logging`, `hashlib`, `tracemalloc`, `subprocess`, ...). It
records what a run did — which case and assets were used, their checksums,
workflow phases, per-firing progress, performance, and the final status — so a
plume-strike run is fully traceable and reproducible.

Operational logs are **not** scientific result files. Full numerical fields
(per-face strike/pressure/shear/heat-flux arrays, JFH records, meshes) stay in
the existing VTK/STL/CSV/JFH outputs. The runtime log contains only summaries,
paths, dimensions, input checksums, and performance data.

## PyRPOD does not configure logging on import

Importing `pyrpod` (or any submodule) has **no logging side effects**: no log
files, no directories, no handlers, no root-logger changes, no output. Modules
obtain a logger the standard way and never configure handlers:

```python
import logging
logger = logging.getLogger(__name__)
```

An application turns logging on explicitly at its boundary.

## Minimal usage

```python
from pyrpod.logging_utils import configure_logging

case_dir = "case/rpod/base_case/"
session = configure_logging(case_dir)
try:
    # ... build the study and run the PyRPOD workflow ...
    #     e.g. study.jfh_plume_strikes()
    session.finalize("successful")
finally:
    session.close()
```

`configure_logging` attaches PyRPOD-owned handlers to the **`pyrpod` package
logger** (never the root logger), opens a fresh runtime log, writes the startup
metadata block, and snapshots the case configuration. `session.close()` detaches
and closes only the handlers PyRPOD installed; handlers belonging to an
embedding application are never inspected or modified. Calling
`configure_logging` again replaces PyRPOD's own handlers, so messages and
handlers are never duplicated.

### Runnable per-case demos

Several `case/rpod/` cases ship a ready-to-run driver and a `logging.ini` that
each highlight a different capability. They mirror the corresponding
`tests/rpod/` cases minus the assertions:

```bash
python case/rpod/base_case/run.py              # default INFO, console + file
python case/rpod/1d_approach/run.py            # progress_every_n_firings = 2
python case/rpod/koz/run.py                    # level = DEBUG (paths, array summaries)
python case/rpod/hollow_cube/run.py            # console = false (file only)
python case/rpod/multi_thrusters_square/run.py # kinetics on: per-firing pressure/shear/heat-flux
python case/rpod/stl_to_vtk/run.py             # asset + directory logging around STL -> VTK
```

Each writes its runtime log to `<case>/results/logs/<case>_<timestamp>.log`.

## Configuration file

Each case may optionally contain `<case_dir>/logging.ini`. When absent, the
built-in defaults are used and PyRPOD logs that it fell back to defaults. See
[`docs/logging.ini.example`](logging.ini.example) for a fully commented file.

```ini
[logging]
enabled = true
console = true
file = true
level = INFO
progress_every_n_firings = 1
log_array_stats = true
log_performance = true
snapshot_config = true
checksum_inputs = true
# format = %(asctime)s [%(levelname)s] %(name)s: %(message)s
```

- **`console`** and **`file`** are toggled independently.
- Console and file handlers share one level.
- No log rotation; retention is user-managed.
- Plain-text logging only (no JSON/JSONL).

## Configuration precedence

Highest wins:

1. Explicit `configure_logging(...)` keyword arguments
2. Environment variables — `PYRPOD_LOG_LEVEL`, `PYRPOD_LOG_FORMAT`
3. `<case_dir>/logging.ini`
4. Built-in defaults

```bash
# Raise verbosity for one run without editing any file:
PYRPOD_LOG_LEVEL=DEBUG python your_driver.py

# Override the record format:
PYRPOD_LOG_FORMAT="%(levelname)s %(name)s %(message)s" python your_driver.py
```

```python
# Explicit arguments beat everything, e.g. silence the console for a batch job:
session = configure_logging(case_dir, console=False, level="INFO")
```

## Runtime-log location and naming

Runtime logs are written under:

```
<case_dir>/results/logs/<case-name>_<timestamp>.log
```

`<case-name>` is the normalized final path component of `case_dir`; the
timestamp is `YYYYMMDD_HHMMSS`. A **new file is created for every configured
run**, e.g. `base_case_20260725_143102.log`. There is one primary runtime log
per run — not one per firing or artifact. (`.log` is reserved for these runtime
logs; expected-result fixtures use `.txt`/`.dat`.)

## Record format and levels

```
timestamp [LEVEL] module.name: message
2026-07-25 14:31:02 [INFO] pyrpod.rpod.PlumeStrikeEstimationStudy: Firing 17/95 completed: struck_faces=246 ...
```

| Level | Use |
|-------|-----|
| `DEBUG` | Candidate/resolved paths, array shapes and summary stats, per-thruster transforms, existing-directory messages, per-artifact paths, DataFrame summaries. |
| `INFO` | Run started, config loaded, assets loaded, JFH loaded, geometry/thruster config loaded, phase start/complete, progress updates, per-firing maxima, summary artifacts, run completed, final status. |
| `WARNING` | Optional config absent, optional visualization failed, parallel→serial fallback, questionable-but-usable input, nonessential write failed (with traceback). |
| `ERROR` | Failed requested operations, invalid required inputs (fail fast after logging), unrecoverable failures. |

`CRITICAL` is not used.

## Startup metadata

`configure_logging` writes a compact startup block once: absolute case path,
start time, Python version, CPU count, git dirty-working-tree status (via a safe
`git status --porcelain`; `unknown` if git/repo is unavailable), effective log
level, console/file enabled flags, the active runtime-log path, and whether
`logging.ini` or built-in defaults were used. The plume-strike run start (next
section) adds execution mode, worker count, and model parameters.

## Configuration snapshots

When `snapshot_config` is enabled, the case `config.ini` (and `logging.ini` if
present) are copied verbatim into `results/logs/` as
`<case-name>_<timestamp>_config.ini` / `..._logging.ini`, and their SHA-256
checksums are logged. Snapshots are for reproducibility and never replace the
original case files.

## Input-asset tracking

Each loaded input asset (target/visiting-vehicle STL, thruster/plume STL, JFH,
TCF, TDF, CCF, ...) is logged at INFO with its category, filename, absolute
resolved path, source (`case-local` vs `shared-data`), size in bytes, and
streaming SHA-256 (`checksum_inputs` toggles the hash). Checksums are cached per
run so an unchanged asset is not rehashed. Candidate paths and case-local vs
shared fallback decisions are logged at DEBUG (see
`pyrpod.util.io.fs.resolve_asset_path`).

## Plume-strike workflow logging

`PlumeStrikeEstimationStudy.jfh_plume_strikes()`:

- **Fail-fast validation** of every required input (config keys, readable
  non-empty target mesh / JFH / thruster config, valid thruster references; and,
  when kinetics is enabled, TDF / surface_temp / sigma / thruster metrics). A
  missing required input is logged at ERROR and raises — never a silent `None`.
- **Run started (INFO):** firings, target faces, kinetics on/off, plume model,
  radius, wedge angle, serial/parallel mode, worker count, progress interval.
- **Progress (INFO):** every `progress_every_n_firings` firings and the final
  firing — index, JFH id, sim time, active thrusters, worker PID, struck-face
  count, and (when kinetics is on) max pressure/shear/heat-flux/heat-flux-load,
  plus per-firing wall time and tracked-array memory. Kinetics-specific maxima
  are omitted (not faked) when kinetics is disabled.
- **DEBUG:** array summaries and per-artifact paths.
- **Completion (INFO):** firings completed, total struck-face events, unique
  struck faces, overall maxima, wall/CPU time, average wall time per firing,
  peak Python-tracked memory, tracked array memory, artifact count, final
  execution mode, whether fallback occurred, and final run status.

`tqdm` progress bars have been replaced by this logging-based progress.

## Performance and memory

Standard library only: `time.perf_counter()` (wall), `time.process_time()`
(CPU), `tracemalloc` (Python-managed current/peak allocations), and NumPy
`.nbytes` for tracked arrays. `tracemalloc` **does not** capture all native
memory allocated by NumPy/compiled libraries, so peak memory is reported as
`python_peak_memory_mb` (not total process memory) and known major arrays are
summed separately as `tracked_array_memory_mb`. To avoid slowing unit tests,
`tracemalloc` is engaged only for an explicitly configured run
(`configure_logging` called) with `log_performance = true`.

Unit-bearing field names are used for physical quantities, e.g. `pressure_pa`,
`shear_pa`, `heat_flux_w_m2`, `heat_flux_load_j_m2`, `wall_time_s`,
`cpu_time_s`, `python_peak_memory_mb`, `tracked_array_memory_mb`.

## Serial / parallel and fallback

Parallel execution uses parent-process logging: workers return only
non-numerical metadata (PID, per-firing wall/CPU time); the parent emits the
records. Numerical accumulation order is unchanged — cumulative arrays and VTK
output are produced serially in JFH order by the parent. Serial and parallel
modes emit the same major event categories.

If process-based execution fails, the full traceback is logged at WARNING
(`exc_info=True`), serial fallback runs, and the final status becomes
`successful_with_warning` when the serial calculation succeeds (or `failed` if
it does not).

## Final run status

One of `successful`, `successful_with_warning`, `partial`, or `failed`. Call
`session.finalize(status)` to record it along with warning/error counts,
runtime, and the runtime-log path. `session.close()` releases the handlers and
should run from a `finally` block so finalization happens even on exceptions
(without suppressing the original exception).

## Helper API summary

From `pyrpod.logging_utils`:

- `configure_logging(case_dir, **overrides) -> LoggingSession`
- `LoggingSession.finalize(status, summary=None)`, `LoggingSession.close()`
- `get_active_session()`, `get_settings()`, `performance_logging_enabled()`
- `log_asset(category, filename, resolved_path, case_dir, logger=None)`
- `summarize_array(name, array)`, `log_array_summary(logger, name, array, level=DEBUG)`
- `sha256_file(path)`, `git_dirty_status(path)`

## Future extensions (not implemented)

`run_summary.json`, output-artifact checksums, JSONL structured logs, log
rotation, true total-process memory via an optional external dependency, a
multiprocessing logging queue, and generated run IDs are documented as possible
future work and are intentionally out of scope for this version.

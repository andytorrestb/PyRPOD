# Prescribed plume-validation trade studies

This document covers the repaired `TradeStudy` architecture and the
package-level API for **prescribed** plume/target validation sweeps: studies
in which the firing poses are placed by the engineer (or generated from a
swept approach angle and source distance about a stationary target) rather
than flown from vehicle dynamics.

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Sweep modes: one JFH per case, or one for the sweep](#sweep-modes-one-jfh-per-case-or-one-for-the-sweep)
- [YAML configuration](#yaml-configuration)
- [`n_firings` and prescribed firings](#n_firings-and-prescribed-firings)
- [Integrated loads](#integrated-loads)
  - [Force](#force)
  - [Moment](#moment)
  - [Center of pressure](#center-of-pressure)
  - [Thermal quantities](#thermal-quantities)
  - [Coefficients](#coefficients)
- [Result schema](#result-schema)
- [VTK outputs](#vtk-outputs)
- [Optional plots](#optional-plots)
- [External reference data](#external-reference-data)
- [Adding another validation geometry](#adding-another-validation-geometry)
- [Known limitations](#known-limitations)

---

## Quick start

A complete flat-plate example ships with the repository. It runs the Cai 2016
Section-4 conditions (argon round jet, `D = 1 m`, `S0 = 2.0`, `T0 = 200 K`,
`Tw = 300 K`, fully diffuse) against the 8 m x 8 m plate of
`case/plume/plume_flat_plate_sweep`, with the plume source head-on at
`L = 4D`:

```python
from pyrpod.mdao.TradeStudy import TradeStudy

study = TradeStudy.from_config(
    'case/plume/plume_flat_plate_sweep/study/flat_plate_baseline.yaml')
results = study.run()

case = results.cases[0]
print(case.force)                 # [~0, ~0, -2.763]  N  (plume pushes -Z)
print(case.moment)                # ~0 about the plate center
print(case.center_of_pressure)    # ~[0, 0, 0]
print(case.max_pressure)          # 0.340 Pa
print(case.max_heat_flux)         # 50.1 W/m^2
print(case.coefficients['CF'])    # 0.0391
print(case.vtk_path)              # .../cases/case000_.../results/strikes/firing-0.vtu
print(results.summary_csv_path)   # .../case_results.csv
```

The multi-angle / multi-distance sweep (19 angles x 5 distances) is the same
call against `flat_plate_sweep.yaml`:

```python
study = TradeStudy.from_config(
    'case/plume/plume_flat_plate_sweep/study/flat_plate_sweep.yaml')
results = study.run()
study.plot()                       # optional trend figures
```

Outputs land under the configured `output_dir` (by default
`<case>/results/studies/<study name>/`, which is gitignored):

```
results/studies/flat_plate_sweep/
  jfh/case000_alpha0p0_d4.A            one prescribed JFH per case
  cases/case000_alpha0p0_d4/results/strikes/firing-0.vtu
  sweep_results.csv                    one row per case x component x firing
  sweep_metadata.json                  provenance + nested case records
  plots/                               optional trend figures
  reference_comparison.csv             written when reference data is configured
```

To run a committed configuration into a scratch location (this is what the
automated tests do), pass `output_dir`:

```python
TradeStudy.from_config(path, output_dir='/tmp/my_study').run()
```

---

## Architecture

`TradeStudy` is a thin façade. The work lives in small modules with one
responsibility each, all under `pyrpod/mdao/`:

| Module | Responsibility |
| --- | --- |
| `study_config.py` | Parse and validate the YAML study configuration |
| `firing_plan.py` | Build prescribed firings; write Jet Firing Histories |
| `plume_validation.py` | **Engine**: one JFH per case (`mode: per_case`) |
| `parameter_sweep.py` | **Engine**: one JFH for the sweep (`mode: single_jfh`) |
| `study_runtime.py` | Plumbing both engines share (assets, geometry, strikes, records) |
| `surface_loads.py` | Integrate per-face fields into component loads |
| `study_results.py` | Result schema; CSV + JSON output |
| `reference_data.py` | Generic external-reference comparison and metrics |
| `study_plots.py` | Optional sweep trend figures |

Everything domain-specific is delegated to the existing PyRPOD objects —
`JetFiringHistory`, `TargetVehicle`, `VisitingVehicle`, `MissionEnvironment`
and `PlumeStrikeEstimationStudy` — so a study inherits the pipeline's
validation, logging, plume physics and VTK conventions instead of
reimplementing them. The single plume model is `SimplifiedGasKinetics`; the
configuration records it explicitly and rejects anything else. No plume-model
registry is introduced.

`TradeStudy.from_config` picks the engine from `sweep.mode` (see the next
section); both do the same seven things, differing only in how many Jet
Firing Histories the sweep is decomposed into:

1. build the prescribed firings;
2. write the Jet Firing History and read it back with the normal
   `JetFiringHistory` parser;
3. run the plume-strike calculation;
4. integrate per-component loads;
5. export the per-face VTK fields;
6. record a structured result;
7. optionally compare against external reference data.

### Repairs to the legacy sweeps

The dynamics-driven sweeps (`run_axial_overshoot_sweep`,
`run_surface_cant_sweep`, `run_multi_var_sweep`) were broken before any
physics ran. Fixed on this branch:

- `init_trade_study` constructed `PlumeStrikeEstimationStudy.RPOD(case_dir)`
  — a class that does not exist — and passed a case directory where the study
  takes a `MissionEnvironment`. It now builds the environment and the real
  `PlumeStrikeEstimationStudy`.
- The sweeps called `jfh_plume_strikes(trade_study=True)`, which that
  method's `(parallel, workers)` signature rejects with `TypeError`.
- `print_mission_report` read impingement maxima from study attributes that
  are never set; it now summarizes them from the per-firing arrays
  `jfh_plume_strikes` returns (the old attribute path still works for
  existing callers).
- `interpret_mission_report` read `self.rpod.config`, which does not exist;
  the configuration lives on the study's `MissionEnvironment`.

### Execution paths

With VTK output enabled (the default) each case runs the full
`PlumeStrikeEstimationStudy.jfh_plume_strikes()` pipeline, with the target's
output root temporarily redirected to that case's folder so sweep cases
cannot overwrite one another's artifacts. With VTK output disabled the study
calls the pipeline's own per-firing core, `compute_plume_strikes()` — the
same function `jfh_plume_strikes` calls per firing, so the numbers are
identical — which keeps large sweeps and automated tests fast and
artifact-free. This applies to both engines.

---

## Sweep modes: one JFH per case, or one for the sweep

The same configuration can be decomposed two ways, chosen with `sweep.mode`.
Per-firing numbers are **identical** either way — an automated test asserts
it pose by pose — so the choice is about what you want to look at, not about
accuracy.

| | `per_case` (default) | `single_jfh` |
| --- | --- | --- |
| Engine | `PlumeValidationStudy` | `ParameterSweepStudy` |
| Jet Firing Histories | one per angle-distance case | **one**, spanning every pose |
| `case_id` / `firing_id` | one case per pose, firings 1..`n_firings` | one case, firings 1..`poses x n_firings` |
| Strike runs | one per case | one, over the whole history |
| VTK layout | `cases/<case_id>/results/strikes/firing-<i>.vtu` | `results/strikes/firing-<i>.vtu` — one series, scrubbable in ParaView |
| Cumulative fields | restart each case | accumulate across the sweep |
| Sweep envelope | n/a | recorded per component |

Use `per_case` when each pose is an independent experiment (a validation
matrix compared pose-by-pose against reference data): nothing one pose does
can affect another's artifacts. Use `single_jfh` when the poses form one
sequence and you want the pipeline's cumulative fields to mean something:

- `max_pressures` / `max_shears` — the worst load each face saw **anywhere**
  in the sweep;
- `cum_strikes` — a coverage map over the whole sweep;
- `cum_heat_flux_load` — accumulated heat-flux dose.

Those per-face fields live in the last firing's VTK file; their per-component
summary is exposed as `study.validation_study.envelope` and written into the
metadata document under `provenance.sweep_envelope`:

```python
study = TradeStudy.from_config('.../flat_plate_sweep_single_jfh.yaml')
results = study.run()
study.validation_study.envelope['plate']
# {'max_pressure': 0.340, 'max_shear_stress': 0.120,
#  'max_heat_flux_load': 126.95, 'total_strike_events': 31104.0,
#  'unique_struck_faces': 10368, 'swept_affected_area': 64.0,
#  'component_area': 64.0}
```

The envelope needs the pipeline's cumulative arrays, which only the full
strike path produces: with `output.vtk.enabled: false` it is reported as
empty rather than reconstructed from the per-firing records.

`n_firings` keeps one meaning in both modes — **entries contributed by each
pose** — so a configuration switches modes by changing one line. In
`single_jfh` mode the single history is required to hold exactly
`len(poses) x n_firings` entries, checked after writing and again after
reading the file back.

---

## YAML configuration

A study configuration is a **layer on top of an existing PyRPOD case**. The
case's `config.ini` keeps owning the STL, TCF, TDF, plume model and gating
geometry, so every existing case and public API is unaffected; the YAML adds
only what a trade study needs and an INI cannot express.

```yaml
study:
  name: cai2016_flat_plate_baseline        # required; identifies every result row
  description: ...
  case_dir: ..                             # required; relative to THIS file
  output_dir: ../results/studies/flat_plate_baseline

thruster:
  id: T1                                   # optional; validated against the case TCF

plume_model:
  name: SimplifiedGasKinetics              # the only accepted value
  parameters:                              # recorded for provenance only
    gas: argon
    speed_ratio_S0: 2.0

target:
  geometry_id: flat_plate_transformed.stl  # defaults to the case's [tv] stl
  reference_point: [0.0, 0.0, 0.0]         # sweep is built about this point
  normal: [0.0, 0.0, 1.0]                  # toward the plume-source side
  tangent: [1.0, 0.0, 0.0]                 # sweep plane's in-plane axis
  components:
    - name: plate
      selector: all                        # or face_indices: [...] / bounds: {...}

sweep:
  mode: per_case                           # per_case (default) | single_jfh
  plate_angles_deg: [0.0]                  # 0 = head-on along `normal`
  source_distances: [4.0]                  # from `reference_point`
  n_firings: 1                             # EXACT JFH entries per pose
  firing_duration_s: 1.0
  thrusters: [1]
  # firings: [...]                         # optional explicit poses; see below

loads:
  moment_reference_point: [0.0, 0.0, 0.0]
  normalization:                           # optional; omit for no coefficients
    reference_area: 64.0
    reference_length: 4.0
    dynamic_pressure: 1.1044652197738332
    reference_heat_flux: 637.3520362956127

output:
  vtk:      {enabled: true}
  summary:  {csv: case_results.csv, metadata: study_metadata.json}
  plots:    {enabled: false}

reference:
  path: null                               # optional external reference data
  label: null

metadata:
  coordinate_system: case global frame
  units: {}                                # overrides the SI defaults
```

Validation is strict and specific: a missing case directory, a case without a
`config.ini`, an unsupported plume model, an empty or non-positive sweep
axis, non-orthogonal target axes, duplicate component names, a non-positive
normalization value, or a firing list whose length disagrees with
`n_firings` each raise `StudyConfigError` naming the offending key.

Angles and distances are geometry, not physics: `normal` and `tangent` define
the plane the source is swept in, so a curved target simply supplies the axes
its sweep should use (see [Adding another validation
geometry](#adding-another-validation-geometry)).

---

## `n_firings` and prescribed firings

**`n_firings` is the exact number of Jet Firing History entries contributed
by each swept pose.** It is validated before anything is generated (positive
integer only — zero, negative, fractional and non-numeric values are
rejected), the generated sequence is asserted to match it, and the JFH read
back from disk is asserted to report the same count. In `per_case` mode that
is the exact length of every case's history; in `single_jfh` mode the one
shared history must hold exactly `len(poses) * n_firings` entries.

Two ways to supply the poses:

- **Generated** (the usual case): the pose comes from the swept
  `plate_angles_deg` / `source_distances` entry and is repeated for
  `n_firings` successive firing intervals — one JFH entry per firing, each of
  `firing_duration_s`.
- **Explicitly prescribed**: list the poses under `sweep.firings`. Their
  count must agree with `n_firings` (in `single_jfh` mode, with
  `len(poses) * n_firings`, the sequence being assigned to the poses in sweep
  order); a mismatch is an error, never a silent truncation or extension.

```yaml
sweep:
  n_firings: 2
  firings:
    - position: [0.0, 0.0, 4.0]
      dcm: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]   # first COLUMN = thruster axis
      thrusters: [1]
      duration_s: 0.5
    - position: [0.5, 0.0, 4.0]
      dcm: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
```

The generated pose convention, for a target reference point `C` with outward
normal `n_hat` and in-plane tangent `t_hat`:

```
d_hat(alpha) = cos(alpha) * n_hat + sin(alpha) * t_hat
source position = C + L * d_hat(alpha)
thruster axis   = -d_hat(alpha)                 (aimed at C)
```

`alpha = 0` is head-on. The JFH DCM carries the thruster axis as its first
**column**, which is what the strike pipeline reads as the plume normal. This
reproduces the committed sweep-JFH generators
(`case/plume/plume_flat_plate_sweep/jfh/generate_sweep_jfh.py`) to file
precision — a regression test pins all 95 poses of that file.

The dynamics-driven path was corrected to the same meaning:
`PlumeStrikeEstimationStudy.print_jfh_1d_approach_n_fire(..., n_firings=N)`
previously accepted `N` and ignored it. It now validates the count, stops the
approach simulation at exactly `N` entries, and raises if the approach
completes in fewer firings than requested.

---

## Integrated loads

`pyrpod/mdao/surface_loads.py` turns the strike pipeline's per-face arrays
into component-level resultants. Sign and direction conventions follow the
pipeline's own face-selection rule: a face is struck only when its unit
normal `n_hat` opposes the plume direction, i.e. points back toward the
source.

### Force

```
F_p,i = -p_i * A_i * n_hat_i                         (pressure, into the surface)
t_hat_i = normalize(u_hat_i - (u_hat_i . n_in_i) * n_in_i),  n_in_i = -n_hat_i
F_s,i = tau_i * A_i * t_hat_i                        (shear, along the tangential flow)
```

`u_hat_i` is the local flow direction — the unit vector from the plume source
to the face centroid, since the collisionless flow is radial. Faces at exactly
normal incidence have no tangential direction and contribute no shear force.
Reported: `pressure_force`, `shear_force`, `force`, `force_magnitude`.

### Moment

```
M_ref = sum (r_i - r_ref) x F_i
```

about the user-defined `loads.moment_reference_point`. The pressure and shear
contributions are reported separately (`pressure_moment`, `shear_moment`)
alongside the combined `moment` and `moment_magnitude`.

### Center of pressure

A unique three-dimensional center of pressure does not exist in general: the
resultant of a distributed load is a force **plus a couple**, and only the
moment component perpendicular to the force can be represented by shifting
the force's line of action. The definition used here is stated explicitly:

```
r_cop = r_ref + (F x M_ref) / |F|^2
```

— the point **on the line of action of the resultant force that lies closest
to the moment reference point**. The force-parallel moment component is
reported separately as `residual_couple`; no choice of `r_cop` can remove it.

Degenerate cases are handled explicitly and never return a misleading value:

| Situation | `center_of_pressure` | `center_of_pressure_status` |
| --- | --- | --- |
| Non-zero resultant | the point above | `ok` |
| Nothing loaded | `None` | `zero_load` |
| Resultant is cancellation noise (below `1e-6` of the summed face-force magnitudes, e.g. a closed target loaded symmetrically) | `None` | `ill_conditioned` |

`pressure_weighted_centroid` — `sum(p_i A_i r_i) / sum(p_i A_i)` — is
reported as an auxiliary, always-defined location; it coincides with the
classical center of pressure for a planar component under unidirectional
pressure.

### Thermal quantities

Instantaneous and peak quantities only (time-integrated heat dose is
deliberately out of scope on this branch): `max_heat_flux` (peak per-face
heat flux, W/m^2) and `total_heat_load` (`sum q_i A_i` over the component, W),
plus `max_pressure`, `max_shear_stress`, `affected_area` and `struck_faces`.
The per-face heat-flux load the pipeline already accumulates stays in the VTK
output.

### Coefficients

Coefficients are computed **only** when the configuration supplies every
input a given coefficient needs. Nothing is defaulted or invented.

| Coefficients | Required inputs |
| --- | --- |
| `Cp_max`, `Cf_max` | `dynamic_pressure` |
| `Cq_max` | `reference_heat_flux` |
| `CF`, `CFx`, `CFy`, `CFz` | `dynamic_pressure`, `reference_area` |
| `CM`, `CMx`, `CMy`, `CMz` | + `reference_length` |

With no `normalization` block at all, `coefficients` is empty and
`coefficients_available` is `False` — which is what the cylinder example
does, since no reference values exist for it yet.

---

## Result schema

`StudyResults` holds one `CaseResult` per **case x component x firing**, each
carrying enough metadata to reproduce and later compare the calculation. In
`per_case` mode there is one `case_id` per pose; in `single_jfh` mode every
record shares the study's single `case_id` and is distinguished by
`firing_id`, but both carry the same `plate_angle_deg` / `source_distance`,
so downstream code (CSV, plots, reference comparison) is mode-agnostic:

- **identity** — `study_name`, `case_id`, `component`, `firing_id`
- **geometry** — `geometry_id`, `mesh_faces`, `component_faces`,
  `component_area`, `coordinate_system`, `units`
- **pose and sweep** — `plume_source_position`, `plume_source_orientation`
  (9 DCM values), `target_normal`, `target_tangent`,
  `target_reference_point`, `plate_angle_deg`, `source_distance`,
  `firing_duration_s`, `thrusters`
- **model** — `plume_model`, `plume_model_parameters`
- **loads** — `pressure_force`, `shear_force`, `force`, `force_magnitude`,
  `moment_reference_point`, `pressure_moment`, `shear_moment`, `moment`,
  `moment_magnitude`, `center_of_pressure`, `center_of_pressure_status`,
  `residual_couple`, `pressure_weighted_centroid`
- **surface fields** — `max_pressure`, `max_shear_stress`, `max_heat_flux`,
  `total_heat_load`, `affected_area`, `struck_faces`
- **coefficients** — `coefficients`, `coefficients_available`
- **artifacts and provenance** — `vtk_path`, `jfh_path`, `config_path`,
  `case_dir`, `code_version` (git commit when available), `generated_at`

Two machine-readable artifacts are written, in formats the repository already
uses (no Parquet, no new dependency):

- **CSV** (`StudyResults.write_csv`) — one flat row per record; vectors are
  expanded to `<name>_x/_y/_z` columns and each coefficient gets its own
  `coeff_<name>` column, so the file is directly plottable;
- **JSON** (`StudyResults.write_metadata`) — `schema`, study-level
  `provenance` (including the plume model, mesh size, component list, code
  version and known limitations) and the nested per-case records. This is the
  document an externally generated dataset is later transformed into for
  comparison.

---

## VTK outputs

Per-face data are never reduced away. With `output.vtk.enabled` (the default)
the standard pipeline writer produces one `.vtu` per firing, laid out
according to the sweep mode:

- `per_case`: `<output_dir>/cases/<case_id>/results/strikes/firing-<i>.vtu`
- `single_jfh`: `<output_dir>/results/strikes/firing-<i>.vtu` — one numbered
  series over the whole sweep, which ParaView opens as a time sequence

Both carry the pipeline's own cell fields:

```
strikes, cum_strikes, pressures, max_pressures, shear_stress, max_shears,
heat_flux_rate, heat_flux_load, cum_heat_flux_load
```

Every case result advertises its own `vtk_path`, and that path is also in the
CSV and JSON summaries, so artifacts are discoverable from the returned
results. Set `output.vtk.enabled: false` for a fast, artifact-free run (the
numbers are identical; only the files are skipped).

---

## Optional plots

Plot generation is entirely optional — no automated test requires graphical
output — and matplotlib's non-interactive `Agg` backend is pinned when
`pyrpod.mdao.study_plots` is imported (lazily, only when plots are asked
for). Enable with `output.plots.enabled: true` or call `study.plot()`:

```
force_vs_angle.png        moment_vs_angle.png      heat_flux_vs_angle.png
force_vs_distance.png     moment_vs_distance.png   center_of_pressure.png
reference_comparison.png  (when a comparison report exists)
```

---

## External reference data

`pyrpod/mdao/reference_data.py` compares study results against independently
generated data **without knowing where it came from**. A reference record is
just named quantities attached to matching keys; DSMC, an analytical
solution, an experiment and another code are all handled identically, and no
producer-specific importer exists.

Supported formats:

- **CSV** — one row per record. `case_id`, `component`, `plate_angle_deg` and
  `source_distance` are recognized as keys; every other numeric column is a
  quantity, and `<name>_x/_y/_z` triplets are folded into vectors. This is
  exactly the layout `StudyResults.write_csv` emits, so an external producer
  can be transformed into it column-for-column.
- **JSON / YAML**:

```yaml
label: DSMC run 12
source: /runs/dsmc-12
units: {force: N}
records:
  - key: {plate_angle_deg: -40.0, source_distance: 2.0, component: plate}
    quantities:
      force: [1.117, 0.0, -2.376]
      max_pressure: 0.34
      center_of_pressure: [0.031, 0.0, 0.015]
```

```python
from pyrpod.mdao.reference_data import load_reference_dataset

report = study.compare(load_reference_dataset('dsmc_run_12.json'))
print(report.max_relative_error('force'))
report.write_csv('comparison.csv')
```

Metrics are applied only where they are mathematically meaningful: absolute
error, relative error (`None` when the reference is zero, never infinity),
normalized RMSE (by reference range or mean), peak-value error,
integrated-load error (compared as vectors, so a load of the right magnitude
pointing the wrong way is an error) and center-of-pressure displacement. A
quantity the reference supplies but the result does not is reported as
`missing_candidate`; a case with no matching record is listed in
`unmatched_cases`. Nothing is fabricated or defaulted.

For the flat-plate tests the independent reference is the exact Cai 2016
solution (`pyrpod/plume/CaiImpingement2016.py`, Eq. 15 quadrature),
converted to dimensional loads with the case's own normalization — PyRPOD's
own output is never used as the validation baseline. Measured agreement over
the sweep: integrated normal load within ~1.5% mean / 2.2% max, component
heat load within ~15%, matching the documented accuracy of the Maxwellian
engineering chain against the exact collisionless solution.

### Adding DSMC results later

Nothing DSMC-specific is needed. Transform the external results into either
supported format, keyed by `plate_angle_deg` / `source_distance` /
`component` (or `case_id`, which the study metadata records), then call
`study.compare(...)` or point `reference.path` at the file. Because the
result schema records the mesh, coordinate system, units, source pose, model
and configuration provenance, the transformation can be done from the stored
metadata without rerunning PyRPOD.

---

## Adding another validation geometry

1. Create (or reuse) a PyRPOD case directory with its `config.ini`, target
   STL, TCF and TDF — the study layer adds nothing here.
2. Add a study YAML next to it under `study/`, setting `target.reference_point`
   to the geometry's reference (plate center, cylinder centroid, ...) and
   `target.normal` / `target.tangent` to the plane the source should be swept
   in.
3. Break the target into components if the loads should be reported
   separately, using `face_indices` or a `bounds` box on face centroids.
4. Supply `loads.normalization` only if valid reference values exist for that
   geometry; otherwise leave it out and the coefficients are reported as
   unavailable.
5. Point `reference.path` at external data whenever it exists.

`case/plume/plume_cylinder_sweep/study/cylinder_baseline.yaml` is a worked
example on a curved, closed target. Its results are a pipeline smoke case,
not a validated physical answer — no cylinder reference data exists yet — and
it deliberately supplies no normalization inputs.

---

## Known limitations

- **No shadowing or occlusion.** Face selection is the existing pipeline
  behavior: a face is struck when it lies inside the plume wedge/radius and
  its normal faces the source. Plume shadowing, occlusion by intervening
  geometry, self-shadowing of concave or closed targets and back-facing
  surfaces are **not** modeled, so a face hidden behind other geometry still
  receives load. This branch changes nothing about that; results on closed
  targets must be read with it in mind.
- **Edge-on poses are degenerate.** At exactly `+/-90 deg` the facing test's
  dot product is zero in exact arithmetic, so float32 STL normals decide
  strike membership arbitrarily. The plate-averaged load there is ~zero, but
  individual near-nozzle faces can carry large grazing values. The committed
  sweep keeps those poses; the symmetry checks exclude them.
- **Cumulative fields mean different things per mode.** In `single_jfh` mode
  they are a genuine sweep envelope; in `per_case` mode they only restate the
  case they belong to, since each case's history starts fresh. Read
  `max_pressures` / `cum_strikes` in a per-case VTK accordingly.
- **Coefficients require explicit normalization.** By design: nothing is
  inferred from the geometry.
- **One thruster, one plume model.** The study workflow prescribes a single
  firing source and `SimplifiedGasKinetics`. Multi-thruster and multi-group
  behavior is untouched elsewhere in PyRPOD but is not exercised here.
- **Cylinder validation is not quantitative yet.** The architecture and the
  comparison interface are ready for it; the reference data is not.
- **Heat loading is instantaneous/peak only.** Time-integrated heat dose and
  impulse are deliberately out of scope on this branch.

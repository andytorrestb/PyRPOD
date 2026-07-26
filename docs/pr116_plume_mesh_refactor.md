# Plume-mesh transformation architecture (PR #116)

This document describes the consolidation of plume-mesh transformation and
composition logic. The change is a **behavior-preserving maintainability
refactor**: it removes duplicated transform/compose code without altering any
numerical result, coordinate-frame convention, transformation order, output
path, filename, artifact count, logging, or failure behavior.

## Ownership of plume-placement logic

`VisitingVehicle.transform_plume_mesh` is the **canonical owner** of plume
placement geometry. It replaces the per-thruster transform blocks that were
duplicated across `PlumeStrikeEstimationStudy.graph_jfh`,
`PlumeStrikeEstimationStudy.visualize_sweep`, and
`PlumeStrikeEstimationStudy.graph_jfh_thruster_check`.

The behavior lives on `VisitingVehicle` (not on a standalone
`PlumeMeshUtils.py` module of free functions that inspect a vehicle from the
outside) because the placement is intrinsically a property of the visiting
vehicle's thruster and cluster configuration. Keeping it a method preserves
encapsulation, lets it reuse the vehicle's own `thruster_data`/`cluster_data`,
and keeps callers passing DCMs exactly as stored while transposition/sequencing
is handled in one place.

### Transformation order (exact, do not "correct")

`transform_plume_mesh(thruster_id, plumeMesh, vv_orientation=None,
vv_position=None)` applies, in this exact order (rotations first, about the
origin, before any translation away from the rotation axes):

1. **Thruster DCM (transposed).** Always applied.
2. **Visiting-vehicle DCM (transposed).** Applied only when `vv_orientation` is
   supplied.
3. **Visiting-vehicle position** in the target/JFH frame. Applied only when
   `vv_position` is supplied.
4. **Cluster exit offset** (legacy semantics). Applied only in complete
   vehicle/JFH placement (a pose is supplied) with clusters enabled.
5. **Thruster exit offset.** Always applied.

Two placement modes are represented explicitly in the API:

- **Local thruster placement** — `vv_orientation`/`vv_position` both omitted;
  applies steps 1 and 5 only. Used by `check_thruster_configuration` and
  `LogisticsModule.plot_thruster_group`. No cluster offset is applied.
- **Complete vehicle/JFH placement** — pose supplied; applies steps 1–5. Used
  by `graph_jfh` and `visualize_sweep`.

This explicit split is what prevents the double-application hazard that a naive
consolidation risks: `graph_jfh_thruster_check` does local placement (thruster
DCM + exit once each) and never re-applies those as a vehicle-level pose.

### Coordinate-frame assumptions

- **Thruster DCM**: maps the thruster frame → visiting-vehicle frame. Stored as
  a 3×3 matrix; passed to `transform_plume_mesh` untransposed and transposed
  internally.
- **JFH/vehicle DCM**: maps the visiting-vehicle frame → target-vehicle frame.
  Stored as a 3×3 matrix; passed untransposed and transposed internally.
- **Position vectors**: length-3, in the target/JFH frame.

These descriptions are documentation for readers and test authors. The refactor
deliberately does **not** re-derive or "fix" the transform chain; the legacy
order is the required behavior.

### Mutation semantics

`transform_plume_mesh` mutates `plumeMesh` **in place** (numpy-stl's
`rotate_using_matrix`/`translate` mutate) and returns the same object for
convenience. Callers load a fresh plume mesh per thruster, so in-place mutation
is safe and matches the historical behavior. The method was **not** changed to
copy the mesh.

### Exception behavior

- Unknown `thruster_id` → `KeyError` (via the `thruster_data` lookup).
- Clusters enabled but the required cluster is missing → `KeyError` with a
  descriptive message. Cluster offsets are **never silently omitted**.
- Malformed shapes (thruster DCM not 3×3, `vv_orientation` not 3×3,
  `vv_position` not length-3) → `ValueError`. This fires only on malformed
  configuration data; valid legacy inputs (always 3×3 DCMs and length-3
  vectors) are unaffected.

## Cluster-ID parsing

The cluster a thruster belongs to is derived from the thruster naming
convention by `VisitingVehicle._cluster_id_for_thruster`, which parses the
leading `P<n>` prefix (regex `^(P\d+)`):

- Single-digit ids behave **identically** to the legacy first-two-character
  parse (`thruster_id[0] + thruster_id[1]`): `P1T2 → P1`.
- Multi-digit cluster ids now resolve correctly: `P10T1 → P10` (legacy produced
  the wrong `P1`).

This depends on the naming convention (`P<cluster><T<thruster>>`). Making the
cluster association explicit configuration metadata rather than name-derived is
a deferred follow-up (see below).

## JFH thruster-index mapping

The JFH references thrusters by 1-based numeric index into the thruster
configuration order. `VisitingVehicle.get_thruster_id_map()` builds and
**caches** the `{'1': first_thruster_id, ...}` mapping, preserving the insertion
order of `thruster_data` (the legacy ordering rule). `get_thruster_id(index)`
returns the canonical thruster id (a string), so callers no longer index a
one-element `['name']` list per lookup.

Lifecycle:

- Built lazily on first use and cached on the instance.
- **Invalidated** by `set_thruster_config()` whenever the configuration is
  loaded or replaced.
- An index outside the configured range raises `KeyError`.

The mapping lives on `VisitingVehicle`, **not** on the base `Vehicle` class.
The numerical strike path (`pyrpod/plume/PlumeStrikeCalculator.py`) keeps its
own module-level `_build_thruster_link` that operates on a plain, picklable
`thruster_data` dict, because that path ships serializable inputs to worker
processes and never has a `VisitingVehicle` in hand.

## Generic mesh composition

`pyrpod.util.stl.stl.compose_meshes(meshes)` is the canonical home for generic
mesh concatenation (it lives in the STL utility layer, not on
`VisitingVehicle`). It:

- accepts an ordered collection of `mesh.Mesh` objects and preserves order;
- concatenates all face data in one `np.concatenate`;
- returns the single input object unchanged when exactly one mesh is supplied
  (matching legacy incremental composition);
- raises `ValueError` for zero meshes or any `None` element (never silently
  returns `None` nor drops a component);
- does not mutate the input meshes.

The study composes each firing as `[VVmesh] (+ active_cones) (+ active_clusters
when clusters are enabled)`. When there are no active plume cones the VV body is
emitted unchanged and cluster geometry is intentionally omitted — exactly the
legacy behavior. Optional per-firing components (`active_cones`,
`active_clusters`) are initialized explicitly rather than probed with
`'active_clusters' in locals()`.

## The `transform_mesh` name collision

`pyrpod/util/stl/stl.py` previously defined `transform_mesh` **twice**: a
reusable in-memory API `transform_mesh(mesh_obj, rotation_matrix=...,
translation_vector=..., scale_factor=...)` and, later in the file, a file-based
CLI helper `transform_mesh(input_file, output_file, scale, translate)`. The
second definition shadowed the first on import, so `from pyrpod.util.stl.stl
import transform_mesh` bound the file-based form — and any caller passing a mesh
object plus `rotation_matrix`/`scale_factor` (e.g. `graph_jfh_thruster_check`)
would have raised `TypeError`.

Resolution (smallest behavior-preserving change):

- The reusable mesh-object API keeps the name `transform_mesh`.
- The CLI/file-based helper is renamed `_transform_stl_file_cli` and its only
  caller — the `__main__` block — is updated. Scaling/translation/save/print
  behavior is unchanged.

## Why `PlumeStudyExport` stays in `pyrpod/rpod`

`PlumeStudyExport` is the study's file/figure export service and is
constructed and used exclusively by `PlumeStrikeEstimationStudy` (in
`pyrpod/rpod`). It carries current-master behavior for directory creation and
optional-visualization export. Moving it (or introducing a new plume-export
package) is out of scope and would be churn without behavioral benefit, so it
remains in `pyrpod/rpod`.

## Why the old `PlumeMeshUtils.py` approach was not retained

The stale PR added a separate `PlumeMeshUtils.py` of free functions that took a
vehicle object and inspected it externally, and it risked double-applying the
thruster DCM/exit in `graph_jfh_thruster_check` by passing them as both
vehicle-level pose inputs and letting the helper re-apply them. Current master
had since diverged (operational logging, asset-path resolution, fail-fast
validation, export error handling). Rather than rebasing that stale module, the
intent was reimplemented against current master by placing the behavior on
`VisitingVehicle` (better encapsulation, no external vehicle inspection) and
representing local-vs-complete placement explicitly to eliminate the
double-apply hazard.

## Deferred follow-ups (not created here)

- Replace naming-convention-based cluster association (`P<n>` prefix) with
  explicit configuration metadata on each thruster.
- Formally document/define the JFH thruster ordering rule (currently the
  insertion order of the thruster configuration).
- Broader cleanup of the legacy visualization methods (dead comments, the
  fixed-path RCS sanity check `graph_jfh_thruster_check`).
- Decide whether mesh transformations should eventually operate on immutable
  copies instead of mutating in place.
- Stronger coordinate-frame types/data structures (typed DCMs/positions).
- Remove one-element lists from thruster configuration fields (`name`, `exit`).

"""
YAML study configuration for prescribed plume/target validation sweeps.

A study configuration is a thin, declarative layer ON TOP OF an existing
PyRPOD case directory: the case's ``config.ini`` keeps owning the vehicle,
thruster, plume-model and target assets (so every existing case and public
API keeps working unchanged), while the YAML file adds only what a trade
study needs and a ``config.ini`` cannot express -- swept plate angles and
source distances, prescribed firing counts, the moment reference point,
coefficient normalization, and output/plot/reference settings.

The schema is deliberately explicit. Nothing is inferred that the caller
did not write down:

* the plume model is recorded by name and must be ``SimplifiedGasKinetics``
  (this branch adds no plume-model registry);
* coefficients are computed only when every normalization input is present
  (see :class:`Normalization`); otherwise they are reported as unavailable;
* ``n_firings`` means the exact number of entries written to the Jet Firing
  History -- a mismatch against an explicitly prescribed firing list is an
  error, never a silent truncation.

Example
-------
>>> cfg = StudyConfig.from_yaml('case/.../flat_plate_baseline.yaml')
>>> cfg.study_name
'cai2016_flat_plate_baseline'
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray

from pyrpod.rpod.approach_maneuvers import (
    validate_n_firings as _validate_n_firings,
)

#: The single plume model this study workflow supports (hard scope constraint).
SUPPORTED_PLUME_MODEL = "SimplifiedGasKinetics"

#: Default unit labels carried into the result metadata. PyRPOD works in SI
#: throughout; recording them makes an exported result self-describing.
DEFAULT_UNITS: dict[str, str] = {
    "length": "m",
    "area": "m^2",
    "force": "N",
    "moment": "N*m",
    "pressure": "Pa",
    "shear_stress": "Pa",
    "heat_flux": "W/m^2",
    "time": "s",
    "angle": "deg",
}


class StudyConfigError(ValueError):
    """Raised when a study configuration is missing or internally inconsistent."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise StudyConfigError(
            f"missing required key {key!r} in {context}")
    return mapping[key]


def _as_vector(value: Any, context: str) -> NDArray[np.float64]:
    """Parse a 3-element vector, failing loudly on anything else."""
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise StudyConfigError(
            f"{context}: expected three numbers, got {value!r}") from exc
    if vector.size != 3:
        raise StudyConfigError(
            f"{context}: expected three numbers, got {value!r}")
    return vector


def _as_float_list(value: Any, context: str) -> list[float]:
    if value is None or isinstance(value, (str, bytes)) or not isinstance(
            value, Sequence):
        raise StudyConfigError(f"{context}: expected a list of numbers, "
                               f"got {value!r}")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise StudyConfigError(f"{context}: expected a list of numbers, "
                               f"got {value!r}") from exc


@dataclass(frozen=True)
class ComponentSpec:
    """One target component: a named subset of the target mesh's faces.

    A component is selected either by ``face_indices`` (explicit) or by
    ``bounds`` (an axis-aligned box on face centroids, in the case's global
    frame). ``selector: all`` -- the default -- takes the whole mesh, which
    is the right answer for the single-plate and single-cylinder targets this
    branch ships. Nothing here assumes a flat plate.

    Attributes
    ----------
    name : str
        Component identifier, reported with every result row.
    selector : str
        ``'all'``, ``'face_indices'`` or ``'bounds'``.
    face_indices : tuple of int
        Explicit face indices when ``selector == 'face_indices'``.
    bounds : dict
        ``{'min': [x, y, z], 'max': [x, y, z]}`` when ``selector == 'bounds'``.
    """

    name: str
    selector: str = "all"
    face_indices: tuple[int, ...] = ()
    bounds: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentSpec":
        name = str(_require(data, "name", "target.components entry"))
        if "face_indices" in data and data["face_indices"] is not None:
            indices = tuple(int(i) for i in data["face_indices"])
            if not indices:
                raise StudyConfigError(
                    f"target component {name!r}: face_indices is empty")
            return cls(name=name, selector="face_indices", face_indices=indices)
        if "bounds" in data and data["bounds"] is not None:
            raw = data["bounds"]
            lo = _as_vector(_require(raw, "min", f"component {name!r} bounds"),
                            f"component {name!r} bounds.min")
            hi = _as_vector(_require(raw, "max", f"component {name!r} bounds"),
                            f"component {name!r} bounds.max")
            if np.any(hi < lo):
                raise StudyConfigError(
                    f"target component {name!r}: bounds.max must be >= "
                    "bounds.min componentwise")
            bounds = {"min": (float(lo[0]), float(lo[1]), float(lo[2])),
                      "max": (float(hi[0]), float(hi[1]), float(hi[2]))}
            return cls(name=name, selector="bounds", bounds=bounds)
        selector = str(data.get("selector", "all"))
        if selector != "all":
            raise StudyConfigError(
                f"target component {name!r}: selector {selector!r} needs "
                "'face_indices' or 'bounds' to be supplied")
        return cls(name=name, selector="all")


@dataclass(frozen=True)
class TargetSpec:
    """Target geometry description and its component breakdown.

    ``reference_point`` is the geometric reference the sweep is built about
    (the plate center, the cylinder axis midpoint, ...). ``normal`` and
    ``tangent`` define the plane the approach angle is swept in: angle 0
    places the plume source on ``normal``, positive angles rotate it toward
    ``tangent``. They are geometry properties, not physics, so a curved
    target simply supplies the axes its sweep should use.
    """

    geometry_id: str
    reference_point: NDArray[np.float64]
    normal: NDArray[np.float64]
    tangent: NDArray[np.float64]
    components: tuple[ComponentSpec, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any],
                     default_geometry_id: str) -> "TargetSpec":
        normal = _as_vector(data.get("normal", [0.0, 0.0, 1.0]),
                            "target.normal")
        tangent = _as_vector(data.get("tangent", [1.0, 0.0, 0.0]),
                             "target.tangent")
        for label, vec in (("normal", normal), ("tangent", tangent)):
            if not np.isfinite(vec).all() or np.linalg.norm(vec) == 0.0:
                raise StudyConfigError(
                    f"target.{label} must be a finite non-zero vector")
        normal = normal / np.linalg.norm(normal)
        tangent = tangent / np.linalg.norm(tangent)
        if abs(float(normal @ tangent)) > 1e-8:
            raise StudyConfigError(
                "target.normal and target.tangent must be orthogonal "
                f"(dot product {float(normal @ tangent):.3g})")

        raw_components = data.get("components")
        if raw_components is None:
            components: tuple[ComponentSpec, ...] = (
                ComponentSpec(name="target"),)
        else:
            if not isinstance(raw_components, Sequence) or not raw_components:
                raise StudyConfigError(
                    "target.components must be a non-empty list")
            components = tuple(ComponentSpec.from_mapping(entry)
                               for entry in raw_components)
            names = [component.name for component in components]
            if len(set(names)) != len(names):
                raise StudyConfigError(
                    f"target.components names must be unique, got {names}")

        return cls(
            geometry_id=str(data.get("geometry_id") or default_geometry_id),
            reference_point=_as_vector(data.get("reference_point",
                                                [0.0, 0.0, 0.0]),
                                       "target.reference_point"),
            normal=normal,
            tangent=tangent,
            components=components,
        )


@dataclass(frozen=True)
class PrescribedFiringSpec:
    """One explicitly prescribed firing: pose, active thrusters, duration."""

    position: NDArray[np.float64]
    dcm: NDArray[np.float64]
    thrusters: tuple[int, ...]
    duration_s: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], index: int,
                     default_thrusters: Sequence[int],
                     default_duration_s: float) -> "PrescribedFiringSpec":
        context = f"firings[{index}]"
        position = _as_vector(_require(data, "position", context),
                              f"{context}.position")
        dcm = np.asarray(_require(data, "dcm", context), dtype=float)
        if dcm.shape != (3, 3):
            raise StudyConfigError(
                f"{context}.dcm must be a 3x3 matrix, got shape {dcm.shape}")
        thrusters = tuple(int(t) for t in
                          data.get("thrusters", default_thrusters))
        if not thrusters:
            raise StudyConfigError(f"{context}.thrusters must not be empty")
        duration = float(data.get("duration_s", default_duration_s))
        if duration <= 0.0:
            raise StudyConfigError(
                f"{context}.duration_s must be positive, got {duration}")
        return cls(position=position, dcm=dcm, thrusters=thrusters,
                   duration_s=duration)


#: Sweep execution modes. ``per_case`` writes one Jet Firing History per
#: angle-distance combination (isolated poses); ``single_jfh`` writes ONE
#: history spanning the whole sweep, so every pose's strikes come from a
#: single pipeline run and the cumulative fields form a sweep envelope.
SWEEP_MODES = ("per_case", "single_jfh")


@dataclass(frozen=True)
class SweepSpec:
    """Parameter sweep and firing-count definition.

    Attributes
    ----------
    plate_angles_deg : tuple of float
        Approach angles swept in the target's (normal, tangent) plane;
        0 deg is head-on along the target normal.
    source_distances : tuple of float
        Plume-source distances from ``TargetSpec.reference_point``, in the
        case's length units.
    n_firings : int
        Number of Jet Firing History entries contributed by EACH pose. In
        ``per_case`` mode that is the exact length of every case's history;
        in ``single_jfh`` mode the one shared history holds exactly
        ``len(poses) * n_firings`` entries. Either way the count is exact.
    mode : str
        ``'per_case'`` (default) or ``'single_jfh'``; see :data:`SWEEP_MODES`.
    firing_duration_s : float
        Firing time recorded for each JFH entry.
    thrusters : tuple of int
        JFH thruster indices active in every generated firing.
    firings : tuple of PrescribedFiringSpec
        Explicitly prescribed firings, replacing the generated poses. In
        ``per_case`` mode their count must equal ``n_firings``; in
        ``single_jfh`` mode it must equal ``len(poses) * n_firings``, and
        they are assigned to the poses in sweep order.
    """

    plate_angles_deg: tuple[float, ...]
    source_distances: tuple[float, ...]
    n_firings: int
    firing_duration_s: float
    thrusters: tuple[int, ...]
    firings: tuple[PrescribedFiringSpec, ...] = ()
    mode: str = "per_case"

    @property
    def poses(self) -> tuple[tuple[float, float], ...]:
        """(plate angle, source distance) pairs in execution order.

        Distance-major, matching the committed sweep-JFH generators: all
        angles at the first distance, then all angles at the next.
        """
        return tuple((angle, distance)
                     for distance in self.source_distances
                     for angle in self.plate_angles_deg)

    @property
    def total_firings(self) -> int:
        """Total JFH entries across the whole sweep."""
        return len(self.poses) * self.n_firings

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SweepSpec":
        angles = tuple(_as_float_list(
            data.get("plate_angles_deg", [0.0]), "sweep.plate_angles_deg"))
        distances = tuple(_as_float_list(
            data.get("source_distances", []), "sweep.source_distances"))
        if not angles:
            raise StudyConfigError("sweep.plate_angles_deg must not be empty")
        if not distances:
            raise StudyConfigError("sweep.source_distances must not be empty")
        if any(d <= 0.0 for d in distances):
            raise StudyConfigError(
                "sweep.source_distances must all be positive")

        mode = str(data.get("mode", "per_case"))
        if mode not in SWEEP_MODES:
            raise StudyConfigError(
                f"sweep.mode must be one of {list(SWEEP_MODES)}, got {mode!r}")

        n_firings = validate_n_firings(data.get("n_firings", 1))
        duration = float(data.get("firing_duration_s", 1.0))
        if duration <= 0.0:
            raise StudyConfigError(
                f"sweep.firing_duration_s must be positive, got {duration}")
        thrusters = tuple(int(t) for t in data.get("thrusters", [1]))
        if not thrusters:
            raise StudyConfigError("sweep.thrusters must not be empty")

        raw_firings = data.get("firings")
        firings: tuple[PrescribedFiringSpec, ...] = ()
        if raw_firings:
            firings = tuple(
                PrescribedFiringSpec.from_mapping(entry, i, thrusters, duration)
                for i, entry in enumerate(raw_firings))
            n_poses = len(angles) * len(distances)
            expected = n_firings if mode == "per_case" else n_poses * n_firings
            if len(firings) != expected:
                where = ("per case" if mode == "per_case"
                         else f"across {n_poses} poses")
                raise StudyConfigError(
                    f"sweep.n_firings is {n_firings} ({expected} entries "
                    f"{where} in {mode!r} mode) but {len(firings)} explicit "
                    "firings were supplied; the counts must agree exactly")

        return cls(plate_angles_deg=angles, source_distances=distances,
                   n_firings=n_firings, firing_duration_s=duration,
                   thrusters=thrusters, firings=firings, mode=mode)


def validate_n_firings(value: Any) -> int:
    """Validate a requested firing count: a positive integer, nothing else.

    ``n_firings`` is the exact number of Jet Firing History entries a case
    writes, so zero, negative, fractional and non-numeric values are rejected
    rather than coerced. The rule itself lives with the JFH-generation code
    (:func:`pyrpod.rpod.approach_maneuvers.validate_n_firings`) so the
    prescribed and dynamics-driven paths cannot drift apart; only the raised
    error type is specialized here.
    """
    try:
        return _validate_n_firings(value)
    except ValueError as exc:
        raise StudyConfigError(str(exc)) from exc


@dataclass(frozen=True)
class Normalization:
    """Coefficient normalization inputs.

    Coefficients are computed only when the configuration supplies every
    value a given coefficient needs; nothing is invented or defaulted:

    * force coefficients need ``reference_area`` and ``dynamic_pressure``;
    * moment coefficients additionally need ``reference_length``;
    * surface-load coefficients (pressure/shear/heat flux) need
      ``dynamic_pressure`` and, for heat flux, ``reference_heat_flux``.
    """

    reference_area: float | None = None
    reference_length: float | None = None
    dynamic_pressure: float | None = None
    reference_heat_flux: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "Normalization":
        if not data:
            return cls()

        def optional(key: str) -> float | None:
            value = data.get(key)
            if value is None:
                return None
            number = float(value)
            if not np.isfinite(number) or number <= 0.0:
                raise StudyConfigError(
                    f"loads.normalization.{key} must be a positive finite "
                    f"number, got {value!r}")
            return number

        return cls(reference_area=optional("reference_area"),
                   reference_length=optional("reference_length"),
                   dynamic_pressure=optional("dynamic_pressure"),
                   reference_heat_flux=optional("reference_heat_flux"))

    @property
    def has_force_inputs(self) -> bool:
        return (self.reference_area is not None
                and self.dynamic_pressure is not None)

    @property
    def has_moment_inputs(self) -> bool:
        return self.has_force_inputs and self.reference_length is not None

    def to_dict(self) -> dict[str, float | None]:
        return {"reference_area": self.reference_area,
                "reference_length": self.reference_length,
                "dynamic_pressure": self.dynamic_pressure,
                "reference_heat_flux": self.reference_heat_flux}


@dataclass(frozen=True)
class LoadsSpec:
    """Surface-load integration settings."""

    moment_reference_point: NDArray[np.float64]
    normalization: Normalization

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "LoadsSpec":
        data = data or {}
        return cls(
            moment_reference_point=_as_vector(
                data.get("moment_reference_point", [0.0, 0.0, 0.0]),
                "loads.moment_reference_point"),
            normalization=Normalization.from_mapping(
                data.get("normalization")),
        )


@dataclass(frozen=True)
class OutputSpec:
    """Output artifact settings (VTK, machine-readable summary, plots)."""

    write_vtk: bool = True
    vtk_subdir: str = "vtk"
    summary_csv: str = "case_results.csv"
    summary_metadata: str = "study_metadata.json"
    write_plots: bool = False
    plots_subdir: str = "plots"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "OutputSpec":
        data = data or {}
        vtk = data.get("vtk") or {}
        summary = data.get("summary") or {}
        plots = data.get("plots") or {}
        return cls(
            write_vtk=bool(vtk.get("enabled", True)),
            vtk_subdir=str(vtk.get("subdir", "vtk")),
            summary_csv=str(summary.get("csv", "case_results.csv")),
            summary_metadata=str(summary.get("metadata",
                                             "study_metadata.json")),
            write_plots=bool(plots.get("enabled", False)),
            plots_subdir=str(plots.get("subdir", "plots")),
        )


@dataclass(frozen=True)
class ReferenceSpec:
    """Optional external reference-data location (see mdao.reference_data)."""

    path: str | None = None
    label: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ReferenceSpec":
        data = data or {}
        path = data.get("path")
        return cls(path=str(path) if path else None,
                   label=str(data["label"]) if data.get("label") else None)


@dataclass(frozen=True)
class StudyConfig:
    """A complete, validated study configuration.

    Attributes
    ----------
    study_name : str
        Identifier recorded with every result row.
    case_dir : str
        Existing PyRPOD case directory (owns ``config.ini`` and the STL / TCD
        / plume assets). Always ends with a path separator, as the rest of
        PyRPOD expects.
    output_dir : str
        Directory the study writes its artifacts to.
    plume_model : str
        Always ``'SimplifiedGasKinetics'`` in this branch.
    source_path : str
        Path the configuration was read from (configuration provenance).
    """

    study_name: str
    description: str
    case_dir: str
    output_dir: str
    target: TargetSpec
    sweep: SweepSpec
    loads: LoadsSpec
    output: OutputSpec
    reference: ReferenceSpec
    thruster_id: str | None = None
    plume_model: str = SUPPORTED_PLUME_MODEL
    plume_model_parameters: dict[str, Any] = field(default_factory=dict)
    coordinate_system: str = "case global frame"
    units: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UNITS))
    source_path: str = ""

    # ------------------------------------------------------------------ load
    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "StudyConfig":
        """Read and validate a YAML study configuration."""
        path = os.fspath(path)
        if not os.path.isfile(path):
            raise StudyConfigError(f"study configuration not found: {path!r}")
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, Mapping):
            raise StudyConfigError(
                f"study configuration {path!r} must be a YAML mapping")
        return cls.from_mapping(data, source_path=path)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any],
                     source_path: str = "") -> "StudyConfig":
        """Validate an already-parsed configuration mapping."""
        study = data.get("study") or {}
        if not isinstance(study, Mapping):
            raise StudyConfigError("'study' section must be a mapping")

        name = str(_require(study, "name", "study"))
        raw_case_dir = str(_require(study, "case_dir", "study"))
        base_dir = os.path.dirname(os.path.abspath(source_path)) if \
            source_path else os.getcwd()
        case_dir = _resolve_dir(raw_case_dir, base_dir)
        if not os.path.isdir(case_dir):
            raise StudyConfigError(
                f"study.case_dir does not exist: {case_dir!r}")
        if not os.path.isfile(os.path.join(case_dir, "config.ini")):
            raise StudyConfigError(
                f"study.case_dir {case_dir!r} has no config.ini; a study "
                "always runs on top of an existing PyRPOD case")
        case_dir = case_dir.rstrip("\\/") + os.sep

        raw_output_dir = str(study.get(
            "output_dir", os.path.join(case_dir, "results", "studies", name)))
        output_dir = _resolve_dir(raw_output_dir, base_dir)

        plume = data.get("plume_model") or {}
        model_name = str(plume.get("name", SUPPORTED_PLUME_MODEL))
        if model_name != SUPPORTED_PLUME_MODEL:
            raise StudyConfigError(
                f"plume_model.name must be {SUPPORTED_PLUME_MODEL!r} "
                f"(this study workflow supports exactly one model), got "
                f"{model_name!r}")
        model_parameters = dict(plume.get("parameters") or {})

        thruster = data.get("thruster") or {}
        thruster_id = thruster.get("id")

        target_data = data.get("target") or {}
        default_geometry = _default_geometry_id(case_dir)

        metadata = data.get("metadata") or {}
        units = dict(DEFAULT_UNITS)
        units.update({str(k): str(v)
                      for k, v in (metadata.get("units") or {}).items()})

        return cls(
            study_name=name,
            description=str(study.get("description", "")),
            case_dir=case_dir,
            output_dir=output_dir,
            target=TargetSpec.from_mapping(target_data, default_geometry),
            sweep=SweepSpec.from_mapping(data.get("sweep") or {}),
            loads=LoadsSpec.from_mapping(data.get("loads")),
            output=OutputSpec.from_mapping(data.get("output")),
            reference=ReferenceSpec.from_mapping(data.get("reference")),
            thruster_id=str(thruster_id) if thruster_id else None,
            plume_model=model_name,
            plume_model_parameters=model_parameters,
            coordinate_system=str(metadata.get("coordinate_system",
                                               "case global frame")),
            units=units,
            source_path=os.path.abspath(source_path) if source_path else "",
        )

    # ------------------------------------------------------------- accessors
    @property
    def n_cases(self) -> int:
        """Number of angle x distance cases in this study."""
        return (len(self.sweep.plate_angles_deg)
                * len(self.sweep.source_distances))

    def with_output_dir(self, output_dir: str) -> "StudyConfig":
        """Copy of this configuration writing to a different directory."""
        return replace(self, output_dir=os.path.abspath(output_dir))

    def provenance(self) -> dict[str, Any]:
        """Configuration provenance recorded in the study metadata."""
        return {
            "study_name": self.study_name,
            "config_path": self.source_path,
            "case_dir": os.path.abspath(self.case_dir),
            "output_dir": os.path.abspath(self.output_dir),
            "plume_model": self.plume_model,
            "plume_model_parameters": dict(self.plume_model_parameters),
            "coordinate_system": self.coordinate_system,
            "units": dict(self.units),
        }


def _resolve_dir(path: str, base_dir: str) -> str:
    """Resolve a configured directory relative to the config file location."""
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def _default_geometry_id(case_dir: str) -> str:
    """Target geometry id taken from the case's own ``[tv] stl`` entry."""
    import configparser

    config = configparser.ConfigParser()
    config.read(os.path.join(case_dir, "config.ini"))
    if config.has_option("tv", "stl"):
        return str(config["tv"]["stl"])
    return "unknown"

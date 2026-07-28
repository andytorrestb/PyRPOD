"""
Generic external reference-data comparison.

This module compares PyRPOD study results against INDEPENDENTLY GENERATED
data without knowing where that data came from. A reference record is just a
set of named quantities attached to matching keys (component, plate angle,
source distance, or an explicit case id); whether those numbers came from a
DSMC solver, an analytical solution, a wind-tunnel test or another numerical
method is irrelevant to everything here, and no importer for any particular
producer exists in this module.

Supported reference formats -- all already used by the repository, no new
dependency:

* **CSV** -- one row per record. Key columns (``case_id``, ``component``,
  ``plate_angle_deg``, ``source_distance``) are recognized by name; every
  other numeric column becomes a quantity. Columns named ``<name>_x``,
  ``<name>_y`` and ``<name>_z`` are assembled into the vector quantity
  ``<name>``. This is exactly the layout
  :meth:`pyrpod.mdao.study_results.StudyResults.write_csv` emits, so an
  external producer can be transformed into it column-for-column.
* **JSON / YAML** -- ``{'label': ..., 'source': ..., 'units': {...},
  'records': [{'key': {...}, 'quantities': {...}}, ...]}``.

Metrics
-------
Only mathematically applicable metrics are computed for a given quantity:
absolute error and relative error for scalars and vectors, normalized RMSE
for sampled arrays, peak-value error for arrays, integrated-load error for
the force/moment resultants, and center-of-pressure displacement for the
center of pressure. A quantity absent from the reference is reported as
``missing_reference`` -- never defaulted, never fabricated.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pyrpod.mdao.study_results import CaseResult

__all__ = [
    "ComparisonReport",
    "QuantityComparison",
    "ReferenceDataset",
    "ReferenceRecord",
    "absolute_error",
    "center_of_pressure_displacement",
    "compare_case",
    "compare_results",
    "integrated_load_error",
    "load_reference_dataset",
    "normalized_rmse",
    "peak_error",
    "relative_error",
]

#: Key columns recognized when a reference record is matched to a case.
KEY_FIELDS = ("case_id", "component", "plate_angle_deg", "source_distance")

#: Tolerance used when matching numeric keys (angles in degrees, distances in
#: the case's length unit).
KEY_TOLERANCE = 1e-6


# --------------------------------------------------------------------- metrics
def _as_array(value: Any) -> NDArray[np.float64]:
    return np.asarray(value, dtype=float).reshape(-1)


def absolute_error(candidate: Any, reference: Any) -> float:
    """|candidate - reference|; the Euclidean norm for vector quantities."""
    diff = _as_array(candidate) - _as_array(reference)
    return float(np.linalg.norm(diff))


def relative_error(candidate: Any, reference: Any) -> float | None:
    """Absolute error normalized by the reference magnitude.

    Returns None when the reference magnitude is zero, where a relative
    error is undefined rather than infinite.
    """
    scale = float(np.linalg.norm(_as_array(reference)))
    if scale == 0.0:
        return None
    return absolute_error(candidate, reference) / scale


def normalized_rmse(candidate: Any, reference: Any,
                    norm: str = "range") -> float | None:
    """RMSE of a sampled array normalized by the reference range or mean.

    ``norm='range'`` (default) divides by ``max(ref) - min(ref)``; ``'mean'``
    divides by ``|mean(ref)|``. Returns None when the chosen normalizer is
    zero, or when the arrays have different lengths (nothing is resampled or
    interpolated here).
    """
    values = _as_array(candidate)
    target = _as_array(reference)
    if values.size != target.size or values.size == 0:
        return None
    rmse = float(np.sqrt(np.mean((values - target) ** 2)))
    if norm == "mean":
        scale = abs(float(np.mean(target)))
    else:
        scale = float(np.max(target) - np.min(target))
    if scale == 0.0:
        return None
    return rmse / scale


def peak_error(candidate: Any, reference: Any) -> tuple[float, float | None]:
    """Absolute and relative error of the peak (maximum) sampled value."""
    peak_candidate = float(np.max(_as_array(candidate)))
    peak_reference = float(np.max(_as_array(reference)))
    absolute = abs(peak_candidate - peak_reference)
    if peak_reference == 0.0:
        return absolute, None
    return absolute, absolute / abs(peak_reference)


def integrated_load_error(candidate: Any, reference: Any
                          ) -> tuple[float, float | None]:
    """Absolute and relative error of an integrated load (force or moment).

    Vector loads are compared as vectors (the error is the norm of the
    difference, not the difference of the norms), so a load of the right
    magnitude pointing the wrong way is reported as an error.
    """
    absolute = absolute_error(candidate, reference)
    return absolute, relative_error(candidate, reference)


def center_of_pressure_displacement(candidate: Any, reference: Any,
                                    reference_length: float | None = None
                                    ) -> tuple[float, float | None]:
    """Distance between two centers of pressure, optionally normalized."""
    distance = absolute_error(candidate, reference)
    if reference_length in (None, 0.0):
        return distance, None
    return distance, distance / float(reference_length)  # type: ignore[arg-type]


# ------------------------------------------------------------------ dataset
@dataclass(frozen=True)
class ReferenceRecord:
    """One reference data point: matching keys plus named quantities.

    ``quantities`` values may be scalars, vectors (lists of three numbers) or
    sampled arrays; the comparison layer selects metrics accordingly.
    """

    key: dict[str, Any]
    quantities: dict[str, Any]

    def matches(self, result: CaseResult) -> bool:
        """Whether this record describes the given case result."""
        for name, value in self.key.items():
            if value is None:
                continue
            actual = getattr(result, name, None)
            if actual is None:
                return False
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isclose(float(actual), float(value),
                                    abs_tol=KEY_TOLERANCE, rel_tol=1e-9):
                    return False
            elif str(actual) != str(value):
                return False
        return True


@dataclass
class ReferenceDataset:
    """A set of reference records with provenance, from any producer.

    Attributes
    ----------
    label : str
        Human-readable name of the dataset, reported with every comparison
        (e.g. ``'Cai 2016 exact'``, ``'DSMC run 12'``). The comparison code
        never behaves differently based on this string.
    source : str
        Where the data came from (path, DOI, run id).
    units : dict
        Unit labels declared by the producer, recorded for traceability.
    records : list of ReferenceRecord
    """

    label: str = "reference"
    source: str = ""
    units: dict[str, str] = field(default_factory=dict)
    records: list[ReferenceRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def match(self, result: CaseResult) -> ReferenceRecord | None:
        """First record matching the given case result, or None."""
        for record in self.records:
            if record.matches(result):
                return record
        return None

    def quantity_names(self) -> list[str]:
        names: list[str] = []
        for record in self.records:
            for name in record.quantities:
                if name not in names:
                    names.append(name)
        return names


def load_reference_dataset(path: str | os.PathLike[str],
                           label: str | None = None) -> ReferenceDataset:
    """Load a reference dataset from CSV, JSON or YAML (by file extension)."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"reference data not found: {path!r}")
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        dataset = _load_csv(path)
    elif suffix in (".json",):
        dataset = _load_structured(json.loads(
            _read_text(path)), path)
    elif suffix in (".yaml", ".yml"):
        import yaml

        dataset = _load_structured(yaml.safe_load(_read_text(path)), path)
    else:
        raise ValueError(
            f"unsupported reference-data format {suffix!r}; use .csv, .json "
            "or .yaml")
    if label:
        dataset.label = label
    return dataset


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_csv(path: str) -> ReferenceDataset:
    records: list[ReferenceRecord] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            key: dict[str, Any] = {}
            scalars: dict[str, float] = {}
            for column, text in raw.items():
                if column is None or text is None or text == "":
                    continue
                if column in KEY_FIELDS:
                    key[column] = _maybe_number(text)
                    continue
                number = _maybe_number(text)
                if isinstance(number, float):
                    scalars[column] = number
            records.append(ReferenceRecord(key=key,
                                           quantities=_assemble_vectors(scalars)))
    return ReferenceDataset(label=os.path.basename(path), source=path,
                            records=records)


def _load_structured(data: Any, path: str) -> ReferenceDataset:
    if not isinstance(data, Mapping):
        raise ValueError(
            f"reference data {path!r} must be a mapping with a 'records' list")
    raw_records = data.get("records")
    if not isinstance(raw_records, Sequence):
        raise ValueError(f"reference data {path!r} has no 'records' list")
    records = []
    for index, entry in enumerate(raw_records):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"reference data {path!r}: record {index} is not a mapping")
        records.append(ReferenceRecord(
            key=dict(entry.get("key") or {}),
            quantities=dict(entry.get("quantities") or {})))
    return ReferenceDataset(
        label=str(data.get("label", os.path.basename(path))),
        source=str(data.get("source", path)),
        units={str(k): str(v) for k, v in (data.get("units") or {}).items()},
        records=records)


def _maybe_number(text: str) -> Any:
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def _assemble_vectors(scalars: Mapping[str, float]) -> dict[str, Any]:
    """Fold ``<name>_x/_y/_z`` column triplets into vector quantities."""
    quantities: dict[str, Any] = {}
    consumed: set[str] = set()
    for name in scalars:
        if not name.endswith("_x"):
            continue
        stem = name[:-2]
        triplet = [f"{stem}_x", f"{stem}_y", f"{stem}_z"]
        if all(component in scalars for component in triplet):
            quantities[stem] = [scalars[component] for component in triplet]
            consumed.update(triplet)
    for name, value in scalars.items():
        if name not in consumed:
            quantities[name] = value
    return quantities


# --------------------------------------------------------------- comparison
@dataclass(frozen=True)
class QuantityComparison:
    """Comparison of one quantity for one case against one reference record."""

    case_id: str
    component: str
    quantity: str
    status: str
    candidate: Any = None
    reference: Any = None
    absolute_error: float | None = None
    relative_error: float | None = None
    normalized_rmse: float | None = None
    peak_absolute_error: float | None = None
    peak_relative_error: float | None = None
    displacement: float | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "component": self.component,
            "quantity": self.quantity,
            "status": self.status,
            "candidate": _scalar_or_text(self.candidate),
            "reference": _scalar_or_text(self.reference),
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
            "normalized_rmse": self.normalized_rmse,
            "peak_absolute_error": self.peak_absolute_error,
            "peak_relative_error": self.peak_relative_error,
            "displacement": self.displacement,
        }


def _scalar_or_text(value: Any) -> Any:
    if value is None:
        return ""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return float(array[0])
    return " ".join(f"{v:.9g}" for v in array)


def compare_case(result: CaseResult, record: ReferenceRecord,
                 reference_length: float | None = None,
                 ) -> list[QuantityComparison]:
    """Compare every quantity the reference record supplies for one case."""
    comparisons: list[QuantityComparison] = []
    for name, reference_value in record.quantities.items():
        candidate = result.quantity(name)
        if candidate is None:
            comparisons.append(QuantityComparison(
                case_id=result.case_id, component=result.component,
                quantity=name, status="missing_candidate",
                reference=reference_value))
            continue
        comparisons.append(_compare_quantity(
            result, name, candidate, reference_value, reference_length))
    return comparisons


def _compare_quantity(result: CaseResult, name: str, candidate: Any,
                      reference: Any, reference_length: float | None,
                      ) -> QuantityComparison:
    candidate_array = _as_array(candidate)
    reference_array = _as_array(reference)
    if candidate_array.size != reference_array.size:
        return QuantityComparison(
            case_id=result.case_id, component=result.component, quantity=name,
            status="shape_mismatch", candidate=candidate, reference=reference)

    absolute = absolute_error(candidate_array, reference_array)
    relative = relative_error(candidate_array, reference_array)
    nrmse = (normalized_rmse(candidate_array, reference_array)
             if candidate_array.size > 3 else None)
    peak_absolute: float | None = None
    peak_relative: float | None = None
    if candidate_array.size > 1:
        peak_absolute, peak_relative = peak_error(candidate_array,
                                                  reference_array)
    displacement: float | None = None
    if name == "center_of_pressure" and candidate_array.size == 3:
        displacement, _normalized = center_of_pressure_displacement(
            candidate_array, reference_array, reference_length)

    return QuantityComparison(
        case_id=result.case_id, component=result.component, quantity=name,
        status="compared", candidate=candidate, reference=reference,
        absolute_error=absolute, relative_error=relative,
        normalized_rmse=nrmse, peak_absolute_error=peak_absolute,
        peak_relative_error=peak_relative, displacement=displacement)


@dataclass
class ComparisonReport:
    """All quantity comparisons for a study run against one dataset."""

    label: str
    source: str
    comparisons: list[QuantityComparison] = field(default_factory=list)
    unmatched_cases: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.comparisons)

    def rows(self) -> list[dict[str, Any]]:
        return [comparison.to_row() for comparison in self.comparisons]

    def for_quantity(self, name: str) -> list[QuantityComparison]:
        return [c for c in self.comparisons if c.quantity == name]

    def max_relative_error(self, name: str) -> float | None:
        """Largest relative error recorded for a quantity, or None."""
        values = [c.relative_error for c in self.for_quantity(name)
                  if c.relative_error is not None]
        return max(values) if values else None

    def write_csv(self, path: str | os.PathLike[str]) -> str:
        rows = self.rows()
        if not rows:
            raise ValueError("no comparisons to write")
        path = os.fspath(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path


def compare_results(results: Iterable[CaseResult], dataset: ReferenceDataset,
                    reference_length: float | None = None) -> ComparisonReport:
    """Compare a study's case results against a reference dataset.

    Cases with no matching reference record are listed in
    ``ComparisonReport.unmatched_cases`` rather than being silently dropped
    or compared against invented values.
    """
    report = ComparisonReport(label=dataset.label, source=dataset.source)
    for result in results:
        record = dataset.match(result)
        if record is None:
            report.unmatched_cases.append(f"{result.case_id}/{result.component}")
            continue
        report.comparisons.extend(
            compare_case(result, record,
                         reference_length=reference_length))
    return report

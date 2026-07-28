"""
Prescribed Jet Firing History generation for plume/target validation studies.

Validation sweeps do not fly a trajectory: the firing poses are PRESCRIBED
(placed by the engineer, or generated from a swept approach angle and source
distance about a stationary target) and the Jet Firing History is written
from them directly. This module owns that generation, so the dynamics-based
approach-maneuver workflow in ``pyrpod.rpod.approach_maneuvers`` stays
untouched.

``n_firings`` semantics
-----------------------
``n_firings`` is the EXACT number of entries written to the JFH. The count is
validated before anything is generated (:func:`validate_n_firings`), the
generated sequence is asserted to match it (:func:`build_case_firings`), and
an explicitly prescribed firing list whose length disagrees is an error --
never a silent truncation or extension.

Pose convention
---------------
For a target reference point ``C`` with outward normal ``n_hat`` (pointing
toward the plume source side) and in-plane tangent ``t_hat``:

    d_hat(alpha) = cos(alpha) * n_hat + sin(alpha) * t_hat
    source position = C + L * d_hat(alpha)
    thruster axis   = -d_hat(alpha)            (aimed at C)

``alpha = 0`` is head-on. The JFH DCM is built with the thruster axis as its
first COLUMN, which is what the strike pipeline reads as the plume normal
(``dcm.T`` rows, with an identity thruster DCM). This reproduces the pose
convention of the committed sweep-JFH generators
(``case/plume/plume_flat_plate_sweep/jfh/generate_sweep_jfh.py``) exactly,
including the binormal choice ``cross(n_hat, t_hat)`` for the DCM's second
column.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from pyrpod.mdao.study_config import (
    PrescribedFiringSpec,
    StudyConfigError,
    SweepSpec,
    TargetSpec,
    validate_n_firings,
)

__all__ = [
    "Firing",
    "build_case_firings",
    "build_sweep_firings",
    "pose_for",
    "validate_n_firings",
    "write_jfh_file",
]


@dataclass(frozen=True)
class Firing:
    """One JFH entry: pose, active thrusters, firing duration and start time.

    Attributes
    ----------
    position : np.ndarray
        Visiting-vehicle (plume-source) position in the case's global frame.
    dcm : np.ndarray
        3x3 direction cosine matrix; its first column is the thruster axis.
    thrusters : tuple of int
        JFH thruster indices active during this firing (1-based).
    duration_s : float
        Firing time; the strike pipeline multiplies heat flux by this to get
        the per-firing heat-flux load.
    start_time_s : float
        Elapsed time at the start of this firing (informational; the strike
        calculation does not read it).
    plate_angle_deg, source_distance : float or None
        The swept parameters this firing realizes. Set when the firing came
        from a sweep pose, so a history spanning many poses stays keyed to
        the sweep grid; None for firings whose pose was prescribed outright
        without a sweep parameterization.
    pose_index : int or None
        Index of the firing's pose in the sweep's execution order.
    """

    position: NDArray[np.float64]
    dcm: NDArray[np.float64]
    thrusters: tuple[int, ...]
    duration_s: float = 1.0
    start_time_s: float = 0.0
    plate_angle_deg: float | None = None
    source_distance: float | None = None
    pose_index: int | None = None


def pose_for(alpha_deg: float, distance: float,
             reference_point: Sequence[float] | NDArray[np.float64],
             normal: Sequence[float] | NDArray[np.float64],
             tangent: Sequence[float] | NDArray[np.float64],
             ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Plume-source pose for one (approach angle, distance) combination.

    Parameters
    ----------
    alpha_deg : float
        Approach angle in degrees; 0 is head-on along ``normal``, positive
        angles rotate the source toward ``tangent``.
    distance : float
        Distance from ``reference_point`` to the plume source.
    reference_point, normal, tangent : array-like
        Target geometry axes (see :class:`pyrpod.mdao.study_config.TargetSpec`).

    Returns
    -------
    (np.ndarray, np.ndarray)
        The source position and the 3x3 DCM whose first column is the
        thruster axis (aimed back at ``reference_point``).
    """
    center = np.asarray(reference_point, dtype=float)
    n_hat = np.asarray(normal, dtype=float)
    t_hat = np.asarray(tangent, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    t_hat = t_hat / np.linalg.norm(t_hat)

    alpha = np.deg2rad(float(alpha_deg))
    d_hat = np.cos(alpha) * n_hat + np.sin(alpha) * t_hat
    position = center + float(distance) * d_hat
    axis = -d_hat                       # aimed at the target reference point

    # Right-handed triad completed with the sweep plane's binormal, matching
    # the committed sweep-JFH generators (which hard-code the global Y axis
    # for their X-Z sweep plane -- the same vector this expression yields).
    binormal = np.cross(n_hat, t_hat)
    binormal = binormal / np.linalg.norm(binormal)
    dcm = np.column_stack([axis, binormal, np.cross(axis, binormal)])
    return position, dcm


def build_case_firings(sweep: SweepSpec, target: TargetSpec,
                       alpha_deg: float, distance: float,
                       pose_index: int = 0,
                       start_time_s: float = 0.0) -> list[Firing]:
    """Build EXACTLY ``sweep.n_firings`` firings for one sweep pose.

    When the configuration prescribes firings explicitly they are used
    verbatim (their count is validated against ``n_firings`` when the
    configuration is parsed). Otherwise the swept pose is generated from
    ``alpha_deg`` / ``distance`` and repeated for ``n_firings`` successive
    firing intervals -- one JFH entry per requested firing.

    ``pose_index`` and ``start_time_s`` place this pose inside a longer
    sequence; they matter only when many poses share one history (see
    :func:`build_sweep_firings`).

    Raises
    ------
    StudyConfigError
        If the generated sequence length would differ from ``n_firings``.
    """
    n_firings = validate_n_firings(sweep.n_firings)

    if sweep.firings:
        specs = sweep.firings
        if sweep.mode == "single_jfh":
            # One shared history: each pose takes its own contiguous slice of
            # the prescribed sequence, in sweep order.
            offset = pose_index * n_firings
            specs = sweep.firings[offset:offset + n_firings]
        firings = [
            _from_spec(spec, index, alpha_deg, distance, pose_index,
                       start_time_s)
            for index, spec in enumerate(specs)
        ]
    else:
        position, dcm = pose_for(alpha_deg, distance,
                                 target.reference_point, target.normal,
                                 target.tangent)
        firings = [
            Firing(position=position, dcm=dcm, thrusters=sweep.thrusters,
                   duration_s=sweep.firing_duration_s,
                   start_time_s=start_time_s
                   + index * sweep.firing_duration_s,
                   plate_angle_deg=float(alpha_deg),
                   source_distance=float(distance),
                   pose_index=pose_index)
            for index in range(n_firings)
        ]

    if len(firings) != n_firings:
        raise StudyConfigError(
            f"requested n_firings={n_firings} but built {len(firings)} "
            "firings; the two must agree exactly")
    return firings


def build_sweep_firings(sweep: SweepSpec,
                        target: TargetSpec) -> list[Firing]:
    """Build the WHOLE sweep as one firing sequence.

    Every pose of ``sweep.poses`` contributes ``sweep.n_firings`` entries, in
    execution order, with firing times running continuously across the
    sequence. The result is the single Jet Firing History of a ``single_jfh``
    study, and its length is exactly ``sweep.total_firings``.

    Raises
    ------
    StudyConfigError
        If the assembled sequence length would differ from that product.
    """
    firings: list[Firing] = []
    elapsed = 0.0
    for pose_index, (angle, distance) in enumerate(sweep.poses):
        pose_firings = build_case_firings(sweep, target, angle, distance,
                                          pose_index=pose_index,
                                          start_time_s=elapsed)
        firings.extend(pose_firings)
        elapsed += sum(firing.duration_s for firing in pose_firings)

    if len(firings) != sweep.total_firings:
        raise StudyConfigError(
            f"sweep of {len(sweep.poses)} poses x n_firings="
            f"{sweep.n_firings} must produce {sweep.total_firings} JFH "
            f"entries, built {len(firings)}")
    return firings


def _from_spec(spec: PrescribedFiringSpec, index: int,
               alpha_deg: float | None = None,
               distance: float | None = None,
               pose_index: int | None = None,
               start_time_s: float = 0.0) -> Firing:
    return Firing(position=np.asarray(spec.position, dtype=float),
                  dcm=np.asarray(spec.dcm, dtype=float),
                  thrusters=spec.thrusters,
                  duration_s=spec.duration_s,
                  start_time_s=start_time_s + index * spec.duration_s,
                  plate_angle_deg=(None if alpha_deg is None
                                   else float(alpha_deg)),
                  source_distance=(None if distance is None
                                   else float(distance)),
                  pose_index=pose_index)


def write_jfh_file(path: str | os.PathLike[str],
                   firings: Iterable[Firing]) -> int:
    """Write firings to a JFH file readable by ``JetFiringHistory.read_jfh``.

    The emitted format matches the committed case generators byte-for-byte in
    structure (header line with the firing count, an unused second line, then
    one row per firing: index, dt, t, unused column, nine DCM values, three
    position values, uncertainty factor, thruster count, thruster indices).

    Returns
    -------
    int
        The number of entries written, which is the number of firings the
        JFH will report.
    """
    firings = list(firings)
    if not firings:
        raise StudyConfigError(
            "refusing to write a JFH with zero firings; n_firings must be a "
            "positive integer")

    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    lines = [f"offseted    {len(firings)}       0",
             "       0.000       0.000       0.000"]
    for index, firing in enumerate(firings, start=1):
        dcm = np.asarray(firing.dcm, dtype=float).reshape(3, 3)
        dcm_str = " ".join(f"{value:.6e}" for value in dcm.ravel())
        xyz_str = " ".join(f"{value:.9g}"
                           for value in np.asarray(firing.position,
                                                   dtype=float).ravel())
        thruster_str = " ".join(str(int(t)) for t in firing.thrusters)
        lines.append(
            f"      {index} {firing.start_time_s:g} {firing.duration_s:g} 0 "
            f"{dcm_str} {xyz_str} 1 {len(firing.thrusters)} {thruster_str}")

    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(firings)

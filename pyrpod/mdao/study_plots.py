"""
Optional parameter-sweep trend plots for plume validation studies.

Every function here is OPTIONAL output: nothing in the study engine, and no
automated test, requires a figure to be produced. The module is imported
lazily by :meth:`pyrpod.mdao.plume_validation.PlumeValidationStudy.plot` and
selects matplotlib's non-interactive Agg backend on import, so a headless or
CI run never opens a window (the same artifact-control behavior the RPOD
integration and verification tests use).

Produced figures (written to the study's ``plots`` subdirectory):

* ``force_vs_angle.png``     -- resultant force magnitude vs plate angle
* ``moment_vs_angle.png``    -- resultant moment magnitude vs plate angle
* ``heat_flux_vs_angle.png`` -- peak surface heat flux vs plate angle
* ``force_vs_distance.png``  -- resultant force magnitude vs source distance
* ``moment_vs_distance.png`` -- resultant moment magnitude vs source distance
* ``center_of_pressure.png`` -- center-of-pressure travel across the sweep
* ``reference_comparison.png`` -- study vs external reference, when a
  comparison report is supplied
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from pyrpod.mdao.reference_data import ComparisonReport  # noqa: E402
from pyrpod.mdao.study_results import CaseResult, StudyResults  # noqa: E402

__all__ = ["plot_sweep_trends"]

#: Categorical palette used across the sweep plots (fixed slot order, the
#: same CVD-safe palette the existing sweep study script uses).
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"]


def _distances(cases: Sequence[CaseResult]) -> list[float]:
    return sorted({case.source_distance for case in cases})


def _angles(cases: Sequence[CaseResult]) -> list[float]:
    return sorted({case.plate_angle_deg for case in cases})


def _series(cases: Sequence[CaseResult], fixed: str, value: float,
            variable: str, quantity: str) -> tuple[list[float], list[float]]:
    selected = [case for case in cases
                if getattr(case, fixed) == value]
    selected.sort(key=lambda case: getattr(case, variable))
    xs = [float(getattr(case, variable)) for case in selected]
    ys = [float(getattr(case, quantity)) for case in selected]
    return xs, ys


def _line_figure(path: str, title: str, xlabel: str, ylabel: str,
                 series: Iterable[tuple[str, list[float], list[float]]],
                 log_y: bool = False) -> str:
    figure, axes = plt.subplots(figsize=(7, 4.5))
    for color, (label, xs, ys) in zip(PALETTE * 8, series):
        axes.plot(xs, ys, "-o", color=color, lw=2, ms=4, label=label)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title, fontsize=10)
    if log_y:
        axes.set_yscale("log")
    axes.grid(True, color="0.9", lw=0.8)
    axes.legend(fontsize=8)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_sweep_trends(results: StudyResults, out_dir: str,
                      comparison: ComparisonReport | None = None,
                      component: str | None = None) -> list[str]:
    """Write the sweep trend figures; returns the paths written.

    Parameters
    ----------
    results : StudyResults
        Executed study results.
    out_dir : str
        Directory to write the figures to (created when missing).
    comparison : ComparisonReport, optional
        When supplied, an extra study-vs-reference figure is written.
    component : str, optional
        Restrict the plots to one target component. Defaults to the first
        component present in the results.
    """
    cases = list(results.cases)
    if not cases:
        return []
    component = component or cases[0].component
    cases = [case for case in cases if case.component == component]
    # One point per swept pose: keep the first record of each (angle,
    # distance). This is the first firing of each case when every pose has
    # its own history, and the first firing of each pose's slice when one
    # history spans the sweep -- both engines plot identically.
    seen: set[tuple[float, float]] = set()
    unique: list[CaseResult] = []
    for case in cases:
        pose = (case.plate_angle_deg, case.source_distance)
        if pose in seen:
            continue
        seen.add(pose)
        unique.append(case)
    cases = unique
    if not cases:
        return []

    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    distances = _distances(cases)
    angles = _angles(cases)

    for quantity, filename, ylabel, title in (
            ("force_magnitude", "force_vs_angle.png", "|F| (N)",
             "Resultant force vs plate angle"),
            ("moment_magnitude", "moment_vs_angle.png", "|M| (N*m)",
             "Resultant moment vs plate angle"),
            ("max_heat_flux", "heat_flux_vs_angle.png",
             "peak heat flux (W/m^2)", "Peak heat flux vs plate angle")):
        series = [(f"d = {distance:g}",
                   *_series(cases, "source_distance", distance,
                            "plate_angle_deg", quantity))
                  for distance in distances]
        written.append(_line_figure(
            os.path.join(out_dir, filename), f"{title} ({component})",
            "plate angle (deg), 0 = head-on", ylabel, series))

    if len(distances) > 1:
        for quantity, filename, ylabel, title in (
                ("force_magnitude", "force_vs_distance.png", "|F| (N)",
                 "Resultant force vs source distance"),
                ("moment_magnitude", "moment_vs_distance.png", "|M| (N*m)",
                 "Resultant moment vs source distance")):
            series = [(f"alpha = {angle:g} deg",
                       *_series(cases, "plate_angle_deg", angle,
                                "source_distance", quantity))
                      for angle in angles]
            written.append(_line_figure(
                os.path.join(out_dir, filename), f"{title} ({component})",
                "source distance", ylabel, series))

    cop_path = _plot_center_of_pressure(cases, out_dir, component)
    if cop_path:
        written.append(cop_path)

    if comparison is not None and len(comparison):
        written.append(_plot_reference_comparison(comparison, out_dir))

    return written


def _plot_center_of_pressure(cases: Sequence[CaseResult], out_dir: str,
                             component: str) -> str | None:
    """Center-of-pressure travel, projected on the two axes that vary most.

    Cases whose center of pressure is unavailable (zero load, or a resultant
    that is cancellation noise) simply do not appear.
    """
    located = [(case, np.asarray(case.center_of_pressure, dtype=float))
               for case in cases if case.center_of_pressure is not None]
    if not located:
        return None

    points = np.vstack([point for _, point in located])
    spread = points.max(axis=0) - points.min(axis=0)
    order = sorted(range(3), key=lambda axis: float(spread[axis]),
                   reverse=True)[:2]
    first, second = sorted(order)
    labels = "xyz"

    distances = sorted({case.source_distance for case, _ in located})
    figure, axes = plt.subplots(figsize=(6, 5))
    for color, distance in zip(PALETTE * 8, distances):
        subset = sorted((entry for entry in located
                         if entry[0].source_distance == distance),
                        key=lambda entry: entry[0].plate_angle_deg)
        axes.plot([point[first] for _, point in subset],
                  [point[second] for _, point in subset],
                  "-o", color=color, lw=1.6, ms=4, label=f"d = {distance:g}")
    axes.set_xlabel(f"CoP {labels[first]} (m)")
    axes.set_ylabel(f"CoP {labels[second]} (m)")
    axes.set_title(f"Center-of-pressure travel across the sweep ({component})",
                   fontsize=10)
    axes.grid(True, color="0.9", lw=0.8)
    axes.legend(fontsize=8)
    path = os.path.join(out_dir, "center_of_pressure.png")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_reference_comparison(comparison: ComparisonReport,
                               out_dir: str) -> str:
    """Bar chart of the relative error per compared quantity."""
    quantities: dict[str, list[float]] = {}
    for entry in comparison.comparisons:
        if entry.relative_error is None:
            continue
        quantities.setdefault(entry.quantity, []).append(entry.relative_error)

    figure, axes = plt.subplots(figsize=(7, 4.5))
    names = sorted(quantities)
    maxima = [max(quantities[name]) for name in names]
    means = [sum(quantities[name]) / len(quantities[name]) for name in names]
    positions = range(len(names))
    axes.bar([p - 0.2 for p in positions], maxima, width=0.4,
             color=PALETTE[0], label="max")
    axes.bar([p + 0.2 for p in positions], means, width=0.4,
             color=PALETTE[1], label="mean")
    axes.set_xticks(list(positions))
    axes.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes.set_ylabel("relative error")
    axes.set_title(f"Study vs {comparison.label}", fontsize=10)
    axes.grid(True, axis="y", color="0.9", lw=0.8)
    axes.legend(fontsize=8)
    path = os.path.join(out_dir, "reference_comparison.png")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path

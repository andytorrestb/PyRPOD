# ========================
# PyRPOD: tests/logging/logging_integration_test_01.py
# ========================
# Integration tests for operational logging of the plume-strike workflow:
# fail-fast validation of required inputs, progress cadence, serial/parallel
# event equivalence, DEBUG-vs-INFO artifact levels, parallel->serial fallback
# with successful_with_warning status, and optional-visualization warn-and-
# continue behavior. Uses the existing lightweight base_case (kinetics off).

import logging

import numpy as np
import pytest

import pyrpod.logging_utils as lu
from pyrpod.logging_utils import configure_logging
from pyrpod.rpod import JetFiringHistory, PlumeStrikeEstimationStudy
from pyrpod.rpod import PlumeStrikeEstimationStudy as PSES_module
from pyrpod.vehicle import LogisticsModule, TargetVehicle
from pyrpod.mission import MissionEnvironment

BASE_CASE = "../case/rpod/base_case/"
PKG_LOGGER = logging.getLogger("pyrpod")


@pytest.fixture(autouse=True)
def _isolate_pyrpod_logging():
    before_handlers = list(PKG_LOGGER.handlers)
    before_level = PKG_LOGGER.level
    yield
    for handler in list(PKG_LOGGER.handlers):
        if handler not in before_handlers:
            PKG_LOGGER.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    PKG_LOGGER.setLevel(before_level)
    lu._ACTIVE_SESSION = None


def build_study(case_dir=BASE_CASE, read_jfh=True):
    jfh = JetFiringHistory.JetFiringHistory(case_dir)
    if read_jfh:
        jfh.read_jfh()

    tv = TargetVehicle.TargetVehicle(case_dir)
    tv.set_stl()

    lm = LogisticsModule.LogisticsModule(case_dir)
    lm.set_thruster_config()

    me = MissionEnvironment.MissionEnvironment(case_dir)
    study = PlumeStrikeEstimationStudy.PlumeStrikeEstimationStudy(me)
    study.study_init(jfh, tv, lm)
    return study


def messages_at(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


# --------------------------------------------------------------------------- #
# 13, 14, 15: fail-fast validation of required inputs
# --------------------------------------------------------------------------- #
def test_missing_jfh_config_fails_fast():
    study = build_study()
    study.environment.config.remove_option("jfh", "jfh")
    with pytest.raises(KeyError):
        study.jfh_plume_strikes()


def test_missing_tcf_config_fails_fast():
    study = build_study()
    study.environment.config.remove_option("tcd", "tcf")
    with pytest.raises(KeyError):
        study.jfh_plume_strikes()


def test_empty_jfh_fails_fast(caplog):
    study = build_study()
    study.jfh.JFH = []
    with caplog.at_level(logging.ERROR, logger="pyrpod"):
        with pytest.raises(ValueError):
            study.jfh_plume_strikes()
    assert any("JFH is empty" in m for m in messages_at(caplog, logging.ERROR))


def test_unloaded_jfh_fails_fast():
    study = build_study(read_jfh=False)  # JFH object exists but never read
    study.jfh.JFH = None
    with pytest.raises(ValueError):
        study.jfh_plume_strikes()


def test_missing_target_mesh_fails_fast():
    study = build_study()
    study.target.mesh = None
    with pytest.raises(ValueError):
        study.jfh_plume_strikes()


# --------------------------------------------------------------------------- #
# 16: conditionally required inputs stay optional when their feature is off
# --------------------------------------------------------------------------- #
def test_kinetics_disabled_does_not_require_metrics():
    # base_case has kinetics = None and never loads thruster metrics.
    study = build_study()
    assert getattr(study.vv, "thruster_metrics", None) is None
    firing_data = study.jfh_plume_strikes()          # must not raise
    assert len(firing_data) == len(study.jfh.JFH)


# --------------------------------------------------------------------------- #
# 20: progress logging respects progress_every_n_firings
# --------------------------------------------------------------------------- #
def _count_progress_records(caplog):
    return sum(1 for r in caplog.records
               if r.levelno == logging.INFO
               and r.getMessage().startswith("Firing ")
               and "completed:" in r.getMessage())


def test_progress_every_n_firings_default(caplog):
    study = build_study()
    n = len(study.jfh.JFH)
    session = configure_logging(BASE_CASE, file=False, console=False,
                                write_startup=False,
                                progress_every_n_firings=1)
    try:
        with caplog.at_level(logging.INFO, logger="pyrpod"):
            study.jfh_plume_strikes()
        assert _count_progress_records(caplog) == n
    finally:
        session.close()


def test_progress_every_n_firings_interval(caplog):
    study = build_study()
    n = len(study.jfh.JFH)
    session = configure_logging(BASE_CASE, file=False, console=False,
                                write_startup=False,
                                progress_every_n_firings=5)
    try:
        with caplog.at_level(logging.INFO, logger="pyrpod"):
            study.jfh_plume_strikes()
        # Every 5th firing plus the final firing.
        expected = len([i for i in range(1, n + 1) if i % 5 == 0 or i == n])
        assert _count_progress_records(caplog) == expected
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# 19: serial and parallel workflows emit equivalent major event categories
# --------------------------------------------------------------------------- #
def _event_categories(caplog):
    joined = "\n".join(r.getMessage() for r in caplog.records)
    return {
        "run_started": "Plume-strike run started" in joined,
        "progress": "completed:" in joined,
        "run_completed": "Plume-strike run completed" in joined,
    }


def test_serial_and_parallel_emit_equivalent_events(caplog):
    with caplog.at_level(logging.INFO, logger="pyrpod"):
        build_study().jfh_plume_strikes()
    serial_events = _event_categories(caplog)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pyrpod"):
        build_study().jfh_plume_strikes(parallel=True, workers=2)
    parallel_events = _event_categories(caplog)

    assert serial_events == parallel_events
    assert all(serial_events.values())


# --------------------------------------------------------------------------- #
# 21: individual artifact paths at DEBUG, batch/summary at INFO
# --------------------------------------------------------------------------- #
def test_artifact_paths_debug_summary_info(caplog):
    study = build_study()
    with caplog.at_level(logging.DEBUG, logger="pyrpod"):
        study.jfh_plume_strikes()

    per_artifact = [r for r in caplog.records
                    if r.getMessage().startswith("Wrote firing artifact:")]
    assert per_artifact and all(r.levelno == logging.DEBUG for r in per_artifact)

    started = [r for r in caplog.records
               if "Plume-strike run started" in r.getMessage()]
    completed = [r for r in caplog.records
                 if "Plume-strike run completed" in r.getMessage()]
    assert started and all(r.levelno == logging.INFO for r in started)
    assert completed and all(r.levelno == logging.INFO for r in completed)


# --------------------------------------------------------------------------- #
# 17 & 18: parallel failure -> logged fallback -> successful_with_warning
# --------------------------------------------------------------------------- #
def test_parallel_fallback_logs_and_continues(caplog, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("forced parallel failure")

    monkeypatch.setattr(PSES_module, "run_parallel_plume_strikes", boom)

    reference = build_study().jfh_plume_strikes()  # plain serial baseline

    study = build_study()
    with caplog.at_level(logging.INFO, logger="pyrpod"):
        result = study.jfh_plume_strikes(parallel=True, workers=2)

    # Numerical result matches the serial baseline exactly.
    for firing in reference:
        np.testing.assert_array_equal(result[firing]["strikes"],
                                      reference[firing]["strikes"])

    warnings = messages_at(caplog, logging.WARNING)
    assert any("falling back to serial execution" in m for m in warnings)
    # Traceback captured on the fallback warning.
    assert any(r.exc_info for r in caplog.records
               if r.levelno == logging.WARNING)

    completed = next(m for m in messages_at(caplog, logging.INFO)
                     if "Plume-strike run completed" in m)
    assert "run_status=successful_with_warning" in completed
    assert "fallback_occurred=True" in completed
    assert "execution_mode=serial" in completed


# --------------------------------------------------------------------------- #
# 22: optional visualization write failures warn and continue
# --------------------------------------------------------------------------- #
def test_optional_visualization_failure_warns_and_continues(caplog):
    study = build_study()
    calls = {"n": 0}
    original = study.viz.export_firing

    def flaky_export(mesh_obj, path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full (simulated)")
        return original(mesh_obj, path)

    study.viz.export_firing = flaky_export

    with caplog.at_level(logging.WARNING, logger="pyrpod"):
        study.graph_jfh()  # must not raise despite the first-firing failure

    warnings = messages_at(caplog, logging.WARNING)
    assert any("Optional JFH visualization write failed" in m for m in warnings)
    # All remaining firings were still attempted.
    assert calls["n"] == len(study.jfh.JFH)


# --------------------------------------------------------------------------- #
# 23: the renamed expected-strikes .txt fixture is read correctly
# --------------------------------------------------------------------------- #
def test_expected_strikes_txt_fixture_is_readable():
    from pathlib import Path
    fixture = Path(__file__).resolve().parents[1] / "rpod" / \
        "rpod_int_test_01_expected_strikes.txt"
    assert fixture.is_file()
    content = fixture.read_text(encoding="utf-8")
    assert "n_firing" in content
    # Non-'n_firing' lines are integer face indices.
    for line in content.splitlines():
        if line and "n_firing" not in line:
            int(line)


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

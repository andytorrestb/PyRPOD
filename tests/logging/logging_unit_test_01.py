# ========================
# PyRPOD: tests/logging/logging_unit_test_01.py
# ========================
# Unit tests for the centralized operational logging system
# (pyrpod.logging_utils): import side effects, handler ownership, configuration
# precedence, console/file toggles, runtime-log naming/location, configuration
# snapshots, input-asset logging, and array summaries.
#
# These tests use temporary case directories and pytest's caplog/monkeypatch;
# none use logging.basicConfig.

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pyrpod.logging_utils as lu
from pyrpod.logging_utils import (
    LoggingSettings,
    configure_logging,
    log_asset,
    summarize_array,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_LOGGER = logging.getLogger("pyrpod")

MINIMAL_CONFIG = """[vv]
stl_lm = cylinder.stl

[tv]
stl = plate.stl

[pm]
kinetics = None

[jfh]
jfh = firings.A

[tcd]
tcf = thrusters.txt

[plume]
radius = 25
wedge_theta = 0.262
"""


@pytest.fixture(autouse=True)
def _isolate_pyrpod_logging():
    """Restore the pyrpod logger and clear the active session after each test."""
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


def make_case(tmp_path, *, config=MINIMAL_CONFIG, logging_ini=None):
    case_dir = tmp_path / "mycase"
    case_dir.mkdir()
    (case_dir / "config.ini").write_text(config, encoding="utf-8")
    if logging_ini is not None:
        (case_dir / "logging.ini").write_text(logging_ini, encoding="utf-8")
    # configure_logging normalizes trailing separators; keep the pyrpod
    # convention of a trailing slash on case_dir strings.
    return str(case_dir) + os.sep


def owned_handlers(logger):
    return [h for h in logger.handlers
            if getattr(h, lu._OWNED_HANDLER_ATTR, False)]


# --------------------------------------------------------------------------- #
# 1 & 2: import side effects
# --------------------------------------------------------------------------- #
def test_import_has_no_logging_side_effects(tmp_path):
    """Importing PyRPOD creates no dirs/log files and configures no root logger."""
    code = (
        "import os, logging, json\n"
        "before = set(os.listdir('.'))\n"
        "import pyrpod\n"
        "import pyrpod.rpod.PlumeStrikeEstimationStudy\n"
        "import pyrpod.rpod.JetFiringHistory\n"
        "import pyrpod.vehicle.VisitingVehicle\n"
        "import pyrpod.vehicle.TargetVehicle\n"
        "root = logging.getLogger()\n"
        "pkg = logging.getLogger('pyrpod')\n"
        "after = set(os.listdir('.'))\n"
        "print(json.dumps({\n"
        "    'root_handlers': len(root.handlers),\n"
        "    'pkg_handlers': len(pkg.handlers),\n"
        "    'new_files': sorted(after - before),\n"
        "}))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["root_handlers"] == 0
    assert data["pkg_handlers"] == 0
    assert data["new_files"] == []


# --------------------------------------------------------------------------- #
# 3 & 25: repeated configuration does not duplicate handlers or messages
# --------------------------------------------------------------------------- #
def test_repeated_configuration_no_duplicate_handlers(tmp_path):
    case = make_case(tmp_path)
    s1 = configure_logging(case, console=False, write_startup=False)
    n1 = len(owned_handlers(PKG_LOGGER))
    s2 = configure_logging(case, console=False, write_startup=False)
    n2 = len(owned_handlers(PKG_LOGGER))
    try:
        assert n1 == n2  # replaced, not accumulated
    finally:
        s2.close()
    # s1's handlers were closed when s2 replaced them.
    assert s1.log_path != s2.log_path


def test_repeated_configuration_no_duplicate_messages(tmp_path):
    case = make_case(tmp_path)
    configure_logging(case, console=False, write_startup=False)
    session = configure_logging(case, console=False, write_startup=False)
    try:
        logging.getLogger("pyrpod.test").info("unique-marker-xyz")
        for h in PKG_LOGGER.handlers:
            h.flush()
        content = Path(session.log_path).read_text(encoding="utf-8")
    finally:
        session.close()
    assert content.count("unique-marker-xyz") == 1


# --------------------------------------------------------------------------- #
# 4 & 5: console and file toggles are independent
# --------------------------------------------------------------------------- #
def test_console_toggle(tmp_path):
    case = make_case(tmp_path)
    s = configure_logging(case, console=True, file=False, write_startup=False)
    try:
        streams = [h for h in owned_handlers(PKG_LOGGER)
                   if isinstance(h, logging.StreamHandler)
                   and not isinstance(h, logging.FileHandler)]
        assert len(streams) == 1
    finally:
        s.close()

    s = configure_logging(case, console=False, file=False, write_startup=False)
    try:
        streams = [h for h in owned_handlers(PKG_LOGGER)
                   if type(h) is logging.StreamHandler]
        assert streams == []
    finally:
        s.close()


def test_file_toggle(tmp_path):
    case = make_case(tmp_path)
    s = configure_logging(case, console=False, file=True, write_startup=False)
    try:
        assert s.log_path is not None
        assert os.path.isfile(s.log_path)
    finally:
        s.close()

    s = configure_logging(case, console=False, file=False, write_startup=False)
    try:
        assert s.log_path is None
        assert [h for h in owned_handlers(PKG_LOGGER)
                if isinstance(h, logging.FileHandler)] == []
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# 6 & 7: a new run creates a new timestamped file under case/results/logs/
# --------------------------------------------------------------------------- #
def test_new_run_creates_new_log_under_results_logs(tmp_path):
    case = make_case(tmp_path)
    s1 = configure_logging(case, console=False, write_startup=False)
    p1 = s1.log_path
    s2 = configure_logging(case, console=False, write_startup=False)
    p2 = s2.log_path
    try:
        assert p1 != p2
        assert os.path.isfile(p1) and os.path.isfile(p2)
        expected_dir = os.path.join(os.path.abspath(case), "results", "logs")
        assert os.path.dirname(p1) == expected_dir
        assert os.path.basename(p1).startswith("mycase_")
        assert p1.endswith(".log")
    finally:
        s2.close()


# --------------------------------------------------------------------------- #
# 8 & 9: precedence — env overrides ini, ini overrides defaults
# --------------------------------------------------------------------------- #
def test_ini_overrides_defaults(tmp_path):
    case = make_case(tmp_path, logging_ini="[logging]\nlevel = WARNING\n")
    s = configure_logging(case, file=False, console=False, write_startup=False)
    try:
        assert s.settings.level == "WARNING"       # default would be INFO
        assert s.used_ini is True
    finally:
        s.close()


def test_env_overrides_ini(tmp_path, monkeypatch):
    case = make_case(tmp_path, logging_ini="[logging]\nlevel = ERROR\n")
    monkeypatch.setenv("PYRPOD_LOG_LEVEL", "DEBUG")
    s = configure_logging(case, file=False, console=False, write_startup=False)
    try:
        assert s.settings.level == "DEBUG"         # env beats ini
    finally:
        s.close()


def test_explicit_argument_overrides_env(tmp_path, monkeypatch):
    case = make_case(tmp_path)
    monkeypatch.setenv("PYRPOD_LOG_LEVEL", "DEBUG")
    s = configure_logging(case, level="ERROR", file=False, console=False,
                          write_startup=False)
    try:
        assert s.settings.level == "ERROR"         # explicit beats env
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# 10: input assets logged with absolute path, source, size, and SHA-256
# --------------------------------------------------------------------------- #
def test_log_asset_records_provenance_and_checksum(tmp_path, caplog):
    case = make_case(tmp_path)
    case_dir = os.path.abspath(case)
    asset_dir = os.path.join(case_dir, "stl")
    os.makedirs(asset_dir)
    asset_path = os.path.join(asset_dir, "plate.stl")
    payload = b"solid test\nendsolid test\n"
    with open(asset_path, "wb") as fh:
        fh.write(payload)

    import hashlib
    expected_sha = hashlib.sha256(payload).hexdigest()

    with caplog.at_level(logging.INFO, logger="pyrpod"):
        log_asset("target STL", "plate.stl", asset_path, case_dir)

    messages = [r.getMessage() for r in caplog.records]
    record = next(m for m in messages if "asset loaded" in m)
    assert os.path.abspath(asset_path) in record
    assert "source=case-local" in record
    assert f"size_bytes={len(payload)}" in record
    assert f"sha256={expected_sha}" in record


def test_log_asset_can_disable_checksums(tmp_path, caplog):
    case = make_case(tmp_path)
    case_dir = os.path.abspath(case)
    asset_path = os.path.join(case_dir, "config.ini")
    session = configure_logging(case, checksum_inputs=False, file=False,
                                console=False, write_startup=False)
    try:
        with caplog.at_level(logging.INFO, logger="pyrpod"):
            log_asset("config", "config.ini", asset_path, case_dir)
        record = next(m for m in (r.getMessage() for r in caplog.records)
                      if "asset loaded" in m)
        assert "sha256=n/a" in record
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# 11: configuration snapshots preserve exact contents
# --------------------------------------------------------------------------- #
def test_config_snapshot_preserves_contents(tmp_path):
    logging_ini = "[logging]\nlevel = INFO\nconsole = false\n"
    case = make_case(tmp_path, logging_ini=logging_ini)
    session = configure_logging(case, console=False, snapshot_config=True)
    try:
        log_dir = os.path.dirname(session.log_path)
        prefix = f"{session.case_name}_{session.timestamp}"
        config_snap = os.path.join(log_dir, f"{prefix}_config.ini")
        logging_snap = os.path.join(log_dir, f"{prefix}_logging.ini")
        assert os.path.isfile(config_snap)
        assert Path(config_snap).read_text(encoding="utf-8") == MINIMAL_CONFIG
        assert os.path.isfile(logging_snap)
        assert Path(logging_snap).read_text(encoding="utf-8") == logging_ini
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# 12: array-summary helper handles normal, empty, integer, and NaN arrays
# --------------------------------------------------------------------------- #
def test_summarize_array_variants():
    normal = summarize_array("p", np.array([1.0, 2.0, 3.0, 0.0]))
    assert "shape=(4,)" in normal and "nonzero=3" in normal and "zero=1" in normal
    assert "nan=0" in normal

    empty = summarize_array("e", np.array([]))
    assert "empty=True" in empty

    integers = summarize_array("i", np.array([0, 1, 2, 3], dtype=np.int64))
    assert "dtype=int64" in integers and "min=0" in integers and "max=3" in integers

    with_nan = summarize_array("n", np.array([1.0, np.nan, 3.0]))
    assert "nan=1" in with_nan and "min=1" in with_nan and "max=3" in with_nan

    boolean = summarize_array("b", np.array([True, False, True]))
    assert "dtype=bool" in boolean and "nonzero=2" in boolean

    non_array = summarize_array("o", object())
    assert "numeric=False" in non_array or "unsummarizable" in non_array


def test_disabled_logging_installs_no_console_or_file_handlers(tmp_path):
    case = make_case(tmp_path)
    session = configure_logging(case, enabled=False)
    try:
        assert session.log_path is None
        stream_or_file = [h for h in owned_handlers(PKG_LOGGER)
                          if isinstance(h, logging.StreamHandler)]
        assert stream_or_file == []          # only the (non-emitting) counter
    finally:
        session.close()


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

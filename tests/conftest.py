import html
import os
import shutil
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

CATEGORIES = ("unit", "integration", "verification")
GROUPS = ("logging", "mdao", "mission", "plume", "rpod", "tooling")

# These files import a flat `pyrpod.<Module>` layout that no longer exists after
# the rpod -> plume/vehicle refactor. Excluded from collection until they are
# fixed or removed outright.
collect_ignore = ["old", "test_case_25.py", "rpod_verification_test_05.py"]

# Manual-run verification FIGURE scripts (design decision D5): plume
# plume_verification_test_04..40 reproduce Cai & Wang 2012 / Cai 2016 paper
# figures when run directly (`python tests/plume/plume_verification_test_NN.py`)
# and define no pytest tests -- collecting them only imports dead weight and
# muddies the "verification" taxonomy. Exclude them so the pytest suite holds
# real tests only; _01..03 are genuine pytest verification tests and stay
# collected. Run a figure script directly to regenerate its figure.
collect_ignore += [
    f"plume/plume_verification_test_{n:02d}.py" for n in range(4, 41)
]


@pytest.fixture(autouse=True, scope="session")
def _run_from_tests_dir():
    # Legacy test cases assume the process CWD is this `tests/` directory,
    # e.g. `case_dir = '../case/...'` and `open('rpod/rpod_int_test_02...')`.
    # Reproduce that historical invocation so paths resolve without editing
    # every test file's path strings.
    previous_cwd = os.getcwd()
    os.chdir(TESTS_DIR)
    yield
    os.chdir(previous_cwd)


def pytest_collection_modifyitems(items):
    for item in items:
        stem = Path(item.fspath).stem
        relative_parts = Path(item.fspath).resolve().relative_to(TESTS_DIR).parts

        for category in CATEGORIES:
            if f"_{category}_test" in stem:
                item.add_marker(getattr(pytest.mark, category))

        for group in GROUPS:
            if group in relative_parts or stem.startswith(f"{group}_"):
                item.add_marker(getattr(pytest.mark, group))


# ---------------------------------------------------------------------------
# pytest-html reporting
# ---------------------------------------------------------------------------
# Inventory metadata (description, subsystem, development status, ...) lives in
# tests/test_manifest.yaml; the hooks below surface it as extra columns in
# reports/pyrpod-pytest-report.html so a reader can tell "this run passed" from
# "this test is finished". Everything here degrades to a plain pytest-html
# report if PyYAML or the manifest is unavailable -- a normal `pytest` run must
# never depend on the reporting extras.

MANIFEST_PATH = TESTS_DIR / "test_manifest.yaml"
REPORT_LOG_DIR = REPO_ROOT / "reports" / "logs"

# Extra result-table columns, in insertion order after pytest-html's "Test".
_EXTRA_COLUMNS = (
    ("Description", "description"),
    ("Subsystem", "subsystem"),
    ("Category", "category"),
    ("Mode", "execution_mode"),
    ("Dev status", "development_status"),
    ("Reference", "reference"),
    ("Source", "path"),
)

_manifest_cache = None


def _manifest_by_path():
    """{repo-relative path: manifest entry}, or {} if it cannot be read."""
    global _manifest_cache
    if _manifest_cache is None:
        entries = {}
        try:
            import yaml

            data = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
            for entry in data["tests"]:
                entries[entry["path"]] = entry
        except Exception:
            # Missing PyYAML / manifest only costs the extra report columns.
            entries = {}
        _manifest_cache = entries
    return _manifest_cache


def _metadata_for(item):
    """Manifest metadata for the file a test item came from."""
    try:
        path = Path(str(item.fspath)).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return {}
    entry = _manifest_by_path().get(path)
    if entry is None:
        return {"path": path}
    return {**entry, "path": path}


def _associated_log_files(item, start, stop):
    """PyRPOD run logs written by this test, matched by modification time.

    Only files under a `results/logs/` directory whose mtime falls inside the
    test's own call phase are returned, so a log can never be attributed to a
    test that did not produce it.
    """
    matched = []
    for log in (REPO_ROOT / "case").rglob("results/logs/*.log"):
        try:
            mtime = log.stat().st_mtime
        except OSError:
            continue
        if start <= mtime <= stop:
            matched.append(log)
    return sorted(matched)


def _slug(nodeid):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in nodeid)[:120]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    report.pyrpod_meta = _metadata_for(item)

    if report.when == "call" and report.failed:
        _attach_logs(report, item, call)
    return report


def _attach_logs(report, item, call):
    """Copy this test's PyRPOD logs into reports/logs/ and embed them."""
    logs = _associated_log_files(item, call.start, call.stop)
    if not logs:
        return
    try:
        from pytest_html import extras
    except ImportError:
        extras = None

    collected = []
    for index, log in enumerate(logs):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        collected.append((log, text))
        try:
            REPORT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                log, REPORT_LOG_DIR / f"{_slug(report.nodeid)}.{index}.log"
            )
        except OSError:
            pass

    if extras is None or not collected:
        return
    report.extras = list(getattr(report, "extras", []))
    for log, text in collected:
        name = log.relative_to(REPO_ROOT).as_posix()
        report.extras.append(extras.text(text, name=f"PyRPOD log: {name}"))


def pytest_html_report_title(report):
    report.title = "PyRPOD test execution report"


def pytest_html_results_table_header(cells):
    for offset, (title, _) in enumerate(_EXTRA_COLUMNS):
        cells.insert(2 + offset, f"<th>{html.escape(title)}</th>")


def pytest_html_results_table_row(report, cells):
    meta = getattr(report, "pyrpod_meta", {}) or {}
    for offset, (_, key) in enumerate(_EXTRA_COLUMNS):
        value = meta.get(key)
        text = " ".join(str(value).split()) if value else "—"
        cells.insert(2 + offset, f"<td>{html.escape(text)}</td>")

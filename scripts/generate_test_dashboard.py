#!/usr/bin/env python
"""Regenerate the PyRPOD test inventory and the pytest HTML execution report.

    python scripts/generate_test_dashboard.py

The script keeps two deliberately separate concepts apart:

* **Development status** - inventory-level metadata maintained by hand in
  ``tests/test_manifest.yaml`` (implemented, placeholder, needs_review, ...).
  It is rendered into the committed ``tests/README.md``.
* **Execution outcome** - pass / fail / skip for a single run, owned by pytest
  and rendered into the run-specific ``reports/pyrpod-pytest-report.html``.

Steps performed:

1. Validate the manifest schema.
2. Collect the current pytest tests (``pytest --collect-only``).
3. Cross-check the manifest against that collection.
4. Regenerate ``tests/README.md``.
5. Create ``reports/``.
6. Run pytest with ``pytest-html``, writing
   ``reports/pyrpod-pytest-report.html``.
7. Exit with pytest's exit code (the report is written either way).

Requires the ``test`` dependency group (``pytest-html``, ``PyYAML``); see the
Testing section of the root README.md for the install command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "tests" / "test_manifest.yaml"
README_PATH = REPO_ROOT / "tests" / "README.md"
REPORTS_DIR = REPO_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "pyrpod-pytest-report.html"

SUBSYSTEMS = ("logging", "mdao", "mission", "plume", "rpod", "tooling")
SUBSYSTEM_TITLES = {
    "logging": "Logging",
    "mdao": "MDAO",
    "mission": "Mission",
    "plume": "Plume",
    "rpod": "RPOD",
    "tooling": "Tooling",
}
CATEGORIES = ("unit", "integration", "verification")
EXECUTION_MODES = ("automated", "manual")
DEVELOPMENT_STATUSES = (
    "implemented",
    "placeholder",
    "needs_review",
    "blocked",
    "archived",
    "deprecated",
)
COLLECTION_STATUSES = ("collected", "ignored", "manual", "archived")

REQUIRED_FIELDS = (
    "path",
    "description",
    "subsystem",
    "category",
    "execution_mode",
    "development_status",
    "collection_status",
)
OPTIONAL_FIELDS = ("reference", "manual_command", "collection_ignore_reason")
KNOWN_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

# Files under tests/ that are support code rather than test assets and so are
# deliberately absent from the manifest.
NON_TEST_FILES = frozenset(
    {
        "tests/conftest.py",
        "tests/plume/plume_figure_utils.py",
        "tests/plume/plume_impingement_utils.py",
    }
)

REVIEW_REQUIRED = "REVIEW REQUIRED"

AUTOGEN_HEADER = (
    "<!-- AUTO-GENERATED FILE. DO NOT EDIT DIRECTLY.\n"
    "     Edit tests/test_manifest.yaml and rerun the inventory generator:\n"
    "         python scripts/generate_test_dashboard.py -->"
)


class ManifestError(Exception):
    """Raised when the manifest cannot be parsed or fails validation."""


# --------------------------------------------------------------------------
# Manifest loading and validation
# --------------------------------------------------------------------------


def load_manifest(path=MANIFEST_PATH):
    """Parse the manifest and return its list of entries.

    Raises ManifestError for anything that makes the file unusable (missing,
    unparseable, wrong top-level shape).
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ManifestError(
            "PyYAML is required to read the test manifest. Install the test "
            'dependency group: python -m pip install "PyYAML>=6"'
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict) or "tests" not in data:
        raise ManifestError(f"{path} must be a mapping with a top-level 'tests' key")

    entries = data["tests"]
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"{path}: 'tests' must be a non-empty list")

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"{path}: entry #{index + 1} is not a mapping")

    return entries


def validate_manifest(entries, repo_root=REPO_ROOT):
    """Return a list of human-readable schema problems (empty when valid)."""
    problems = []
    seen = set()

    for index, entry in enumerate(entries):
        label = entry.get("path") or f"entry #{index + 1}"

        for field in REQUIRED_FIELDS:
            value = entry.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                problems.append(f"{label}: missing required field '{field}'")

        for field in sorted(set(entry) - set(KNOWN_FIELDS)):
            problems.append(f"{label}: unknown field '{field}'")

        path = entry.get("path")
        if isinstance(path, str) and path:
            if path in seen:
                problems.append(f"{label}: duplicate path")
            seen.add(path)
            if "\\" in path:
                problems.append(f"{label}: path must use '/' separators")
            if not (repo_root / path).is_file():
                problems.append(f"{label}: file does not exist")

        for field, allowed in (
            ("subsystem", SUBSYSTEMS),
            ("category", CATEGORIES),
            ("execution_mode", EXECUTION_MODES),
            ("development_status", DEVELOPMENT_STATUSES),
            ("collection_status", COLLECTION_STATUSES),
        ):
            value = entry.get(field)
            if value is not None and value not in allowed:
                problems.append(
                    f"{label}: {field} '{value}' is not one of {', '.join(allowed)}"
                )

        mode = entry.get("execution_mode")
        collection = entry.get("collection_status")
        if mode == "manual" and not entry.get("manual_command"):
            # Blocked/archived assets have no meaningful command to document.
            if entry.get("development_status") not in ("blocked", "archived"):
                problems.append(
                    f"{label}: execution_mode 'manual' requires a manual_command"
                )
        if mode == "automated" and collection != "collected":
            problems.append(
                f"{label}: execution_mode 'automated' requires "
                f"collection_status 'collected', got '{collection}'"
            )
        if collection != "collected" and not entry.get("collection_ignore_reason"):
            problems.append(
                f"{label}: collection_status '{collection}' requires a "
                "collection_ignore_reason"
            )
        if collection == "collected" and entry.get("collection_ignore_reason"):
            problems.append(
                f"{label}: collection_status 'collected' must not set "
                "collection_ignore_reason"
            )

        description = entry.get("description")
        if isinstance(description, str) and description.startswith(REVIEW_REQUIRED):
            if entry.get("development_status") != "needs_review":
                problems.append(
                    f"{label}: '{REVIEW_REQUIRED}' description requires "
                    "development_status 'needs_review'"
                )

    return problems


# --------------------------------------------------------------------------
# Pytest collection
# --------------------------------------------------------------------------


def collect_pytest_nodes(repo_root=REPO_ROOT):
    """Return the sorted node IDs pytest currently collects."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise ManifestError(
            "pytest collection failed with exit code "
            f"{completed.returncode}:\n{completed.stdout}\n{completed.stderr}"
        )
    nodes = [
        line.strip()
        for line in completed.stdout.splitlines()
        if "::" in line and not line.startswith(("=", "_", " "))
    ]
    return sorted(set(nodes))


def files_from_nodes(nodes):
    """Map collected node IDs to their sorted set of source files."""
    return sorted({node.split("::", 1)[0] for node in nodes})


def cross_check(entries, collected_files, repo_root=REPO_ROOT):
    """Compare manifest entries against actual pytest collection.

    Returns a list of human-readable problems: collected tests with no
    manifest entry, manifest entries claiming collection that pytest does not
    collect, and test files on disk that the manifest does not document.
    """
    problems = []
    collected = set(collected_files)
    by_path = {entry["path"]: entry for entry in entries if entry.get("path")}

    for path in sorted(collected - set(by_path)):
        problems.append(f"{path}: collected by pytest but missing from the manifest")

    for path in sorted(by_path):
        entry = by_path[path]
        if entry.get("collection_status") == "collected" and path not in collected:
            problems.append(
                f"{path}: manifest says collection_status 'collected' but pytest "
                "collects no tests from it (stale entry)"
            )
        if entry.get("collection_status") != "collected" and path in collected:
            problems.append(
                f"{path}: pytest collects tests from it but the manifest says "
                f"collection_status '{entry.get('collection_status')}'"
            )

    tests_dir = Path(repo_root) / "tests"
    on_disk = {
        p.relative_to(repo_root).as_posix()
        for p in tests_dir.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    for path in sorted(on_disk - set(by_path) - NON_TEST_FILES - collected):
        problems.append(f"{path}: test file on disk is missing from the manifest")

    return problems


# --------------------------------------------------------------------------
# tests/README.md rendering
# --------------------------------------------------------------------------


def _cell(value, default="—"):
    """Collapse a manifest value into a single markdown table cell."""
    if value is None:
        return default
    text = " ".join(str(value).split())
    return text.replace("|", "\\|") if text else default


def _code(value, default="—"):
    text = _cell(value, default)
    return f"`{text}`" if text != default else default


def _sorted(entries):
    return sorted(
        entries,
        key=lambda e: (
            SUBSYSTEMS.index(e["subsystem"]),
            CATEGORIES.index(e["category"]),
            e["path"],
        ),
    )


def _table(lines, header, rows):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(" --- " for _ in header) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")


def render_readme(entries, collected_nodes=()):
    """Render the full text of tests/README.md.

    Deterministic: the output depends only on the manifest contents and the
    set of collected node IDs, never on timestamps or run outcomes.
    """
    entries = _sorted(entries)
    node_counts = {}
    for node in collected_nodes:
        path = node.split("::", 1)[0]
        node_counts[path] = node_counts.get(path, 0) + 1

    automated = [e for e in entries if e["collection_status"] == "collected"]
    manual = [e for e in entries if e["collection_status"] == "manual"]
    ignored = [e for e in entries if e["collection_status"] == "ignored"]
    archived = [e for e in entries if e["collection_status"] == "archived"]
    placeholders = [e for e in entries if e["development_status"] == "placeholder"]
    blocked = [e for e in entries if e["development_status"] == "blocked"]
    needs_review = [e for e in entries if e["development_status"] == "needs_review"]

    lines = [AUTOGEN_HEADER, "", "# PyRPOD Test Inventory", ""]
    lines += [
        "This inventory documents every test asset in `tests/`: what it is for,",
        "which subsystem and category it belongs to, how it is executed, and how",
        "far its development has progressed.",
        "",
        "**It deliberately records no pass/fail results.** Execution outcomes belong",
        "to a specific run and are published in the pytest HTML report",
        "(`reports/pyrpod-pytest-report.html`); development status is long-lived",
        "metadata that a green test run does not change.",
        "",
        "Test files follow the `<subsystem>_<category>_test_NN.py` naming convention,",
        "and `tests/conftest.py` uses it to tag every collected test with a subsystem",
        "marker (`logging`, `mdao`, `mission`, `plume`, `rpod`, `tooling`) and a",
        "category marker (`unit`, `integration`, `verification`).",
        "",
        "Regenerate this file and the HTML report with:",
        "",
        "```bash",
        "python scripts/generate_test_dashboard.py",
        "```",
        "",
        "Source of truth for the metadata below: "
        "[`test_manifest.yaml`](test_manifest.yaml).",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]
    _table(
        lines,
        ["Metric", "Count"],
        [
            ["Manifest entries", str(len(entries))],
            ["Collected by pytest (files)", str(len(automated))],
            ["Collected by pytest (test cases)", str(sum(node_counts.values()))],
            ["Manual verification scripts", str(len(manual))],
            ["Placeholder tests", str(len(placeholders))],
            ["Blocked tests", str(len(blocked))],
            ["Archived / legacy tests", str(len(archived))],
            ["Needs review", str(len(needs_review))],
        ],
    )

    lines += [
        "---",
        "",
        "## Automated pytest tests",
        "",
        "Files pytest collects and runs. The **Cases** column is the number of test",
        "cases pytest currently collects from the file (parametrized variants count",
        "separately).",
        "",
    ]
    for subsystem in SUBSYSTEMS:
        subset = [e for e in automated if e["subsystem"] == subsystem]
        if not subset:
            continue
        lines += [f"### {SUBSYSTEM_TITLES[subsystem]}", ""]
        for category in CATEGORIES:
            rows = [e for e in subset if e["category"] == category]
            if not rows:
                continue
            lines += [f"#### {category.capitalize()}", ""]
            _table(
                lines,
                [
                    "Test file",
                    "Cases",
                    "Description",
                    "Development status",
                    "Reference",
                ],
                [
                    [
                        f"[`{Path(e['path']).name}`]({_relative(e['path'])})",
                        str(node_counts.get(e["path"], 0)),
                        _cell(e["description"]),
                        f"`{e['development_status']}`",
                        _cell(e.get("reference")),
                    ]
                    for e in rows
                ],
            )

    lines += [
        "---",
        "",
        "## Manual verification scripts",
        "",
        "Run directly from the repository root; they are **not** pytest tests and",
        "never appear in the HTML execution report. Most reproduce a published",
        "figure and write a PNG under `tests/plume/output/`.",
        "",
    ]
    for subsystem in SUBSYSTEMS:
        rows = [e for e in manual if e["subsystem"] == subsystem]
        if not rows:
            continue
        lines += [f"### {SUBSYSTEM_TITLES[subsystem]}", ""]
        _table(
            lines,
            ["Script", "Description", "Development status", "Command", "Reference"],
            [
                [
                    f"[`{Path(e['path']).name}`]({_relative(e['path'])})",
                    _cell(e["description"]),
                    f"`{e['development_status']}`",
                    _code(e.get("manual_command")),
                    _cell(e.get("reference")),
                ]
                for e in rows
            ],
        )

    lines += [
        "---",
        "",
        "## Placeholder tests",
        "",
        "Collected by pytest but containing no assertion or verification behavior -",
        "typically an empty body or a fully commented-out one. Each is explicitly",
        "skipped in its source file so that the HTML report shows it as **skipped**",
        "rather than passed; a placeholder must never be mistaken for coverage.",
        "",
    ]
    _table(
        lines,
        ["Test file", "Subsystem", "Category", "Description"],
        [
            [
                f"[`{Path(e['path']).name}`]({_relative(e['path'])})",
                e["subsystem"],
                e["category"],
                _cell(e["description"]),
            ]
            for e in placeholders
        ],
    )

    lines += [
        "---",
        "",
        "## Ignored or blocked tests",
        "",
        "Excluded from pytest collection (see `collect_ignore` in",
        "[`conftest.py`](conftest.py)) or otherwise unable to run.",
        "",
    ]
    _table(
        lines,
        [
            "Test file",
            "Subsystem",
            "Development status",
            "Collection status",
            "Reason",
            "Command",
        ],
        [
            [
                f"[`{Path(e['path']).name}`]({_relative(e['path'])})",
                e["subsystem"],
                f"`{e['development_status']}`",
                f"`{e['collection_status']}`",
                _cell(e.get("collection_ignore_reason")),
                _code(e.get("manual_command")),
            ]
            for e in ignored
            + [b for b in blocked if b["collection_status"] != "ignored"]
        ],
    )

    lines += [
        "---",
        "",
        "## Archived or legacy tests",
        "",
        "Kept for historical reference under `tests/old/`. They are not repaired,",
        "renamed, or deleted as part of routine work.",
        "",
    ]
    _table(
        lines,
        ["Test file", "Subsystem", "Category", "Description", "Reason"],
        [
            [
                f"[`{e['path'].split('tests/', 1)[-1]}`]({_relative(e['path'])})",
                e["subsystem"],
                e["category"],
                _cell(e["description"]),
                _cell(e.get("collection_ignore_reason")),
            ]
            for e in archived
        ],
    )

    lines += [
        "---",
        "",
        "## Legend",
        "",
        "Three independent axes. A test can be `implemented` and still fail today;",
        "a `placeholder` can be green in CI only because it is skipped.",
        "",
        "**Execution outcome** — owned by pytest, one value per run, published only",
        "in `reports/pyrpod-pytest-report.html`:",
        "",
        "| Outcome | Meaning |",
        "| --- | --- |",
        "| passed | The test ran and every assertion held. |",
        "| failed | The test ran and an assertion or the code under test failed. |",
        "| error | The test could not run to completion (setup or teardown raised). |",
        "| skipped | The test was not executed (placeholder, or a skip condition). |",
        "",
        "**Development status** — maintained in `test_manifest.yaml`, long-lived:",
        "",
        "| Status | Meaning |",
        "| --- | --- |",
        "| `implemented` | Complete: exercises the code and asserts a result. |",
        "| `placeholder` | No assertion or verification behavior yet; skipped. |",
        "| `needs_review` | Runs real code but asserts nothing, or its purpose "
        "is insufficiently documented. |",
        "| `blocked` | Cannot run against the current architecture. |",
        "| `archived` | Superseded; kept for historical reference only. |",
        "| `deprecated` | Slated for removal. |",
        "",
        "**Collection status** — whether pytest picks the file up:",
        "",
        "| Status | Meaning |",
        "| --- | --- |",
        "| `collected` | Collected and run by pytest; appears in the HTML report. |",
        "| `manual` | Run by hand; defines no pytest tests. |",
        "| `ignored` | Listed in `collect_ignore` in `conftest.py`. |",
        "| `archived` | Under `tests/old/`, excluded from collection wholesale. |",
        "",
    ]

    return "\n".join(lines).rstrip("\n") + "\n"


def _relative(path):
    """Manifest paths are repo-relative; README links are tests/-relative."""
    return path.split("tests/", 1)[-1] if path.startswith("tests/") else path


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _report_problems(title, problems):
    print(f"\n{title}:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="validate the manifest and regenerate tests/README.md without "
        "running the test suite",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing if tests/README.md is out of date "
        "(implies --inventory-only)",
    )
    args = parser.parse_args(argv)

    try:
        entries = load_manifest()
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    problems = validate_manifest(entries)
    if problems:
        _report_problems(f"error: {MANIFEST_PATH.name} failed validation", problems)
        return 2
    print(f"manifest OK ({len(entries)} entries)")

    print("collecting pytest tests ...")
    try:
        nodes = collect_pytest_nodes()
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    collected_files = files_from_nodes(nodes)
    print(f"collected {len(nodes)} test cases in {len(collected_files)} files")

    problems = cross_check(entries, collected_files)
    if problems:
        _report_problems("error: manifest is out of sync with pytest", problems)
        return 2

    readme = render_readme(entries, nodes)
    if args.check:
        current = (
            README_PATH.read_text(encoding="utf-8") if README_PATH.is_file() else ""
        )
        if current != readme:
            print(
                f"error: {README_PATH.relative_to(REPO_ROOT)} is out of date; "
                "rerun python scripts/generate_test_dashboard.py",
                file=sys.stderr,
            )
            return 2
        print(f"{README_PATH.relative_to(REPO_ROOT)} is up to date")
        return 0

    README_PATH.write_text(readme, encoding="utf-8", newline="\n")
    print(f"wrote {README_PATH.relative_to(REPO_ROOT)}")

    if args.inventory_only:
        return 0

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"running pytest -> {REPORT_PATH.relative_to(REPO_ROOT)}")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"--html={REPORT_PATH}",
            "--self-contained-html",
        ],
        cwd=REPO_ROOT,
    )
    if REPORT_PATH.is_file():
        print(f"report written: {REPORT_PATH.relative_to(REPO_ROOT)}")
    else:
        print(
            "warning: pytest did not produce the HTML report; is pytest-html "
            "installed?",
            file=sys.stderr,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

# ========================
# PyRPOD: tests/tooling/tooling_unit_test_01.py
# ========================
# Unit tests for the test-inventory tooling in
# scripts/generate_test_dashboard.py: manifest schema validation, detection of
# missing/stale entries against real pytest collection, deterministic rendering
# of tests/README.md, and the invariant that every manifest-declared
# placeholder is actually skipped by pytest.
#
# These exercise the tooling only; they assert nothing about PyRPOD's physics.

import copy

import pytest

from scripts import generate_test_dashboard as dashboard


@pytest.fixture(scope="module")
def entries():
    """The real manifest, parsed once."""
    return dashboard.load_manifest()


def _entry(entries, **overrides):
    """A valid copy of the first manifest entry, with fields overridden."""
    entry = copy.deepcopy(entries[0])
    entry.update(overrides)
    return entry


# --------------------------------------------------------------------------
# Manifest parsing and schema validation
# --------------------------------------------------------------------------


def test_real_manifest_parses_and_validates(entries):
    """The committed manifest must always satisfy its own schema."""
    assert entries
    assert dashboard.validate_manifest(entries) == []


def test_missing_manifest_raises_manifest_error(tmp_path):
    with pytest.raises(dashboard.ManifestError, match="not found"):
        dashboard.load_manifest(tmp_path / "absent.yaml")


def test_manifest_without_tests_key_raises(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text("other: []\n", encoding="utf-8")
    with pytest.raises(dashboard.ManifestError, match="tests"):
        dashboard.load_manifest(path)


def test_missing_required_field_is_reported(entries):
    broken = _entry(entries)
    del broken["description"]
    problems = dashboard.validate_manifest([broken])
    assert any("missing required field 'description'" in p for p in problems)


def test_unknown_field_is_reported(entries):
    problems = dashboard.validate_manifest([_entry(entries, status="passed")])
    assert any("unknown field 'status'" in p for p in problems)


@pytest.mark.parametrize(
    "field, value",
    [
        ("subsystem", "propulsion"),
        ("category", "smoke"),
        ("execution_mode", "semi-automated"),
        ("development_status", "passed"),
        ("collection_status", "failed"),
    ],
)
def test_controlled_vocabulary_is_enforced(entries, field, value):
    """Runtime outcomes such as 'passed'/'failed' are never valid metadata."""
    problems = dashboard.validate_manifest([_entry(entries, **{field: value})])
    assert any(f"{field} '{value}' is not one of" in p for p in problems)


def test_nonexistent_path_is_reported(entries):
    broken = _entry(entries, path="tests/does/not/exist.py")
    problems = dashboard.validate_manifest([broken])
    assert any("file does not exist" in p for p in problems)


def test_duplicate_path_is_reported(entries):
    duplicated = [_entry(entries), _entry(entries)]
    problems = dashboard.validate_manifest(duplicated)
    assert any("duplicate path" in p for p in problems)


def test_automated_entry_must_be_collected(entries):
    broken = _entry(
        entries,
        execution_mode="automated",
        collection_status="manual",
        collection_ignore_reason="because",
    )
    problems = dashboard.validate_manifest([broken])
    assert any("requires collection_status 'collected'" in p for p in problems)


def test_uncollected_entry_requires_a_reason(entries):
    broken = _entry(
        entries,
        execution_mode="manual",
        collection_status="ignored",
        manual_command=None,
        development_status="blocked",
        collection_ignore_reason=None,
    )
    problems = dashboard.validate_manifest([broken])
    assert any("requires a collection_ignore_reason" in p for p in problems)


def test_review_required_description_forces_needs_review(entries):
    broken = _entry(
        entries,
        description=f"{dashboard.REVIEW_REQUIRED}: test purpose is not "
        "sufficiently documented.",
        development_status="implemented",
    )
    problems = dashboard.validate_manifest([broken])
    assert any("requires development_status 'needs_review'" in p for p in problems)


# --------------------------------------------------------------------------
# Cross-checking the manifest against pytest collection
# --------------------------------------------------------------------------


def test_files_from_nodes_deduplicates_parametrized_cases():
    nodes = [
        "tests/plume/plume_unit_test_02.py::test_q[0.1-0.5]",
        "tests/plume/plume_unit_test_02.py::test_q[0.2-0.5]",
        "tests/rpod/rpod_unit_test_01.py::Checks::test_stl",
    ]
    assert dashboard.files_from_nodes(nodes) == [
        "tests/plume/plume_unit_test_02.py",
        "tests/rpod/rpod_unit_test_01.py",
    ]


def test_collected_test_without_manifest_entry_is_reported(entries):
    """A newly added test file must not be silently omitted."""
    problems = dashboard.cross_check(
        entries, ["tests/rpod/rpod_unit_test_01.py", "tests/brand/new_test_01.py"]
    )
    assert any(
        "tests/brand/new_test_01.py" in p and "missing from the manifest" in p
        for p in problems
    )


def test_stale_collected_entry_is_reported(entries):
    """An entry claiming collection that pytest no longer collects is stale."""
    problems = dashboard.cross_check(entries, [])
    stale = [p for p in problems if "stale entry" in p]
    assert stale
    assert all("collection_status 'collected'" in p for p in stale)


def test_manual_entry_that_pytest_collects_is_reported(entries):
    """A manual script that starts defining pytest tests must be reclassified."""
    manual = next(e for e in entries if e["collection_status"] == "manual")
    problems = dashboard.cross_check(entries, [manual["path"]])
    assert any(
        manual["path"] in p and "the manifest says collection_status 'manual'" in p
        for p in problems
    )


def test_real_manifest_is_in_sync_with_this_session(request, entries):
    """Every test file collected in this very run is documented."""
    collected = dashboard.files_from_nodes(
        item.nodeid for item in request.session.items
    )
    documented = {e["path"] for e in entries}
    assert set(collected) <= documented


# --------------------------------------------------------------------------
# Placeholder bookkeeping
# --------------------------------------------------------------------------


def test_placeholder_status_matches_pytest_skip_markers(request, entries):
    """`development_status: placeholder` and an explicit skip must agree.

    Uses pytest's own collected items and their markers rather than parsing
    source, so it cannot drift from what the HTML report actually shows.
    """
    status_by_path = {e["path"]: e["development_status"] for e in entries}

    mismatched = []
    for item in request.session.items:
        path = dashboard.files_from_nodes([item.nodeid])[0]
        if path not in status_by_path:
            continue
        is_placeholder = status_by_path[path] == "placeholder"
        is_skipped = any(m.name == "skip" for m in item.iter_markers())
        if is_placeholder != is_skipped:
            mismatched.append((item.nodeid, is_placeholder, is_skipped))

    assert not mismatched, (
        "placeholder metadata and skip markers disagree "
        "(nodeid, manifest_says_placeholder, has_skip_marker): " + repr(mismatched)
    )


# --------------------------------------------------------------------------
# tests/README.md rendering
# --------------------------------------------------------------------------


def test_render_readme_is_deterministic(entries):
    nodes = ["tests/rpod/rpod_unit_test_01.py::Checks::test_stl"]
    assert dashboard.render_readme(entries, nodes) == dashboard.render_readme(
        entries, nodes
    )


def test_render_readme_marks_the_file_as_generated(entries):
    readme = dashboard.render_readme(entries, [])
    assert readme.startswith("<!-- AUTO-GENERATED FILE. DO NOT EDIT DIRECTLY.")
    assert "python scripts/generate_test_dashboard.py" in readme


def test_render_readme_has_every_required_section(entries):
    readme = dashboard.render_readme(entries, [])
    for heading in (
        "## Summary",
        "## Automated pytest tests",
        "## Manual verification scripts",
        "## Placeholder tests",
        "## Ignored or blocked tests",
        "## Archived or legacy tests",
        "## Legend",
    ):
        assert heading in readme


def test_render_readme_tables_carry_no_execution_outcomes(entries):
    """Execution outcomes belong to the HTML report, never to the inventory.

    Checked against the rendered table cells rather than the prose, which is
    free to explain the distinction.
    """
    readme = dashboard.render_readme(entries, [])
    inventory, legend = readme.split("## Legend", 1)

    outcomes = {"passed", "failed", "error", "skipped", "✅", "❌", "⏳"}
    offenders = [
        line
        for line in inventory.splitlines()
        if line.startswith("|")
        and outcomes & {cell.strip().strip("`").lower() for cell in line.split("|")}
    ]
    assert not offenders, offenders

    # The legend is allowed to *define* the outcome vocabulary.
    assert "passed" in legend


def test_render_readme_counts_collected_cases_per_file(entries):
    path = next(e["path"] for e in entries if e["collection_status"] == "collected")
    nodes = [f"{path}::test_a", f"{path}::test_b[1]", f"{path}::test_b[2]"]
    readme = dashboard.render_readme(entries, nodes)
    name = path.rsplit("/", 1)[-1]
    row = next(line for line in readme.splitlines() if f"`{name}`" in line)
    assert "| 3 |" in row


def test_committed_readme_matches_the_manifest(request, entries):
    """tests/README.md must be regenerated whenever the manifest changes."""
    collected = dashboard.files_from_nodes(
        item.nodeid for item in request.session.items
    )
    if len(collected) < len(
        [e for e in entries if e["collection_status"] == "collected"]
    ):
        pytest.skip("partial test selection; README comparison needs a full run")

    nodes = sorted(item.nodeid for item in request.session.items)
    expected = dashboard.render_readme(entries, nodes)
    actual = dashboard.README_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "tests/README.md is out of date; rerun "
        "python scripts/generate_test_dashboard.py"
    )

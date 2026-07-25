import os
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

CATEGORIES = ("unit", "integration", "verification")
GROUPS = ("logging", "mdao", "mission", "plume", "rpod")

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

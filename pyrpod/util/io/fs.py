import logging
import os
from pathlib import Path

# pyrpod/util/io/fs.py -> pyrpod/util/io -> pyrpod/util -> pyrpod -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHARED_DATA_DIR = _REPO_ROOT / "data"

log = logging.getLogger(__name__)

def resolve_asset_path(case_dir, subdir, filename, shared_subdir=None, *,
                       required=True):
    """
    Resolve the path to a shared data asset (STL, JFH, TCD, flight plan, ...).

    Checks the case-local copy first (case_dir/subdir/filename) so that
    case-specific files always take priority. Falls back to the repo-level
    shared `data/shared_subdir/filename` directory when no case-local copy
    exists, which is where duplicate assets get consolidated to avoid
    carrying multiple copies of the same file across cases.

    Parameters
    ----------
    case_dir : str
        Path to the case directory (as used elsewhere in pyrpod).
    subdir : str
        Case-local asset subfolder name, e.g. 'stl', 'jfh', 'tcd'.
    filename : str
        Name of the asset file.
    shared_subdir : str, optional
        Asset subfolder name under the shared `data/` directory, if it
        differs from `subdir` (e.g. flight plans live under a case's
        `jfh/` folder but under `data/flight_plan/`). Defaults to `subdir`.

    required : bool, optional
        When True (default) a missing asset raises FileNotFoundError naming
        both searched paths. Set False to get the case-local path back for a
        file that is expected not to exist yet (e.g. an output target).

    Precedence
    ----------
    The case-local copy (``case_dir/subdir/filename``) always wins over the
    shared ``data/`` copy, so a case can override a shared asset by keeping
    its own file, and two cases may carry same-named files with different
    contents. The chosen path is logged at DEBUG level.

    Returns
    -------
    str
        Path to the resolved asset (case-local if present, else shared).

    Raises
    ------
    FileNotFoundError
        If ``required`` and the asset exists in neither location.
    """
    local_path = os.path.join(case_dir, subdir, filename)
    if os.path.exists(local_path):
        log.debug("asset %r resolved to case-local %s", filename, local_path)
        return local_path

    shared_path = str(_SHARED_DATA_DIR / (shared_subdir or subdir) / filename)
    if os.path.exists(shared_path):
        log.debug("asset %r not case-local; resolved to shared %s",
                  filename, shared_path)
        return shared_path

    if required:
        raise FileNotFoundError(
            f"asset {filename!r} not found: looked for case-local "
            f"{local_path!r} then shared {shared_path!r}")
    return local_path

def ensure_dir(path):
    """
    Ensure that a directory exists. If it does not exist, create it.

    Parameters
    ----------
    path : str
        Path to the directory to ensure.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def ensure_parent_dir(file_path):
    """
    Ensure that the parent directory of a file exists. If it does not exist, create it.

    Parameters
    ----------
    file_path : str
        Path to the file whose parent directory should be ensured.
    """
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
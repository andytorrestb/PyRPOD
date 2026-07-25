"""Demonstration driver for the stl_to_vtk utility case.

Mirrors tests/rpod/rpod_unit_test_01.py (minus the assertions) and wraps the
STL -> VTK conversion in PyRPOD's operational logging. Unlike the other cases
this one does not run a plume-strike sweep; it showcases the asset-load
provenance/checksum records and directory-creation logging. Run it with:

    python case/rpod/stl_to_vtk/run.py

A runtime log is written to case/rpod/stl_to_vtk/results/logs/.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pyrpod.logging_utils import configure_logging
from pyrpod.vehicle import Vehicle
from pyrpod.mission import MissionEnvironment

CASE_DIR = _HERE.replace(os.sep, "/") + "/"


def main():
    session = configure_logging(CASE_DIR)
    try:
        # The vehicle STL (cylinder.stl) resolves from the shared data/ tree,
        # which the asset log records as source=shared-data.
        MissionEnvironment.MissionEnvironment(CASE_DIR)
        v = Vehicle.Vehicle(CASE_DIR)
        v.set_stl()             # read STL surface data (logged as an asset load)
        v.convert_stl_to_vtk()  # convert to VTK and save under results/

        session.finalize("successful")
    except Exception:
        session.finalize("failed")
        raise
    finally:
        session.close()

    print("Run complete. Runtime log:", session.log_path)


if __name__ == "__main__":
    main()

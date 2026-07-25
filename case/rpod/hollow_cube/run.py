"""Demonstration driver for the hollow_cube RPOD plume-strike case.

Mirrors tests/rpod/rpod_integration_test_04.py (minus the assertions) and wraps
the workflow in PyRPOD's operational logging. Run it with:

    python case/rpod/hollow_cube/run.py

This case's logging.ini disables console output, so records go only to the
runtime log file in case/rpod/hollow_cube/results/logs/.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pyrpod.logging_utils import configure_logging
from pyrpod.rpod import JetFiringHistory, PlumeStrikeEstimationStudy
from pyrpod.vehicle import TargetVehicle, VisitingVehicle
from pyrpod.mission import MissionEnvironment

CASE_DIR = _HERE.replace(os.sep, "/") + "/"


def main():
    session = configure_logging(CASE_DIR)
    try:
        # 1. Set up assets.
        jfh = JetFiringHistory.JetFiringHistory(CASE_DIR)
        jfh.read_jfh()

        tv = TargetVehicle.TargetVehicle(CASE_DIR)
        tv.set_stl()

        vv = VisitingVehicle.VisitingVehicle(CASE_DIR)
        vv.set_stl()
        vv.set_thruster_config()

        me = MissionEnvironment.MissionEnvironment(CASE_DIR)
        study = PlumeStrikeEstimationStudy.PlumeStrikeEstimationStudy(me)
        study.study_init(jfh, tv, vv)

        # 2. Execute the RPOD plume-strike analysis.
        study.graph_jfh()
        study.jfh_plume_strikes()

        session.finalize("successful")
    except Exception:
        session.finalize("failed")
        raise
    finally:
        session.close()

    print("Run complete. Runtime log:", session.log_path)


if __name__ == "__main__":
    main()

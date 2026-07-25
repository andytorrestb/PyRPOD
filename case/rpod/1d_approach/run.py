"""Demonstration driver for the 1d_approach RPOD plume-strike case.

Mirrors tests/rpod/rpod_integration_test_02.py (minus the assertions) and wraps
the workflow in PyRPOD's operational logging. This case models a notional VV
firing its adverse thrusters to slow down along a 1D approach. Run it with:

    python case/rpod/1d_approach/run.py

A runtime log is written to case/rpod/1d_approach/results/logs/. Logging
behavior (here, a progress record every 2 firings) is set by logging.ini.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pyrpod.logging_utils import configure_logging
from pyrpod.rpod import JetFiringHistory, PlumeStrikeEstimationStudy
from pyrpod.vehicle import LogisticsModule, TargetVehicle
from pyrpod.mission import MissionEnvironment

CASE_DIR = _HERE.replace(os.sep, "/") + "/"


def main():
    session = configure_logging(CASE_DIR)
    try:
        # 1. Set up assets.
        jfh = JetFiringHistory.JetFiringHistory(CASE_DIR)

        tv = TargetVehicle.TargetVehicle(CASE_DIR)
        tv.set_stl()

        lm = LogisticsModule.LogisticsModule(CASE_DIR)
        # Define LM mass distribution properties.
        lm.set_inertial_props(14000, 11, 2)  # mass (kg), height (m), radius (m)
        lm.set_thruster_config()
        lm.set_thruster_metrics()
        lm.assign_thruster_groups()

        me = MissionEnvironment.MissionEnvironment(CASE_DIR)
        study = PlumeStrikeEstimationStudy.PlumeStrikeEstimationStudy(me)
        study.study_init(jfh, tv, lm)

        # Uses the pre-generated 1D-approach JFH shipped with the case.
        jfh.read_jfh()

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

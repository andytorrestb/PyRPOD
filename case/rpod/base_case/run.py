"""Demonstration driver for the base_case RPOD plume-strike case.

Mirrors tests/rpod/rpod_integration_test_01.py (minus the assertions) and wraps
the workflow in PyRPOD's operational logging. Run it directly from anywhere:

    python case/rpod/base_case/run.py

Logging is configured from this case's logging.ini. A fresh runtime log is
written to:

    case/rpod/base_case/results/logs/base_case_<timestamp>.log

along with a config.ini snapshot and SHA-256 checksums of every input asset.
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

# PyRPOD's case_dir convention is a directory path with a trailing separator.
CASE_DIR = _HERE.replace(os.sep, "/") + "/"


def main():
    session = configure_logging(CASE_DIR)
    try:
        # 1. Set up assets.
        jfh = JetFiringHistory.JetFiringHistory(CASE_DIR)

        tv = TargetVehicle.TargetVehicle(CASE_DIR)
        tv.set_stl()

        lm = LogisticsModule.LogisticsModule(CASE_DIR)
        lm.set_thruster_config()

        me = MissionEnvironment.MissionEnvironment(CASE_DIR)
        study = PlumeStrikeEstimationStudy.PlumeStrikeEstimationStudy(me)
        study.study_init(jfh, tv, lm)

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

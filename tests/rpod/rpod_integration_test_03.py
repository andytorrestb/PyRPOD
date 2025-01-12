# Andy Torres
# University of Central Florida
# Department of Mechanical and Aerospace Engineering
# Last Changed: 03-16-24

# ========================
# PyRPOD: tests/rpod/rpod_integration_test_03.py
# ========================
# Test case to analyze Keep Out Zone Impingement. (WIP)


import test_header
import unittest, os, sys
from pyrpod import LogisticsModule, MissionPlanner

class KeepOutZoneChecks(unittest.TestCase):
    def test_keep_out_zone(self):

        # # set case directory
        # case_dir = '../case/rpod/flight_envelopes/'

        # # Instantiate LogisticModule object.
        # lm = LogisticsModule.LogisticsModule(case_dir)

        # # Define LM mass distrubtion properties.
        # m = 0.45*30000 # lb converted to kg
        # h = 14 # m
        # r = 4.0/2.0 # m
        # lm.set_inertial_props(m, h, r)

        # # Load in thruster configuration data from text file
        # lm.set_thruster_config()

        # # Draco/Hypergolic thrusters.
        # lm.add_thruster_performance(400, 300)
        # lm.assign_thruster_groups()

        # mp = MissionPlanner.MissionPlanner(case_dir)
        # mp.set_lm(lm)
        # mp.read_flight_plan()

        return


if __name__ == '__main__':
    unittest.main()
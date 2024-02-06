# Andy Torres, Pearce Patterson, Nicholas A. Palumbo
# University of Central Florida
# Department of Mechanical and Aerospace Engineering
# Last Changed: 1-29-24

# ========================
# PyRPOD: test/test_case_19.py
# ========================
# Test case for producing hollow cube data.

import test_header
import configparser
import unittest, os, sys
from pyrpod import JetFiringHistory, TargetVehicle, VisitingVehicle, RPOD

class LoadJFHChecks(unittest.TestCase):
    def test_hollow_cube(self):

        # Path to directory holding data assets and results for a specific RPOD study.
        case_dir = '../case/hollow_cube/'

        #Check for JFH data
        jfh_dir = case_dir + 'jfh/'
        files_in_folder = os.listdir(jfh_dir)
        self.assertTrue(files_in_folder, f"Folder {jfh_dir} is empty")

        # Load JFH data.
        jfh = JetFiringHistory.JetFiringHistory(case_dir)
        jfh.read_jfh()

        # Check if target vehicle stl is there
        config = configparser.ConfigParser()
        config.read(case_dir + 'config.ini')
        tv_name = config['stl']['tv']
        file_path = os.path.join(case_dir + 'stl/', tv_name)
        assert os.path.exists(file_path), f"File {tv_name} does not exist in the directory {case_dir + 'stl/'}"

        # Load Target Vehicle.
        tv = TargetVehicle.TargetVehicle(case_dir)
        tv.set_stl()

        # Check if visiting vehicle stl is there
        config = configparser.ConfigParser()
        config.read(case_dir + 'config.ini')
        vv_name = config['stl']['vv']
        file_path = os.path.join(case_dir + 'stl/', vv_name)
        assert os.path.exists(file_path), f"File {vv_name} does not exist in the directory {case_dir + 'stl/'}"

        # Load Visiting Vehicle.
        vv = VisitingVehicle.VisitingVehicle(case_dir)
        vv.set_stl()
        vv.set_thruster_config()
        # vv.set_thruster_metrics()

        # Initiate RPOD study.
        rpod = RPOD.RPOD(case_dir)
        rpod.study_init(jfh, tv, vv)

        # Run plume strike analysis
        rpod.graph_jfh()
        rpod.jfh_plume_strikes()

if __name__ == '__main__':
    unittest.main()
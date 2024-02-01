# Andy Torres
# University of Central Florida
# Department of Mechanical and Aerospace Engineering
# Last Changed: 12-14-23

# ========================
# PyRPOD: test/test_case_13.py
# ========================
# Test case for converting STL data to VTK data.
# This is accomplished by checking for the proper data format of VTK files.

import test_header
import unittest, os, sys
from pyrpod import Vehicle
from pyrpod import MissionPlanner

class STLtoVTKChecks(unittest.TestCase):
    def test_delta_m_plots(self):

        # Path to directory holding data assets and results for a specific RPOD study.
        case_dir = '../case/stl_to_vtk/'

        # Load configuration data for RPOD analysis.
        mp = MissionPlanner.MissionPlanner(case_dir)

        # Use Vehicle Object to read STL surface data.
        v = Vehicle.Vehicle(case_dir)
        v.set_stl()

        # Convert to VTK and save in case directory.
        v.convert_stl_to_vtk()

        # Check if the folder exists
        self.assertTrue(os.path.exists(case_dir + 'results/'))

        # Check that folder is not empty
        files_in_folder = os.listdir(case_dir + 'results/')
        self.assertTrue(files_in_folder, f"Folder {case_dir + 'results/'} is empty")

        # Check that the file type is correct
        for file_name in files_in_folder:
            self.assertTrue(file_name.endswith('.vtu'), f"File {file_name} in folder {case_dir + 'results/'} does not have '.vtu' extension")



if __name__ == '__main__':
    unittest.main()
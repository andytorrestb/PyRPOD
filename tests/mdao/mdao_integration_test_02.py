# ========================
# PyRPOD: tests/mdao/mdao_integration_test_02.py
# ========================
# Baseline flat-plate case run end to end through the package-level trade
# study API:
#
#     study = TradeStudy.from_config('.../flat_plate_baseline.yaml')
#     results = study.run()
#
# The case is the Cai 2016 Section-4 geometry reframed flat
# (case/plume/plume_flat_plate_sweep): argon round jet, D = 1 m, S0 = 2.0,
# T0 = 200 K, Tw = 300 K, fully diffuse, 8 m x 8 m plate, source head-on at
# L = 4D. Checked here:
#
#   * exactly one case x component x firing, with a Jet Firing History
#     holding exactly the requested single entry;
#   * the standard per-face strike VTK is written at the path the result
#     record advertises, carrying the pipeline's own cell fields;
#   * the machine-readable summary (CSV) and metadata (JSON) are written and
#     carry enough provenance to reproduce the run;
#   * the integrated loads are physically right for a head-on pose: force
#     along the plume direction, no in-plane resultant, center of pressure at
#     the plate center, every face struck;
#   * the integrated normal load agrees with the INDEPENDENT Cai 2016 exact
#     reference (pyrpod/plume/CaiImpingement2016.py, Eq. 15 quadrature) to
#     within the documented Maxwellian-chain gap -- compared through the
#     generic reference-data interface, not against PyRPOD's own output.
#
# The study writes into a temporary directory, so a test run leaves no
# artifacts in the repository.
#
# Run:  python -m pytest mdao/mdao_integration_test_02.py -s   (from tests/)

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyrpod.mdao.TradeStudy import TradeStudy
from pyrpod.mdao.reference_data import ReferenceDataset, ReferenceRecord
from pyrpod.plume import CaiImpingement2016 as cai

_TESTS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = _TESTS_DIR.parent / 'case' / 'plume' / 'plume_flat_plate_sweep'
CONFIG_PATH = CASE_DIR / 'study' / 'flat_plate_baseline.yaml'

# Paper conditions of the case (and of the committed configuration).
S_0 = 2.0
EPS = 1.5                     # Tw / T0
R_0 = 0.5                     # nozzle radius (m)
PLATE_SEMI = 4.0              # 8 m x 8 m plate
PLATE_AREA = 64.0
Q_DYN = 1.1044652197738332    # n0*m*U0^2/2 (Pa)
Q_DYN_HEAT = 637.3520362956127  # n0*m*U0^3/2 (W/m^2)


def cai_reference(alpha_deg, distance):
    """Exact Eq.-15 plate averages for one pose, as dimensional loads.

    The sweep angle maps onto the paper's inclination as
    alpha_paper = 90 deg - |alpha|. The plate's faces all share the +Z
    normal, so the exact pressure average CP becomes a pure normal force
    -CP * q_dyn * S, and the average heat-flux coefficient CQ becomes a
    component heat load CQ * q_heat * S.
    """
    coefficients = cai.averaged_coefficients(
        S_0, np.deg2rad(90.0 - abs(alpha_deg)), EPS, R_0, distance,
        PLATE_SEMI, PLATE_SEMI)
    return {
        'pressure_force': [0.0, 0.0,
                           -float(coefficients['CP']) * Q_DYN * PLATE_AREA],
        'total_heat_load': float(coefficients['CQ']) * Q_DYN_HEAT * PLATE_AREA,
    }


class FlatPlateBaselineStudy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp(prefix='pyrpod_baseline_study_')
        cls.study = TradeStudy.from_config(CONFIG_PATH,
                                           output_dir=cls.output_dir)
        cls.results = cls.study.run()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir, ignore_errors=True)

    # ------------------------------------------------------------ structure
    def test_one_case_one_component_one_firing(self):
        self.assertEqual(len(self.results), 1)
        case = self.results.cases[0]
        self.assertEqual(case.study_name, 'cai2016_flat_plate_baseline')
        self.assertEqual(case.component, 'plate')
        self.assertEqual(case.firing_id, 1)
        self.assertEqual(case.plate_angle_deg, 0.0)
        self.assertEqual(case.source_distance, 4.0)
        self.assertEqual(case.plume_model, 'SimplifiedGasKinetics')
        self.assertEqual(case.mesh_faces, case.component_faces)

    def test_jfh_holds_exactly_the_requested_firing_count(self):
        case = self.results.cases[0]
        self.assertTrue(os.path.isfile(case.jfh_path))
        lines = [line for line in
                 Path(case.jfh_path).read_text(encoding='utf-8').splitlines()
                 if line.strip()]
        # header + unused second line + one firing row
        self.assertEqual(len(lines), 3)
        self.assertIn('1', lines[0].split())

    def test_source_pose_is_head_on_at_the_configured_distance(self):
        case = self.results.cases[0]
        np.testing.assert_allclose(case.plume_source_position,
                                   [0.0, 0.0, 4.0], atol=1e-9)
        # DCM first column is the thruster axis: aimed back at the plate.
        dcm = np.asarray(case.plume_source_orientation).reshape(3, 3)
        np.testing.assert_allclose(dcm[:, 0], [0.0, 0.0, -1.0], atol=1e-9)

    # ------------------------------------------------------------ artifacts
    def test_vtk_is_written_with_the_standard_per_face_fields(self):
        case = self.results.cases[0]
        self.assertIsNotNone(case.vtk_path)
        self.assertTrue(os.path.isfile(case.vtk_path),
                        msg=f'missing VTK artifact {case.vtk_path}')

        head = Path(case.vtk_path).read_bytes()[:8000].decode('utf-8',
                                                              'replace')
        for field in ('strikes', 'cum_strikes', 'pressures', 'max_pressures',
                      'shear_stress', 'max_shears', 'heat_flux_rate',
                      'heat_flux_load', 'cum_heat_flux_load'):
            self.assertIn(f'Name="{field}"', head,
                          msg=f'per-face field {field} missing from the VTK')

    def test_summary_csv_and_metadata_are_written(self):
        csv_path = self.results.summary_csv_path
        metadata_path = self.results.metadata_path
        self.assertTrue(os.path.isfile(csv_path))
        self.assertTrue(os.path.isfile(metadata_path))

        rows = Path(csv_path).read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(rows), 2)                  # header + one case
        for column in ('force_z', 'moment_y', 'center_of_pressure_x',
                       'max_pressure', 'affected_area', 'coeff_CF',
                       'vtk_path'):
            self.assertIn(column, rows[0])

        document = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
        self.assertEqual(document['n_cases'], 1)
        provenance = document['provenance']
        self.assertEqual(provenance['plume_model'], 'SimplifiedGasKinetics')
        self.assertEqual(provenance['geometry_id'],
                         'flat_plate_transformed.stl')
        self.assertEqual(provenance['n_firings_per_pose'], 1)
        self.assertEqual(provenance['total_firings'], 1)
        self.assertEqual(provenance['sweep_mode'], 'per_case')
        self.assertEqual(provenance['config_path'], str(CONFIG_PATH))
        self.assertIn('code_version', provenance)
        self.assertIn('known_limitations', provenance)
        self.assertIn('units', document['cases'][0])

    # -------------------------------------------------------------- physics
    def test_head_on_loads_are_physically_consistent(self):
        case = self.results.cases[0]
        force = np.asarray(case.force)

        # The plume travels along -Z here, so the plate is pushed along -Z.
        self.assertLess(force[2], 0.0)
        self.assertLess(abs(force[0]), 1e-6 * abs(force[2]))
        self.assertLess(abs(force[1]), 1e-6 * abs(force[2]))
        # Symmetric pose: no resultant moment about the plate center.
        self.assertLess(case.moment_magnitude, 1e-6 * abs(force[2]))

        self.assertEqual(case.center_of_pressure_status, 'ok')
        np.testing.assert_allclose(case.center_of_pressure,
                                   [0.0, 0.0, 0.0], atol=1e-6)

        # The whole plate sits inside the gating wedge at this pose.
        self.assertEqual(case.struck_faces, case.component_faces)
        self.assertAlmostEqual(case.affected_area, PLATE_AREA, places=6)
        self.assertGreater(case.max_pressure, 0.0)
        self.assertGreater(case.max_heat_flux, 0.0)

    def test_coefficients_are_available_with_the_configured_normalization(self):
        case = self.results.cases[0]
        self.assertTrue(case.coefficients_available)
        for name in ('CF', 'CFz', 'CM', 'Cp_max', 'Cf_max', 'Cq_max'):
            self.assertIn(name, case.coefficients)
        self.assertAlmostEqual(
            case.coefficients['CF'],
            case.force_magnitude / (Q_DYN * PLATE_AREA), places=12)

    # ------------------------------------------- independent Cai reference
    def test_matches_the_independent_cai_reference(self):
        dataset = ReferenceDataset(
            label='Cai 2016 exact (Eq. 15)',
            source='pyrpod/plume/CaiImpingement2016.py',
            records=[ReferenceRecord(
                key={'plate_angle_deg': 0.0, 'source_distance': 4.0},
                quantities=cai_reference(0.0, 4.0))])

        report = self.study.compare(dataset)
        self.assertEqual(report.unmatched_cases, [])
        by_name = {entry.quantity: entry for entry in report.comparisons}
        self.assertEqual(set(by_name), {'pressure_force', 'total_heat_load'})

        force_error = by_name['pressure_force'].relative_error
        heat_error = by_name['total_heat_load'].relative_error
        print(f'[baseline] integrated normal load vs Cai 2016 exact: '
              f'{force_error:.3%} relative error')
        print(f'[baseline] component heat load vs Cai 2016 exact: '
              f'{heat_error:.3%} relative error')

        # Documented accuracy of the Maxwellian engineering chain against the
        # exact collisionless solution (plate-averaged CP tracks the
        # reference to a few percent, the heat flux to ~15%). A convention
        # or sign regression would be O(1) here.
        self.assertLess(force_error, 0.08)
        self.assertLess(heat_error, 0.25)

        report_path = Path(self.output_dir) / 'cai_reference_comparison.csv'
        report.write_csv(report_path)
        self.assertTrue(report_path.is_file())


if __name__ == '__main__':
    unittest.main()

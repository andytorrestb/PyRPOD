# ========================
# PyRPOD: tests/mdao/mdao_integration_test_03.py
# ========================
# Multi-angle / multi-distance flat-plate sweep through the package-level
# trade study API, plus the architecture checks that keep the study honest
# for non-plate geometry.
#
# The committed sweep configuration
# (case/plume/plume_flat_plate_sweep/study/flat_plate_sweep.yaml) covers 19
# angles x 5 distances; this test runs a REDUCED subset of the same
# configuration (5 angles x 2 distances = 10 cases) so the automated suite
# stays quick -- the exact-reference quadrature, not the pipeline, dominates
# the runtime. What is checked:
#
#   * one structured result per angle x distance combination, each with
#     exactly the requested number of JFH entries;
#   * mirror symmetry in +/- angle, which a paper-frame convention error
#     would break as O(1);
#   * monotonic decay of the integrated load with source distance, and
#     center-of-pressure travel with approach angle;
#   * agreement with the INDEPENDENT Cai 2016 exact reference across the
#     sweep, through the generic reference-data interface;
#   * per-case VTK artifacts land in per-case directories (sweep cases never
#     overwrite one another);
#   * the optional trend plots can be generated (into a temporary directory,
#     so no artifact reaches the repository);
#   * the same machinery runs against the CYLINDER case -- nothing in the
#     study assumes a flat target -- and correctly reports coefficients as
#     unavailable when the configuration supplies no normalization inputs.
#
# Run:  python -m pytest mdao/mdao_integration_test_03.py -s   (from tests/)

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from pyrpod.mdao.TradeStudy import TradeStudy
from pyrpod.mdao.plume_validation import PlumeValidationStudy
from pyrpod.mdao.reference_data import (
    ReferenceDataset,
    ReferenceRecord,
    compare_results,
)
from pyrpod.mdao.study_config import StudyConfig
from pyrpod.plume import CaiImpingement2016 as cai

_TESTS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = _TESTS_DIR.parent / 'case' / 'plume' / 'plume_flat_plate_sweep'
SWEEP_CONFIG = CASE_DIR / 'study' / 'flat_plate_sweep.yaml'
CYLINDER_CONFIG = (_TESTS_DIR.parent / 'case' / 'plume'
                   / 'plume_cylinder_sweep' / 'study'
                   / 'cylinder_baseline.yaml')

ANGLES = [-40.0, -20.0, 0.0, 20.0, 40.0]
DISTANCES = [2.0, 6.0]

S_0 = 2.0
EPS = 1.5
R_0 = 0.5
PLATE_SEMI = 4.0
PLATE_AREA = 64.0
Q_DYN = 1.1044652197738332
Q_DYN_HEAT = 637.3520362956127


def reduced_sweep_config(output_dir, angles=ANGLES, distances=DISTANCES,
                         write_vtk=False, write_plots=False):
    """The committed sweep configuration, restricted to a small subset."""
    data = yaml.safe_load(SWEEP_CONFIG.read_text(encoding='utf-8'))
    data['sweep']['plate_angles_deg'] = list(angles)
    data['sweep']['source_distances'] = list(distances)
    data['output']['vtk']['enabled'] = write_vtk
    data['output']['plots']['enabled'] = write_plots
    data['study']['output_dir'] = str(output_dir)
    return StudyConfig.from_mapping(data, source_path=str(SWEEP_CONFIG))


def cai_reference_dataset(angles, distances):
    """Exact Cai 2016 plate averages as dimensional reference records.

    Independent of PyRPOD's strike pipeline: the Eq.-15 quadrature of
    pyrpod/plume/CaiImpingement2016.py evaluated at alpha_paper = 90 - |alpha|,
    converted to a normal pressure force and a component heat load with the
    case's own normalization. Mirror-invariant in +/- alpha, so it is cached
    per |alpha|.
    """
    cache = {}
    records = []
    for distance in distances:
        for angle in angles:
            key = (abs(angle), distance)
            if key not in cache:
                cache[key] = cai.averaged_coefficients(
                    S_0, np.deg2rad(90.0 - abs(angle)), EPS, R_0, distance,
                    PLATE_SEMI, PLATE_SEMI)
            coefficients = cache[key]
            records.append(ReferenceRecord(
                key={'plate_angle_deg': float(angle),
                     'source_distance': float(distance),
                     'component': 'plate'},
                quantities={
                    'pressure_force': [
                        0.0, 0.0,
                        -float(coefficients['CP']) * Q_DYN * PLATE_AREA],
                    'total_heat_load': (float(coefficients['CQ'])
                                        * Q_DYN_HEAT * PLATE_AREA)}))
    return ReferenceDataset(label='Cai 2016 exact (Eq. 15)',
                            source='pyrpod/plume/CaiImpingement2016.py',
                            records=records)


class FlatPlateSweep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp(prefix='pyrpod_sweep_study_')
        cls.config = reduced_sweep_config(cls.output_dir)
        cls.study = PlumeValidationStudy(cls.config)
        cls.results = cls.study.run()
        cls.by_pose = {(case.plate_angle_deg, case.source_distance): case
                       for case in cls.results.cases}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir, ignore_errors=True)

    def test_every_angle_distance_combination_produced_a_result(self):
        self.assertEqual(len(self.results), len(ANGLES) * len(DISTANCES))
        self.assertEqual(set(self.by_pose),
                         {(angle, distance) for angle in ANGLES
                          for distance in DISTANCES})
        for case in self.results.cases:
            self.assertEqual(case.component, 'plate')
            self.assertEqual(case.firing_id, 1)
            self.assertEqual(case.study_name, 'cai2016_flat_plate_sweep')

    def test_each_case_wrote_its_own_jfh_with_the_exact_entry_count(self):
        seen = set()
        for case in self.results.cases:
            self.assertTrue(os.path.isfile(case.jfh_path))
            self.assertNotIn(case.jfh_path, seen)
            seen.add(case.jfh_path)
            header = Path(case.jfh_path).read_text(
                encoding='utf-8').splitlines()[0]
            self.assertEqual(int(header.split()[1]),
                             self.config.sweep.n_firings)

    def test_results_are_mirror_symmetric_in_plus_minus_angle(self):
        for distance in DISTANCES:
            for angle in (20.0, 40.0):
                positive = self.by_pose[(angle, distance)]
                negative = self.by_pose[(-angle, distance)]
                scale = abs(positive.force[2])

                # Normal load and heat load are mirror invariant.
                self.assertAlmostEqual(positive.force[2], negative.force[2],
                                       delta=1e-6 * scale)
                self.assertAlmostEqual(positive.total_heat_load,
                                       negative.total_heat_load,
                                       delta=1e-6 * positive.total_heat_load)
                # The in-plane resultant mirrors.
                self.assertAlmostEqual(positive.force[0], -negative.force[0],
                                       delta=1e-6 * scale)
                # ... and so does the center of pressure.
                self.assertAlmostEqual(positive.center_of_pressure[0],
                                       -negative.center_of_pressure[0],
                                       delta=1e-6 * PLATE_SEMI)

    def test_out_of_plane_resultant_vanishes_by_symmetry(self):
        for case in self.results.cases:
            self.assertLess(abs(case.force[1]), 1e-6 * abs(case.force[2]))

    def test_load_decays_with_source_distance(self):
        for angle in ANGLES:
            near = self.by_pose[(angle, min(DISTANCES))]
            far = self.by_pose[(angle, max(DISTANCES))]
            self.assertLess(abs(far.force[2]), abs(near.force[2]))
            self.assertLess(far.max_pressure, near.max_pressure)
            self.assertLess(far.max_heat_flux, near.max_heat_flux)

    def test_center_of_pressure_moves_with_approach_angle(self):
        for distance in DISTANCES:
            head_on = self.by_pose[(0.0, distance)]
            inclined = self.by_pose[(-40.0, distance)]
            self.assertEqual(head_on.center_of_pressure_status, 'ok')
            self.assertEqual(inclined.center_of_pressure_status, 'ok')
            np.testing.assert_allclose(head_on.center_of_pressure[:2],
                                       [0.0, 0.0], atol=1e-6)
            self.assertGreater(
                abs(inclined.center_of_pressure[0]
                    - head_on.center_of_pressure[0]), 1e-3)

    def test_sweep_agrees_with_the_independent_cai_reference(self):
        dataset = cai_reference_dataset(ANGLES, DISTANCES)
        report = compare_results(self.results.cases, dataset,
                                 reference_length=PLATE_SEMI)
        self.assertEqual(report.unmatched_cases, [])
        self.assertEqual(len(report.comparisons),
                         2 * len(ANGLES) * len(DISTANCES))

        for quantity, envelope in (('pressure_force', 0.08),
                                   ('total_heat_load', 0.25)):
            errors = [entry.relative_error
                      for entry in report.for_quantity(quantity)]
            worst, mean = max(errors), sum(errors) / len(errors)
            print(f'[sweep] {quantity} vs Cai 2016 exact: max '
                  f'{worst:.3%}, mean {mean:.3%} over {len(errors)} poses')
            # Documented Maxwellian-chain gap, not a physics tolerance: a
            # convention regression would be O(1) rather than a few percent.
            self.assertLess(mean, envelope)

        report_path = Path(self.output_dir) / 'cai_reference_comparison.csv'
        report.write_csv(report_path)
        self.assertTrue(report_path.is_file())

    def test_summary_artifacts_cover_every_case(self):
        rows = Path(self.results.summary_csv_path).read_text(
            encoding='utf-8').splitlines()
        self.assertEqual(len(rows), len(self.results) + 1)
        self.assertTrue(os.path.isfile(self.results.metadata_path))


class SweepArtifacts(unittest.TestCase):
    """VTK export and optional plots, on a two-case slice of the sweep."""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp(prefix='pyrpod_sweep_artifacts_')
        cls.config = reduced_sweep_config(cls.output_dir,
                                          angles=[-30.0, 30.0],
                                          distances=[4.0], write_vtk=True)
        cls.study = PlumeValidationStudy(cls.config)
        cls.results = cls.study.run()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir, ignore_errors=True)

    def test_each_case_writes_its_own_vtk(self):
        paths = [case.vtk_path for case in self.results.cases]
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(set(paths)), 2,
                         msg='sweep cases must not share a VTK path')
        for path in paths:
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(Path(path).name.startswith('firing-'))

    def test_optional_plots_are_generated_on_request(self):
        # Plot generation is optional: nothing above needed it. Here it is
        # exercised explicitly, writing into the temporary output directory.
        paths = self.study.plot()
        self.assertTrue(paths)
        for path in paths:
            self.assertTrue(os.path.isfile(path))
        names = {Path(path).name for path in paths}
        self.assertIn('force_vs_angle.png', names)
        self.assertIn('moment_vs_angle.png', names)
        self.assertIn('heat_flux_vs_angle.png', names)


class CylinderTargetSupport(unittest.TestCase):
    """The study architecture does not assume a flat plate."""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp(prefix='pyrpod_cylinder_study_')
        cls.study = TradeStudy.from_config(CYLINDER_CONFIG,
                                           output_dir=cls.output_dir)
        cls.results = cls.study.run()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir, ignore_errors=True)

    def test_curved_closed_target_runs_end_to_end(self):
        self.assertEqual(len(self.results), 1)
        case = self.results.cases[0]
        self.assertEqual(case.component, 'cylinder')
        self.assertGreater(case.mesh_faces, 0)
        self.assertGreater(case.struck_faces, 0)
        self.assertLess(case.struck_faces, case.mesh_faces,
                        msg='a closed target must not be struck all over '
                            'from a single pose')
        self.assertGreater(case.affected_area, 0.0)
        self.assertLess(case.affected_area, case.component_area)
        self.assertTrue(os.path.isfile(case.vtk_path))

    def test_coefficients_are_unavailable_without_normalization_inputs(self):
        case = self.results.cases[0]
        self.assertFalse(case.coefficients_available)
        self.assertEqual(case.coefficients, {})
        row = case.to_row()
        self.assertNotIn('coeff_CF', row)

    def test_center_of_pressure_is_reported_with_a_status(self):
        case = self.results.cases[0]
        self.assertIn(case.center_of_pressure_status,
                      ('ok', 'ill_conditioned', 'zero_load'))
        if case.center_of_pressure_status == 'ok':
            # Whatever it is, it must reproduce the reported moment.
            arm = (np.asarray(case.center_of_pressure)
                   - np.asarray(case.moment_reference_point))
            np.testing.assert_allclose(
                np.cross(arm, case.force), case.moment,
                atol=1e-6 * max(1.0, case.moment_magnitude))
        else:
            self.assertIsNone(case.center_of_pressure)


if __name__ == '__main__':
    unittest.main()

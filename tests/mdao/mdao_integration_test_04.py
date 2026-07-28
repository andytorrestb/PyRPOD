# ========================
# PyRPOD: tests/mdao/mdao_integration_test_04.py
# ========================
# The single-history sweep engine (pyrpod.mdao.parameter_sweep.
# ParameterSweepStudy, sweep.mode: single_jfh): the whole angle x distance
# sweep runs as ONE case driven by ONE Jet Firing History, so every firing's
# strikes come from the same pipeline run.
#
# The committed configuration
# (case/plume/plume_flat_plate_sweep/study/flat_plate_sweep_single_jfh.yaml)
# covers 19 angles x 5 distances; this test runs a REDUCED subset of it so
# the automated suite stays quick. What is checked:
#
#   * TradeStudy.from_config dispatches on sweep.mode -- single_jfh binds the
#     sweep engine, per_case the original one;
#   * exactly ONE Jet Firing History is written, holding exactly
#     len(poses) x n_firings entries, and one result record per firing;
#   * every record shares one case_id but carries the pose it realizes, with
#     firings numbered continuously through the history;
#   * EQUIVALENCE: per-firing loads are identical to the per-case engine's,
#     so the two decompositions are numerically interchangeable;
#   * the strike VTK files form a single results/strikes/firing-<i>.vtu
#     series (the repository's sweep convention), not one folder per case;
#   * the sweep envelope -- worst pressure/shear over every pose, accumulated
#     heat-flux load, coverage -- is recorded per component, and is absent
#     (not fabricated) when the full pipeline path is disabled;
#   * the same independent Cai 2016 reference data compares against this
#     engine's results unchanged.
#
# Run:  python -m pytest mdao/mdao_integration_test_04.py -s   (from tests/)

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from pyrpod.mdao.TradeStudy import TradeStudy
from pyrpod.mdao.parameter_sweep import ParameterSweepStudy
from pyrpod.mdao.plume_validation import PlumeValidationStudy
from pyrpod.mdao.reference_data import compare_results
from pyrpod.mdao.study_config import StudyConfig

from mdao_integration_test_03 import (  # noqa: E402  (sibling test module)
    ANGLES,
    DISTANCES,
    PLATE_SEMI,
    cai_reference_dataset,
)

_TESTS_DIR = Path(__file__).resolve().parents[1]
CASE_DIR = _TESTS_DIR.parent / 'case' / 'plume' / 'plume_flat_plate_sweep'
SINGLE_JFH_CONFIG = CASE_DIR / 'study' / 'flat_plate_sweep_single_jfh.yaml'
PER_CASE_CONFIG = CASE_DIR / 'study' / 'flat_plate_sweep.yaml'


def reduced_config(source, output_dir, angles=ANGLES, distances=DISTANCES,
                   write_vtk=False, n_firings=1):
    """A committed configuration restricted to a small subset of poses."""
    data = yaml.safe_load(Path(source).read_text(encoding='utf-8'))
    data['sweep']['plate_angles_deg'] = list(angles)
    data['sweep']['source_distances'] = list(distances)
    data['sweep']['n_firings'] = n_firings
    data['output']['vtk']['enabled'] = write_vtk
    data['output']['plots']['enabled'] = False
    data['study']['output_dir'] = str(output_dir)
    return StudyConfig.from_mapping(data, source_path=str(source))


class EngineSelection(unittest.TestCase):

    def test_sweep_mode_selects_the_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            single = TradeStudy.from_config(SINGLE_JFH_CONFIG, output_dir=tmp)
            per_case = TradeStudy.from_config(PER_CASE_CONFIG, output_dir=tmp)
        self.assertIsInstance(single.validation_study, ParameterSweepStudy)
        self.assertIsInstance(per_case.validation_study, PlumeValidationStudy)
        self.assertEqual(single.study_config.sweep.mode, 'single_jfh')
        self.assertEqual(per_case.study_config.sweep.mode, 'per_case')


class SingleHistorySweep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp(prefix='pyrpod_single_jfh_')
        cls.config = reduced_config(SINGLE_JFH_CONFIG, cls.output_dir)
        cls.study = ParameterSweepStudy(cls.config)
        cls.results = cls.study.run()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir, ignore_errors=True)

    def test_one_history_holds_every_pose(self):
        expected = len(ANGLES) * len(DISTANCES)
        self.assertEqual(self.config.sweep.total_firings, expected)
        self.assertEqual(len(self.results), expected)

        jfh_paths = {case.jfh_path for case in self.results.cases}
        self.assertEqual(len(jfh_paths), 1, msg='the sweep must write ONE JFH')
        jfh_path = jfh_paths.pop()
        self.assertTrue(os.path.isfile(jfh_path))

        lines = [line for line in
                 Path(jfh_path).read_text(encoding='utf-8').splitlines()
                 if line.strip()]
        # header + unused second line + one row per firing
        self.assertEqual(len(lines), expected + 2)
        self.assertEqual(int(lines[0].split()[1]), expected)

    def test_records_share_a_case_but_carry_their_own_pose(self):
        case_ids = {case.case_id for case in self.results.cases}
        self.assertEqual(case_ids, {self.study.case_id})
        self.assertEqual([case.firing_id for case in self.results.cases],
                         list(range(1, len(self.results) + 1)))
        self.assertEqual(
            [(case.plate_angle_deg, case.source_distance)
             for case in self.results.cases],
            [pose for pose in self.config.sweep.poses])

    def test_multiple_firings_per_pose_extend_the_same_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = reduced_config(SINGLE_JFH_CONFIG, tmp,
                                    angles=[-20.0, 20.0], distances=[4.0],
                                    n_firings=3)
            results = ParameterSweepStudy(config).run()

        self.assertEqual(len(results), 6)          # 2 poses x 3 firings
        self.assertEqual([case.plate_angle_deg for case in results.cases],
                         [-20.0] * 3 + [20.0] * 3)
        self.assertEqual({case.jfh_path for case in results.cases}.__len__(), 1)

    def test_per_firing_loads_match_the_per_case_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            per_case = PlumeValidationStudy(
                reduced_config(PER_CASE_CONFIG, tmp)).run()

        by_pose = {(case.plate_angle_deg, case.source_distance): case
                   for case in per_case.cases}
        self.assertEqual(len(by_pose), len(self.results))

        for case in self.results.cases:
            reference = by_pose[(case.plate_angle_deg, case.source_distance)]
            np.testing.assert_array_equal(case.force, reference.force)
            np.testing.assert_array_equal(case.moment, reference.moment)
            self.assertEqual(case.max_pressure, reference.max_pressure)
            self.assertEqual(case.max_heat_flux, reference.max_heat_flux)
            self.assertEqual(case.struck_faces, reference.struck_faces)

    def test_envelope_is_absent_without_the_full_pipeline(self):
        # This run has VTK disabled, so the pipeline's cumulative arrays were
        # never produced; the envelope must be empty rather than invented.
        self.assertEqual(self.study.envelope, {})
        self.assertNotIn('sweep_envelope', self.results.provenance)

    def test_metadata_records_the_sweep_mode_and_history(self):
        provenance = self.results.provenance
        self.assertEqual(provenance['sweep_mode'], 'single_jfh')
        self.assertEqual(provenance['n_firings_per_pose'], 1)
        self.assertEqual(provenance['total_firings'],
                         len(ANGLES) * len(DISTANCES))
        self.assertTrue(os.path.isfile(provenance['jfh_path']))

    def test_independent_cai_reference_compares_unchanged(self):
        dataset = cai_reference_dataset(ANGLES, DISTANCES)
        report = compare_results(self.results.cases, dataset,
                                 reference_length=PLATE_SEMI)
        self.assertEqual(report.unmatched_cases, [])

        errors = [entry.relative_error
                  for entry in report.for_quantity('pressure_force')]
        worst, mean = max(errors), sum(errors) / len(errors)
        print(f'[single-jfh] pressure_force vs Cai 2016 exact: max '
              f'{worst:.3%}, mean {mean:.3%} over {len(errors)} poses')
        self.assertLess(mean, 0.08)


class SingleHistoryArtifacts(unittest.TestCase):
    """VTK layout and the sweep envelope, on a three-pose slice."""

    ANGLES = [-30.0, 0.0, 30.0]

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp(prefix='pyrpod_single_jfh_vtk_')
        cls.config = reduced_config(SINGLE_JFH_CONFIG, cls.output_dir,
                                    angles=cls.ANGLES, distances=[4.0],
                                    write_vtk=True)
        cls.study = ParameterSweepStudy(cls.config)
        cls.results = cls.study.run()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir, ignore_errors=True)

    def test_strikes_form_one_numbered_series(self):
        strikes_dir = Path(self.output_dir) / 'results' / 'strikes'
        self.assertTrue(strikes_dir.is_dir())
        self.assertEqual(sorted(p.name for p in strikes_dir.glob('*.vtu')),
                         [f'firing-{i}.vtu' for i in range(len(self.ANGLES))])
        # ... and every record points into that one series.
        for index, case in enumerate(self.results.cases):
            self.assertEqual(Path(case.vtk_path).name, f'firing-{index}.vtu')
            self.assertTrue(os.path.isfile(case.vtk_path))
        self.assertFalse((Path(self.output_dir) / 'cases').exists(),
                         msg='single_jfh mode must not create per-case dirs')

    def test_cumulative_fields_are_in_the_last_firing_vtk(self):
        last = Path(self.results.cases[-1].vtk_path)
        head = last.read_bytes()[:8000].decode('utf-8', 'replace')
        for field in ('strikes', 'cum_strikes', 'pressures', 'max_pressures',
                      'shear_stress', 'max_shears', 'heat_flux_rate',
                      'cum_heat_flux_load'):
            self.assertIn(f'Name="{field}"', head)

    def test_sweep_envelope_bounds_every_pose(self):
        envelope = self.study.envelope
        self.assertIn('plate', envelope)
        plate = envelope['plate']

        peak_pressure = max(case.max_pressure for case in self.results.cases)
        peak_shear = max(case.max_shear_stress
                         for case in self.results.cases)
        self.assertAlmostEqual(plate['max_pressure'], peak_pressure, places=9)
        self.assertAlmostEqual(plate['max_shear_stress'], peak_shear,
                               places=9)

        # Coverage: each pose struck at most the whole plate, and the sweep
        # as a whole struck at least as much as any single pose did.
        self.assertGreaterEqual(
            plate['unique_struck_faces'],
            max(case.struck_faces for case in self.results.cases))
        self.assertGreaterEqual(
            plate['swept_affected_area'],
            max(case.affected_area for case in self.results.cases))
        self.assertAlmostEqual(plate['component_area'],
                               self.results.cases[0].component_area,
                               places=9)
        self.assertEqual(self.results.provenance['sweep_envelope'], envelope)

    def test_optional_plots_work_from_the_single_history_results(self):
        paths = self.study.plot()
        self.assertTrue(paths)
        names = {Path(path).name for path in paths}
        self.assertIn('force_vs_angle.png', names)
        for path in paths:
            self.assertTrue(os.path.isfile(path))


if __name__ == '__main__':
    unittest.main()

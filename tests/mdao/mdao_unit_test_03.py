# ========================
# PyRPOD: tests/mdao/mdao_unit_test_03.py
# ========================
# Unit tests for prescribed firing generation and the exact meaning of
# n_firings (pyrpod.mdao.firing_plan, pyrpod.rpod.approach_maneuvers):
#
#   * invalid firing counts (zero, negative, fractional, non-numeric) are
#     rejected rather than coerced;
#   * one requested firing produces exactly one JFH entry, and N requested
#     firings produce exactly N -- verified by reading the written file back
#     through the normal JetFiringHistory parser;
#   * an explicit firing list whose length disagrees with n_firings is an
#     error, never a silent truncation;
#   * the generated pose convention reproduces the committed sweep-JFH
#     generator of case/plume/plume_flat_plate_sweep to file precision;
#   * compute_1d_approach honors an exact n_firings on the dynamics path.
#
# No case file I/O beyond reading the committed sweep JFH; generated JFH
# files are written to a temporary directory.
#
# Run:  python -m pytest mdao/mdao_unit_test_03.py   (from tests/)

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from pyrpod.mdao.firing_plan import (
    build_case_firings,
    build_sweep_firings,
    pose_for,
    validate_n_firings,
    write_jfh_file,
)
from pyrpod.mdao.study_config import StudyConfigError, SweepSpec, TargetSpec
from pyrpod.rpod import JetFiringHistory
from pyrpod.rpod.approach_maneuvers import ApproachInputs, compute_1d_approach

CASE_DIR = '../case/plume/plume_flat_plate_sweep/'

# The committed sweep: 19 approach angles x 5 stand-off distances, plate
# centered at the origin, normal +Z, tangent +X (see the case's
# jfh/generate_sweep_jfh.py, run with --alpha0-deg 0 --distance 0).
SWEEP_ANGLES = np.arange(-90.0, 90.0 + 1e-9, 10.0)
SWEEP_DISTANCES = [2.0, 4.0, 6.0, 8.0, 10.0]


def _target(reference_point=(0.0, 0.0, 0.0)):
    return TargetSpec.from_mapping(
        {'reference_point': list(reference_point),
         'normal': [0.0, 0.0, 1.0],
         'tangent': [1.0, 0.0, 0.0]},
        default_geometry_id='test_geometry.stl')


def _sweep(n_firings=1, firings=None, angles=(0.0,), distances=(4.0,),
           mode='per_case'):
    data = {'plate_angles_deg': list(angles),
            'source_distances': list(distances),
            'n_firings': n_firings, 'firing_duration_s': 1.0,
            'thrusters': [1], 'mode': mode}
    if firings is not None:
        data['firings'] = firings
    return SweepSpec.from_mapping(data)


def _read_back(firings):
    """Write firings to a temp JFH and parse it with JetFiringHistory."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'jfh' / 'generated.A'
        written = write_jfh_file(path, firings)

        jfh = JetFiringHistory.JetFiringHistory(CASE_DIR)
        jfh.case_dir = str(tmp) + '/'
        jfh.config.set('jfh', 'jfh', 'generated.A')
        jfh.read_jfh()
        return written, list(jfh.JFH)


class FiringCountValidation(unittest.TestCase):
    """n_firings is an exact entry count, so only positive integers pass."""

    def test_invalid_counts_are_rejected(self):
        for value in (0, -1, -10, 1.5, 0.0, '2', None, True):
            with self.subTest(value=value):
                with pytest.raises(ValueError):
                    validate_n_firings(value)

    def test_valid_counts_are_returned_as_int(self):
        for value in (1, 3, 95, 10.0):
            with self.subTest(value=value):
                self.assertEqual(validate_n_firings(value), int(value))

    def test_zero_firings_rejected_by_the_sweep_specification(self):
        with pytest.raises(StudyConfigError):
            _sweep(n_firings=0)

    def test_writing_an_empty_jfh_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(StudyConfigError):
                write_jfh_file(Path(tmp) / 'empty.A', [])


class GeneratedFiringCounts(unittest.TestCase):
    """N requested firings produce exactly N JFH entries."""

    def test_one_firing_produces_one_entry(self):
        firings = build_case_firings(_sweep(n_firings=1), _target(), 0.0, 4.0)
        self.assertEqual(len(firings), 1)

        written, entries = _read_back(firings)
        self.assertEqual(written, 1)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['thrusters'], [1])

    def test_n_firings_produce_exactly_n_entries(self):
        for n_firings in (1, 2, 5, 17):
            with self.subTest(n_firings=n_firings):
                firings = build_case_firings(_sweep(n_firings=n_firings),
                                             _target(), -30.0, 6.0)
                self.assertEqual(len(firings), n_firings)

                written, entries = _read_back(firings)
                self.assertEqual(written, n_firings)
                self.assertEqual(len(entries), n_firings)
                # Every entry is a complete, valid firing record.
                for index, entry in enumerate(entries):
                    self.assertEqual(int(entry['nt']), index + 1)
                    self.assertEqual(np.asarray(entry['dcm']).shape, (3, 3))
                    self.assertEqual(len(entry['xyz']), 3)
                    self.assertGreater(float(entry['t']), 0.0)

    def test_repeated_firings_share_the_pose_and_advance_in_time(self):
        firings = build_case_firings(_sweep(n_firings=3), _target(), 20.0, 8.0)
        for firing in firings[1:]:
            np.testing.assert_allclose(firing.position, firings[0].position)
            np.testing.assert_allclose(firing.dcm, firings[0].dcm)
        self.assertEqual([f.start_time_s for f in firings], [0.0, 1.0, 2.0])


class PrescribedFiringList(unittest.TestCase):
    """Explicit firing sequences are honored, mismatches are not."""

    def test_explicit_firings_are_used_verbatim(self):
        explicit = [{'position': [0.0, 0.0, 3.0],
                     'dcm': [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0]],
                     'thrusters': [1], 'duration_s': 0.5},
                    {'position': [1.0, 0.0, 3.0],
                     'dcm': [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0]],
                     'thrusters': [1], 'duration_s': 0.5}]
        sweep = _sweep(n_firings=2, firings=explicit)
        firings = build_case_firings(sweep, _target(), 0.0, 4.0)

        self.assertEqual(len(firings), 2)
        np.testing.assert_allclose(firings[0].position, [0.0, 0.0, 3.0])
        np.testing.assert_allclose(firings[1].position, [1.0, 0.0, 3.0])
        self.assertEqual([f.duration_s for f in firings], [0.5, 0.5])

        written, entries = _read_back(firings)
        self.assertEqual((written, len(entries)), (2, 2))

    def test_count_mismatch_is_an_error(self):
        explicit = [{'position': [0.0, 0.0, 3.0],
                     'dcm': np.eye(3).tolist()}]
        with pytest.raises(StudyConfigError) as excinfo:
            _sweep(n_firings=4, firings=explicit)
        self.assertIn('n_firings', str(excinfo.value))


class WholeSweepFiringSequence(unittest.TestCase):
    """single_jfh mode: one history spanning every pose, exact length."""

    ANGLES = (-30.0, 0.0, 30.0)
    DISTANCES = (2.0, 6.0)

    def _sweep(self, n_firings=1, firings=None):
        return _sweep(n_firings=n_firings, firings=firings,
                      angles=self.ANGLES, distances=self.DISTANCES,
                      mode='single_jfh')

    def test_pose_order_is_distance_major(self):
        sweep = self._sweep()
        self.assertEqual(
            sweep.poses,
            tuple((angle, distance) for distance in self.DISTANCES
                  for angle in self.ANGLES))
        self.assertEqual(sweep.total_firings, 6)

    def test_sequence_holds_exactly_poses_times_n_firings(self):
        for n_firings in (1, 3):
            with self.subTest(n_firings=n_firings):
                sweep = self._sweep(n_firings=n_firings)
                firings = build_sweep_firings(sweep, _target())
                expected = len(self.ANGLES) * len(self.DISTANCES) * n_firings
                self.assertEqual(len(firings), expected)
                self.assertEqual(sweep.total_firings, expected)

                written, entries = _read_back(firings)
                self.assertEqual(written, expected)
                self.assertEqual(len(entries), expected)

    def test_each_firing_is_tagged_with_the_pose_it_realizes(self):
        firings = build_sweep_firings(self._sweep(n_firings=2), _target())
        tags = [(f.plate_angle_deg, f.source_distance) for f in firings]
        # Two consecutive entries per pose, poses in distance-major order.
        expected = [pose for pose in self._sweep().poses for _ in range(2)]
        self.assertEqual(tags, expected)
        self.assertEqual([f.pose_index for f in firings],
                         [i for i in range(6) for _ in range(2)])

    def test_firing_times_run_continuously_across_the_sweep(self):
        firings = build_sweep_firings(self._sweep(n_firings=2), _target())
        self.assertEqual([f.start_time_s for f in firings],
                         [float(i) for i in range(len(firings))])

    def test_poses_match_the_per_case_engine_pose_by_pose(self):
        sweep = self._sweep()
        combined = build_sweep_firings(sweep, _target())
        for index, (angle, distance) in enumerate(sweep.poses):
            per_case = build_case_firings(_sweep(angles=self.ANGLES,
                                                 distances=self.DISTANCES),
                                          _target(), angle, distance)
            np.testing.assert_allclose(combined[index].position,
                                       per_case[0].position)
            np.testing.assert_allclose(combined[index].dcm, per_case[0].dcm)

    def test_explicit_firings_are_sliced_across_the_poses(self):
        # Six poses x one firing each: the supplied sequence IS the history.
        explicit = [{'position': [0.0, 0.0, float(i + 2)],
                     'dcm': np.eye(3).tolist()} for i in range(6)]
        sweep = self._sweep(n_firings=1, firings=explicit)
        firings = build_sweep_firings(sweep, _target())

        self.assertEqual(len(firings), 6)
        self.assertEqual([f.position[2] for f in firings],
                         [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        # ... and each entry still carries the pose it stands for.
        self.assertEqual([(f.plate_angle_deg, f.source_distance)
                          for f in firings], list(sweep.poses))

    def test_explicit_firing_count_must_cover_every_pose(self):
        explicit = [{'position': [0.0, 0.0, 4.0], 'dcm': np.eye(3).tolist()}]
        with pytest.raises(StudyConfigError) as excinfo:
            self._sweep(n_firings=1, firings=explicit)
        self.assertIn('single_jfh', str(excinfo.value))


class PoseConvention(unittest.TestCase):
    """The generated poses match the committed sweep-JFH generator."""

    def test_head_on_pose_is_on_the_target_normal(self):
        position, dcm = pose_for(0.0, 4.0, [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(position, [0.0, 0.0, 4.0], atol=1e-12)
        # The DCM's first column is the thruster axis, aimed at the target.
        np.testing.assert_allclose(dcm[:, 0], [0.0, 0.0, -1.0], atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(dcm)), 1.0, places=12)

    def test_positive_angle_rotates_toward_the_tangent(self):
        position, _ = pose_for(90.0, 5.0, [0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(position, [5.0, 0.0, 0.0], atol=1e-12)

    def test_matches_the_committed_flat_plate_sweep_jfh(self):
        jfh = JetFiringHistory.JetFiringHistory(CASE_DIR)
        jfh.read_jfh()
        self.assertEqual(len(jfh.JFH),
                         len(SWEEP_ANGLES) * len(SWEEP_DISTANCES))

        for index, entry in enumerate(jfh.JFH):
            distance = SWEEP_DISTANCES[index // len(SWEEP_ANGLES)]
            angle = float(SWEEP_ANGLES[index % len(SWEEP_ANGLES)])
            position, dcm = pose_for(angle, distance, [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
            # The JFH file stores positions to 9 and DCMs to 6 significant
            # digits, so file precision is the tolerance here.
            np.testing.assert_allclose(entry['xyz'], position, atol=1e-7)
            np.testing.assert_allclose(entry['dcm'], dcm, atol=1e-6)


class _StubFuelManager:
    def calc_delta_v(self, dt, v_e, m_dot_sum, m_o):
        return 0.1


class _StubGrouping:
    def calc_m_dot_sum(self, group):
        return 0.01

    def calc_v_e(self, group):
        return 3000.0


class _StubVehicle:
    """Minimal stand-in for the attributes compute_1d_approach reads."""

    mass = 1000.0
    rcs_groups = {'neg_x': ['T1']}
    thruster_data = {'T1': {'type': ['R4D']}}
    thruster_metrics = {'R4D': {'MIB': 0.4, 'F': 400.0}}


class DynamicsPathFiringCount(unittest.TestCase):
    """The dynamics-driven approach honors an exact firing count too."""

    def _run(self, n_firings):
        return compute_1d_approach(
            inputs=ApproachInputs(v_ida=0.0, v_o=5.0, r_o=20.0),
            vv=_StubVehicle(), fuel_mgr=_StubFuelManager(),
            grouping=_StubGrouping(), cant_rad=0.0,
            dt_strategy={'multiplier': 1.0}, n_firings=n_firings)

    def test_requested_count_is_exact(self):
        for n_firings in (1, 3, 10):
            with self.subTest(n_firings=n_firings):
                results = self._run(n_firings)
                self.assertEqual(len(results['t']), n_firings)
                self.assertEqual(len(results['x']), n_firings)
                self.assertEqual(len(results['rot']), n_firings)

    def test_invalid_count_is_rejected(self):
        for value in (0, -3, 2.5):
            with self.subTest(value=value):
                with pytest.raises(ValueError):
                    self._run(value)

    def test_unreachable_count_is_reported_not_silently_shortened(self):
        # 5 m/s of delta-v at 0.1 m/s per firing completes in ~50 firings.
        with pytest.raises(ValueError) as excinfo:
            self._run(500)
        self.assertIn('n_firings', str(excinfo.value))

    def test_omitting_the_count_keeps_the_original_behavior(self):
        results = compute_1d_approach(
            inputs=ApproachInputs(v_ida=0.0, v_o=5.0, r_o=20.0),
            vv=_StubVehicle(), fuel_mgr=_StubFuelManager(),
            grouping=_StubGrouping(), cant_rad=0.0,
            dt_strategy={'multiplier': 1.0})
        # Unbounded, the loop runs until the required delta-v is spent: 5.0
        # m/s at 0.1 m/s per firing (one extra iteration from the residual
        # of the accumulated subtraction), plus the initial state entry.
        self.assertEqual(len(results['t']), 52)


if __name__ == '__main__':
    unittest.main()

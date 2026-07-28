# ========================
# PyRPOD: tests/mdao/mdao_unit_test_06.py
# ========================
# Unit tests for the generic external-reference comparison
# (pyrpod.mdao.reference_data) and the structured result schema
# (pyrpod.mdao.study_results):
#
#   * the metrics themselves -- absolute error, relative error, normalized
#     RMSE, peak error, integrated-load error and center-of-pressure
#     displacement -- against hand-computed values, including the cases
#     where a metric is undefined and must return None instead of infinity;
#   * loading reference datasets from CSV, JSON and YAML with the SAME
#     comparison behavior, since the interface must not know (or care)
#     whether the data came from DSMC, an analytical solution, an experiment
#     or another code;
#   * matching by case keys, and the reporting of unmatched cases and of
#     quantities the candidate does not provide -- nothing is fabricated;
#   * the result schema round-tripping to CSV rows and JSON.
#
# Run:  python -m pytest mdao/mdao_unit_test_06.py   (from tests/)

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import yaml

from pyrpod.mdao.reference_data import (
    ReferenceDataset,
    ReferenceRecord,
    absolute_error,
    center_of_pressure_displacement,
    compare_case,
    compare_results,
    integrated_load_error,
    load_reference_dataset,
    normalized_rmse,
    peak_error,
    relative_error,
)
from pyrpod.mdao.study_results import CaseResult, StudyResults
from pyrpod.mdao.surface_loads import ComponentLoads


def make_result(case_id='case000', component='plate', force=(0.0, 0.0, -10.0),
                moment=(0.0, 2.0, 0.0), cop=(0.1, 0.0, 0.0),
                plate_angle_deg=0.0, source_distance=4.0,
                coefficients=None):
    loads = ComponentLoads(
        component=component, n_faces=8, n_struck_faces=8, total_area=4.0,
        affected_area=4.0,
        pressure_force=np.asarray(force, dtype=float),
        shear_force=np.zeros(3),
        force=np.asarray(force, dtype=float),
        force_magnitude=float(np.linalg.norm(force)),
        moment_reference_point=np.zeros(3),
        pressure_moment=np.asarray(moment, dtype=float),
        shear_moment=np.zeros(3),
        moment=np.asarray(moment, dtype=float),
        moment_magnitude=float(np.linalg.norm(moment)),
        center_of_pressure=(None if cop is None
                            else np.asarray(cop, dtype=float)),
        center_of_pressure_status='zero_load' if cop is None else 'ok',
        residual_couple=0.0,
        pressure_weighted_centroid=None if cop is None
        else np.asarray(cop, dtype=float),
        max_pressure=12.0, max_shear_stress=1.5, max_heat_flux=300.0,
        total_heat_load=1200.0,
        coefficients=dict(coefficients or {}))
    return CaseResult.from_loads(
        loads, study_name='unit', case_id=case_id, firing_id=1,
        geometry_id='plate.stl', mesh_faces=8,
        coordinate_system='case global frame', units={'force': 'N'},
        plume_source_position=[0.0, 0.0, 4.0],
        plume_source_orientation=list(np.eye(3).ravel()),
        target_normal=[0.0, 0.0, 1.0], target_tangent=[1.0, 0.0, 0.0],
        target_reference_point=[0.0, 0.0, 0.0],
        plate_angle_deg=plate_angle_deg, source_distance=source_distance,
        firing_duration_s=1.0, thrusters=[1],
        plume_model='SimplifiedGasKinetics', plume_model_parameters={})


class Metrics(unittest.TestCase):

    def test_absolute_error_of_scalars_and_vectors(self):
        self.assertAlmostEqual(absolute_error(5.0, 4.0), 1.0)
        self.assertAlmostEqual(absolute_error([3.0, 4.0, 0.0],
                                              [0.0, 0.0, 0.0]), 5.0)

    def test_relative_error_is_none_for_a_zero_reference(self):
        self.assertAlmostEqual(relative_error(11.0, 10.0), 0.1)
        self.assertIsNone(relative_error(1.0, 0.0))

    def test_normalized_rmse_uses_the_reference_range(self):
        reference = np.array([0.0, 1.0, 2.0, 3.0])
        candidate = reference + 0.3
        # RMSE 0.3 over a range of 3.0
        self.assertAlmostEqual(normalized_rmse(candidate, reference), 0.1)
        self.assertAlmostEqual(
            normalized_rmse(candidate, reference, norm='mean'),
            0.3 / 1.5)

    def test_normalized_rmse_is_none_when_undefined(self):
        constant = np.array([2.0, 2.0, 2.0])
        self.assertIsNone(normalized_rmse(constant + 0.1, constant))
        self.assertIsNone(normalized_rmse([1.0, 2.0], [1.0, 2.0, 3.0]))

    def test_peak_error(self):
        absolute, relative = peak_error([1.0, 4.0, 2.0], [1.0, 5.0, 2.0])
        self.assertAlmostEqual(absolute, 1.0)
        self.assertAlmostEqual(relative, 0.2)

    def test_integrated_load_error_compares_vectors_not_magnitudes(self):
        # Same magnitude, opposite direction: a magnitude-only comparison
        # would report zero error.
        absolute, relative = integrated_load_error([0.0, 0.0, 10.0],
                                                   [0.0, 0.0, -10.0])
        self.assertAlmostEqual(absolute, 20.0)
        self.assertAlmostEqual(relative, 2.0)

    def test_center_of_pressure_displacement(self):
        distance, normalized = center_of_pressure_displacement(
            [0.3, 0.0, 0.0], [0.0, 0.4, 0.0], reference_length=2.0)
        self.assertAlmostEqual(distance, 0.5)
        self.assertAlmostEqual(normalized, 0.25)

        distance, normalized = center_of_pressure_displacement(
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(distance, 0.0)
        self.assertIsNone(normalized)


class DatasetLoading(unittest.TestCase):
    """The same records, expressed in any supported format, compare alike."""

    QUANTITIES = {'force': [0.0, 0.0, -9.0], 'max_pressure': 11.0}

    def _csv(self, directory):
        path = Path(directory) / 'reference.csv'
        path.write_text(
            'case_id,component,plate_angle_deg,source_distance,'
            'force_x,force_y,force_z,max_pressure\n'
            'case000,plate,0.0,4.0,0.0,0.0,-9.0,11.0\n',
            encoding='utf-8')
        return path

    def _json(self, directory):
        path = Path(directory) / 'reference.json'
        path.write_text(json.dumps({
            'label': 'independent solver',
            'source': 'run-42',
            'units': {'force': 'N'},
            'records': [{'key': {'case_id': 'case000', 'component': 'plate'},
                         'quantities': self.QUANTITIES}]}), encoding='utf-8')
        return path

    def _yaml(self, directory):
        path = Path(directory) / 'reference.yaml'
        path.write_text(yaml.safe_dump({
            'label': 'experiment',
            'records': [{'key': {'case_id': 'case000'},
                         'quantities': self.QUANTITIES}]}), encoding='utf-8')
        return path

    def test_every_format_yields_the_same_comparison(self):
        result = make_result()
        with tempfile.TemporaryDirectory() as tmp:
            for builder in (self._csv, self._json, self._yaml):
                with self.subTest(builder=builder.__name__):
                    dataset = load_reference_dataset(builder(tmp))
                    self.assertEqual(len(dataset), 1)
                    report = compare_results([result], dataset)
                    by_name = {c.quantity: c for c in report.comparisons}
                    self.assertEqual(set(by_name), {'force', 'max_pressure'})
                    self.assertAlmostEqual(by_name['force'].absolute_error,
                                           1.0)
                    self.assertAlmostEqual(
                        by_name['max_pressure'].relative_error,
                        1.0 / 11.0)

    def test_csv_vector_columns_are_assembled(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = load_reference_dataset(self._csv(tmp))
            record = dataset.records[0]
            self.assertEqual(record.quantities['force'], [0.0, 0.0, -9.0])
            self.assertEqual(record.key['case_id'], 'case000')
            self.assertAlmostEqual(record.key['plate_angle_deg'], 0.0)

    def test_label_can_be_overridden_and_is_only_a_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = load_reference_dataset(self._json(tmp), label='DSMC')
            self.assertEqual(dataset.label, 'DSMC')
            report = compare_results([make_result()], dataset)
            # The metrics do not depend on where the data came from.
            self.assertEqual(report.label, 'DSMC')
            self.assertAlmostEqual(
                report.max_relative_error('max_pressure'), 1.0 / 11.0)

    def test_unknown_format_and_missing_file_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'reference.parquet'
            bad.write_text('', encoding='utf-8')
            with pytest.raises(ValueError):
                load_reference_dataset(bad)
            with pytest.raises(FileNotFoundError):
                load_reference_dataset(Path(tmp) / 'absent.csv')


class Matching(unittest.TestCase):

    def test_records_match_on_swept_parameters(self):
        dataset = ReferenceDataset(records=[
            ReferenceRecord(key={'plate_angle_deg': 30.0,
                                 'source_distance': 4.0},
                            quantities={'max_pressure': 10.0})])
        matching = make_result(plate_angle_deg=30.0, source_distance=4.0)
        other = make_result(plate_angle_deg=-30.0, source_distance=4.0)

        self.assertIsNotNone(dataset.match(matching))
        self.assertIsNone(dataset.match(other))

    def test_unmatched_cases_are_listed_not_compared(self):
        dataset = ReferenceDataset(records=[
            ReferenceRecord(key={'case_id': 'case000'},
                            quantities={'max_pressure': 10.0})])
        report = compare_results(
            [make_result(case_id='case000'), make_result(case_id='case001')],
            dataset)

        self.assertEqual(len(report.comparisons), 1)
        self.assertEqual(report.unmatched_cases, ['case001/plate'])

    def test_quantity_absent_from_the_candidate_is_flagged(self):
        result = make_result()          # no coefficients supplied
        record = ReferenceRecord(key={'case_id': 'case000'},
                                 quantities={'CF': 0.5})
        comparison = compare_case(result, record)[0]

        self.assertEqual(comparison.status, 'missing_candidate')
        self.assertIsNone(comparison.absolute_error)
        self.assertEqual(comparison.reference, 0.5)

    def test_coefficients_are_compared_when_available(self):
        result = make_result(coefficients={'CF': 0.4})
        record = ReferenceRecord(key={'case_id': 'case000'},
                                 quantities={'CF': 0.5})
        comparison = compare_case(result, record)[0]

        self.assertEqual(comparison.status, 'compared')
        self.assertAlmostEqual(comparison.absolute_error, 0.1)
        self.assertAlmostEqual(comparison.relative_error, 0.2)

    def test_shape_mismatch_is_reported_rather_than_coerced(self):
        record = ReferenceRecord(key={'case_id': 'case000'},
                                 quantities={'force': [1.0, 2.0]})
        comparison = compare_case(make_result(), record)[0]
        self.assertEqual(comparison.status, 'shape_mismatch')

    def test_center_of_pressure_displacement_is_recorded(self):
        record = ReferenceRecord(
            key={'case_id': 'case000'},
            quantities={'center_of_pressure': [0.0, 0.0, 0.0]})
        comparison = compare_case(make_result(cop=(0.1, 0.0, 0.0)), record,
                                  reference_length=2.0)[0]
        self.assertAlmostEqual(comparison.displacement, 0.1)

    def test_empty_dataset_produces_an_empty_report(self):
        report = compare_results([make_result()], ReferenceDataset())
        self.assertEqual(len(report), 0)
        self.assertEqual(report.unmatched_cases, ['case000/plate'])
        with pytest.raises(ValueError):
            report.write_csv(Path(tempfile.gettempdir()) / 'empty.csv')


class ResultSchema(unittest.TestCase):

    def test_row_expands_vectors_and_coefficients(self):
        row = make_result(coefficients={'CF': 0.25}).to_row()
        self.assertAlmostEqual(row['force_z'], -10.0)
        self.assertAlmostEqual(row['moment_y'], 2.0)
        self.assertAlmostEqual(row['center_of_pressure_x'], 0.1)
        self.assertAlmostEqual(row['coeff_CF'], 0.25)
        self.assertTrue(row['coefficients_available'])

    def test_missing_center_of_pressure_leaves_empty_columns(self):
        row = make_result(cop=None).to_row()
        self.assertEqual(row['center_of_pressure_x'], '')
        self.assertEqual(row['center_of_pressure_status'], 'zero_load')

    def test_results_write_csv_and_json(self):
        results = StudyResults(study_name='unit', output_dir='',
                               provenance={'plume_model':
                                           'SimplifiedGasKinetics'},
                               cases=[make_result(case_id='case000'),
                                      make_result(case_id='case001')])
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = results.write_csv(Path(tmp) / 'results.csv')
            json_path = results.write_metadata(Path(tmp) / 'metadata.json')

            lines = Path(csv_path).read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines), 3)          # header + two rows
            self.assertIn('force_z', lines[0])

            document = json.loads(Path(json_path).read_text(encoding='utf-8'))
            self.assertEqual(document['n_cases'], 2)
            self.assertEqual(document['provenance']['plume_model'],
                             'SimplifiedGasKinetics')
            self.assertEqual(document['cases'][0]['case_id'], 'case000')
            self.assertIn('units', document['cases'][0])

    def test_quantity_lookup_covers_the_comparable_fields(self):
        result = make_result(coefficients={'CF': 0.25})
        self.assertEqual(result.quantity('force'), [0.0, 0.0, -10.0])
        self.assertTrue(math.isclose(result.quantity('max_heat_flux'), 300.0))
        self.assertAlmostEqual(result.quantity('CF'), 0.25)
        self.assertIsNone(result.quantity('not_a_quantity'))


if __name__ == '__main__':
    unittest.main()

# ========================
# PyRPOD: tests/mdao/mdao_unit_test_04.py
# ========================
# Unit tests for the surface-load integration of pyrpod.mdao.surface_loads,
# on small meshes whose loading has a known closed-form resultant:
#
#   * force integration: uniform pressure on a flat plate gives -p*A*n_hat,
#     and shear acts along the tangential projection of the flow direction;
#   * moment about a user-defined reference point: M = (r_c - r_ref) x F,
#     checked against a hand-computed value and against the plate's own
#     linear pressure distribution;
#   * center of pressure: recovered exactly for a linearly varying pressure
#     field, consistent with the reported moment, and reported as unavailable
#     (with a status naming the reason) for zero load and for near-total
#     cancellation;
#   * coefficients: computed when the normalization inputs are complete,
#     omitted entirely when they are not -- no invented defaults;
#   * component selection: a bounds-selected pair of components reproduces
#     the whole-mesh resultant when summed.
#
# Everything here is constructed in memory; no case files are read.
#
# Run:  python -m pytest mdao/mdao_unit_test_04.py   (from tests/)

import unittest

import numpy as np
import pytest

from pyrpod.mdao.study_config import ComponentSpec, Normalization
from pyrpod.mdao.surface_loads import (
    face_areas,
    flow_directions,
    integrate_component_loads,
    select_component_faces,
)


def unit_square_plate(n=4, half=1.0, z=0.0):
    """Triangulated square plate in the z-plane, unit normals along +Z.

    Returns (vectors, centroids, normals, areas) for an n x n grid of cells
    split into two triangles each.
    """
    edges = np.linspace(-half, half, n + 1)
    faces = []
    for i in range(n):
        for j in range(n):
            x0, x1 = edges[i], edges[i + 1]
            y0, y1 = edges[j], edges[j + 1]
            # Counter-clockwise seen from +Z, so the unit normal is +Z.
            faces.append([[x0, y0, z], [x1, y0, z], [x1, y1, z]])
            faces.append([[x0, y0, z], [x1, y1, z], [x0, y1, z]])
    vectors = np.asarray(faces, dtype=float)
    centroids = vectors.mean(axis=1)
    normals = np.tile([0.0, 0.0, 1.0], (len(vectors), 1))
    return vectors, centroids, normals, face_areas(vectors)


def integrate(centroids, normals, areas, pressures, shears=None,
              heat=None, moment_ref=(0.0, 0.0, 0.0),
              source=(0.0, 0.0, 5.0), normalization=None, strikes=None):
    n_faces = len(centroids)
    zeros = np.zeros(n_faces)
    return integrate_component_loads(
        component_name='plate',
        face_indices=np.arange(n_faces),
        centroids=centroids, unit_normals=normals, areas=areas,
        pressures=np.asarray(pressures, dtype=float),
        shear_stresses=zeros if shears is None
        else np.asarray(shears, dtype=float),
        heat_fluxes=zeros if heat is None else np.asarray(heat, dtype=float),
        strikes=strikes,
        moment_reference_point=moment_ref,
        source_position=source,
        normalization=normalization)


class ForceIntegration(unittest.TestCase):

    def test_uniform_pressure_on_a_flat_plate(self):
        vectors, centroids, normals, areas = unit_square_plate(n=4, half=1.0)
        pressure = 250.0
        loads = integrate(centroids, normals, areas,
                          np.full(len(centroids), pressure))

        total_area = 4.0                       # 2 m x 2 m plate
        self.assertAlmostEqual(loads.total_area, total_area, places=12)
        # Pressure pushes into the surface: along -n_hat.
        np.testing.assert_allclose(loads.force,
                                   [0.0, 0.0, -pressure * total_area],
                                   atol=1e-9)
        np.testing.assert_allclose(loads.pressure_force, loads.force,
                                   atol=1e-12)
        np.testing.assert_allclose(loads.shear_force, np.zeros(3), atol=1e-12)
        self.assertAlmostEqual(loads.force_magnitude, pressure * total_area,
                               places=9)

    def test_shear_acts_along_the_tangential_flow_direction(self):
        vectors, centroids, normals, areas = unit_square_plate(n=2, half=1.0)
        shear = 3.0
        # Flow arriving at 45 deg in the X-Z plane: the tangential
        # projection on the plate is +X for every face.
        flow = np.tile([np.sqrt(0.5), 0.0, -np.sqrt(0.5)], (len(centroids), 1))
        loads = integrate_component_loads(
            component_name='plate', face_indices=np.arange(len(centroids)),
            centroids=centroids, unit_normals=normals, areas=areas,
            pressures=np.zeros(len(centroids)),
            shear_stresses=np.full(len(centroids), shear),
            heat_fluxes=np.zeros(len(centroids)), strikes=None,
            moment_reference_point=[0.0, 0.0, 0.0], flow_unit_vectors=flow)

        np.testing.assert_allclose(loads.shear_force,
                                   [shear * 4.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(loads.pressure_force, np.zeros(3),
                                   atol=1e-12)

    def test_affected_area_counts_struck_faces_only(self):
        vectors, centroids, normals, areas = unit_square_plate(n=2, half=1.0)
        strikes = np.zeros(len(centroids))
        strikes[:4] = 1.0
        loads = integrate(centroids, normals, areas,
                          np.full(len(centroids), 10.0), strikes=strikes)

        self.assertEqual(loads.n_struck_faces, 4)
        self.assertAlmostEqual(loads.affected_area, float(np.sum(areas[:4])),
                               places=12)
        self.assertAlmostEqual(loads.total_area, 4.0, places=12)


class MomentIntegration(unittest.TestCase):

    def test_moment_about_a_user_defined_point(self):
        vectors, centroids, normals, areas = unit_square_plate(n=4, half=1.0)
        pressure = 100.0
        reference = np.array([2.0, -1.0, 3.0])
        loads = integrate(centroids, normals, areas,
                          np.full(len(centroids), pressure),
                          moment_ref=reference)

        # Uniform load: the resultant acts at the plate centroid (origin).
        force = np.array([0.0, 0.0, -pressure * 4.0])
        expected = np.cross(np.zeros(3) - reference, force)
        np.testing.assert_allclose(loads.moment, expected, atol=1e-9)
        np.testing.assert_allclose(loads.pressure_moment, expected, atol=1e-9)
        np.testing.assert_allclose(loads.shear_moment, np.zeros(3),
                                   atol=1e-12)

    def test_moment_reference_point_is_reported_back(self):
        vectors, centroids, normals, areas = unit_square_plate(n=2, half=1.0)
        loads = integrate(centroids, normals, areas,
                          np.full(len(centroids), 5.0),
                          moment_ref=(0.5, 0.25, 0.0))
        np.testing.assert_allclose(loads.moment_reference_point,
                                   [0.5, 0.25, 0.0])

    def test_zero_reference_moment_when_taken_at_the_load_center(self):
        vectors, centroids, normals, areas = unit_square_plate(n=6, half=1.0)
        loads = integrate(centroids, normals, areas,
                          np.full(len(centroids), 42.0),
                          moment_ref=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(loads.moment, np.zeros(3), atol=1e-9)


class CenterOfPressure(unittest.TestCase):

    def test_recovered_for_a_linear_pressure_distribution(self):
        # p(x) = p0 * (1 + x) on [-1, 1] has its area-weighted center at
        # x = integral(x*p)/integral(p) = (2/3) / 2 = 1/3.
        vectors, centroids, normals, areas = unit_square_plate(n=40, half=1.0)
        pressures = 100.0 * (1.0 + centroids[:, 0])
        loads = integrate(centroids, normals, areas, pressures)

        self.assertEqual(loads.center_of_pressure_status, 'ok')
        cop = loads.center_of_pressure
        self.assertIsNotNone(cop)
        # Face-centroid sampling of a continuous field is a midpoint rule,
        # and the cell-diagonal split is not mirror-invariant, so both the
        # recovered x and the residual y carry an O(h^2) = (2/40)^2 ~ 2e-4
        # discretization error.
        self.assertAlmostEqual(float(cop[0]), 1.0 / 3.0, delta=5e-4)
        self.assertAlmostEqual(float(cop[1]), 0.0, delta=5e-4)
        self.assertAlmostEqual(float(cop[2]), 0.0, places=9)
        # The auxiliary pressure-weighted centroid agrees for a planar
        # component under unidirectional pressure.
        np.testing.assert_allclose(loads.pressure_weighted_centroid, cop,
                                   atol=1e-6)

    def test_is_consistent_with_the_reported_moment(self):
        vectors, centroids, normals, areas = unit_square_plate(n=20, half=1.0)
        pressures = 50.0 * (2.0 + centroids[:, 0] + 0.5 * centroids[:, 1])
        loads = integrate(centroids, normals, areas, pressures,
                          moment_ref=(0.3, -0.2, 0.1))

        arm = loads.center_of_pressure - loads.moment_reference_point
        np.testing.assert_allclose(np.cross(arm, loads.force), loads.moment,
                                   atol=1e-6)

    def test_zero_load_returns_no_center_of_pressure(self):
        vectors, centroids, normals, areas = unit_square_plate(n=2, half=1.0)
        loads = integrate(centroids, normals, areas,
                          np.zeros(len(centroids)))

        np.testing.assert_allclose(loads.force, np.zeros(3), atol=0.0)
        np.testing.assert_allclose(loads.moment, np.zeros(3), atol=0.0)
        self.assertIsNone(loads.center_of_pressure)
        self.assertEqual(loads.center_of_pressure_status, 'zero_load')
        self.assertIsNone(loads.pressure_weighted_centroid)
        self.assertEqual(loads.max_pressure, 0.0)
        self.assertEqual(loads.affected_area, 0.0)

    def test_cancelling_load_is_reported_as_ill_conditioned(self):
        # Two parallel plates with opposite normals and equal pressure: the
        # face forces are large but the resultant cancels, so no line of
        # action exists. A naive F x M / |F|^2 would explode here.
        vectors, centroids, normals, areas = unit_square_plate(n=4, half=1.0)
        vectors_b, centroids_b, normals_b, areas_b = unit_square_plate(
            n=4, half=1.0, z=1.0)
        centroids = np.vstack([centroids, centroids_b])
        normals = np.vstack([normals, -normals_b])
        areas = np.concatenate([areas, areas_b])
        loads = integrate(centroids, normals, areas,
                          np.full(len(centroids), 75.0))

        self.assertLess(loads.force_magnitude, 1e-9)
        self.assertIsNone(loads.center_of_pressure)
        self.assertEqual(loads.center_of_pressure_status, 'ill_conditioned')

    def test_residual_couple_is_reported(self):
        vectors, centroids, normals, areas = unit_square_plate(n=8, half=1.0)
        pressures = 20.0 * (1.0 + centroids[:, 0])
        loads = integrate(centroids, normals, areas, pressures,
                          moment_ref=(0.0, 0.0, 0.0))
        # Pure pressure on a planar component: the moment is perpendicular
        # to the force, so nothing is left over.
        self.assertLess(loads.residual_couple, 1e-9)


class Coefficients(unittest.TestCase):

    def _loads(self, normalization):
        vectors, centroids, normals, areas = unit_square_plate(n=4, half=1.0)
        pressures = np.full(len(centroids), 10.0)
        shears = np.full(len(centroids), 1.0)
        heat = np.full(len(centroids), 500.0)
        return integrate(centroids, normals, areas, pressures, shears=shears,
                         heat=heat, moment_ref=(1.0, 0.0, 0.0),
                         normalization=normalization)

    def test_computed_with_complete_normalization_inputs(self):
        normalization = Normalization(reference_area=4.0,
                                      reference_length=2.0,
                                      dynamic_pressure=100.0,
                                      reference_heat_flux=1000.0)
        loads = self._loads(normalization)

        self.assertTrue(loads.has_coefficients)
        self.assertAlmostEqual(loads.coefficients['CF'],
                               loads.force_magnitude / (100.0 * 4.0),
                               places=12)
        self.assertAlmostEqual(loads.coefficients['CM'],
                               loads.moment_magnitude / (100.0 * 4.0 * 2.0),
                               places=12)
        self.assertAlmostEqual(loads.coefficients['Cp_max'], 10.0 / 100.0,
                               places=12)
        self.assertAlmostEqual(loads.coefficients['Cf_max'], 1.0 / 100.0,
                               places=12)
        self.assertAlmostEqual(loads.coefficients['Cq_max'], 500.0 / 1000.0,
                               places=12)

    def test_omitted_entirely_when_no_normalization_is_supplied(self):
        loads = self._loads(None)
        self.assertFalse(loads.has_coefficients)
        self.assertEqual(loads.coefficients, {})

    def test_partial_normalization_yields_only_the_valid_coefficients(self):
        # Dynamic pressure but no reference area: surface-load coefficients
        # are available, force and moment coefficients are not.
        loads = self._loads(Normalization(dynamic_pressure=100.0))
        self.assertIn('Cp_max', loads.coefficients)
        self.assertNotIn('CF', loads.coefficients)
        self.assertNotIn('CM', loads.coefficients)

        # Force inputs but no reference length: no moment coefficient.
        loads = self._loads(Normalization(reference_area=4.0,
                                          dynamic_pressure=100.0))
        self.assertIn('CF', loads.coefficients)
        self.assertNotIn('CM', loads.coefficients)
        self.assertNotIn('Cq_max', loads.coefficients)

    def test_invalid_normalization_values_are_rejected(self):
        from pyrpod.mdao.study_config import StudyConfigError

        for payload in ({'reference_area': 0.0}, {'dynamic_pressure': -1.0}):
            with self.subTest(payload=payload):
                with pytest.raises(StudyConfigError):
                    Normalization.from_mapping(payload)


class ComponentSelection(unittest.TestCase):

    def test_components_partition_the_resultant(self):
        vectors, centroids, normals, areas = unit_square_plate(n=6, half=1.0)
        pressures = 30.0 * (2.0 + centroids[:, 0])

        left = ComponentSpec.from_mapping(
            {'name': 'left', 'bounds': {'min': [-1.0, -1.0, -0.1],
                                        'max': [0.0, 1.0, 0.1]}})
        right = ComponentSpec.from_mapping(
            {'name': 'right', 'bounds': {'min': [0.0, -1.0, -0.1],
                                         'max': [1.0, 1.0, 0.1]}})
        whole = integrate(centroids, normals, areas, pressures)

        partial = np.zeros(3)
        for component in (left, right):
            indices = select_component_faces(component, centroids,
                                             len(centroids))
            loads = integrate_component_loads(
                component_name=component.name, face_indices=indices,
                centroids=centroids, unit_normals=normals, areas=areas,
                pressures=pressures,
                shear_stresses=np.zeros(len(centroids)),
                heat_fluxes=np.zeros(len(centroids)), strikes=None,
                moment_reference_point=[0.0, 0.0, 0.0],
                source_position=[0.0, 0.0, 5.0])
            partial = partial + loads.force
            self.assertLess(loads.n_faces, whole.n_faces)

        np.testing.assert_allclose(partial, whole.force, atol=1e-9)

    def test_explicit_face_indices(self):
        vectors, centroids, normals, areas = unit_square_plate(n=2, half=1.0)
        component = ComponentSpec.from_mapping(
            {'name': 'strip', 'face_indices': [0, 1, 2]})
        indices = select_component_faces(component, centroids, len(centroids))
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_out_of_range_and_empty_selections_are_rejected(self):
        vectors, centroids, normals, areas = unit_square_plate(n=2, half=1.0)
        with pytest.raises(ValueError):
            select_component_faces(
                ComponentSpec.from_mapping({'name': 'bad',
                                            'face_indices': [999]}),
                centroids, len(centroids))
        with pytest.raises(ValueError):
            select_component_faces(
                ComponentSpec.from_mapping(
                    {'name': 'empty',
                     'bounds': {'min': [10.0, 10.0, 10.0],
                                'max': [11.0, 11.0, 11.0]}}),
                centroids, len(centroids))


class FlowDirections(unittest.TestCase):

    def test_unit_vectors_point_from_the_source_to_each_face(self):
        centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        directions = flow_directions(centroids, [0.0, 0.0, 2.0])
        np.testing.assert_allclose(directions[0], [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1),
                                   [1.0, 1.0], atol=1e-12)

    def test_face_at_the_source_yields_a_zero_direction(self):
        directions = flow_directions(np.array([[0.0, 0.0, 2.0]]),
                                     [0.0, 0.0, 2.0])
        np.testing.assert_allclose(directions[0], np.zeros(3))


if __name__ == '__main__':
    unittest.main()

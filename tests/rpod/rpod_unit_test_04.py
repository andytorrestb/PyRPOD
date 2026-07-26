# ========================
# PyRPOD: tests/rpod/rpod_unit_test_04.py
# ========================
# Unit tests for the PR #116 plume-mesh consolidation refactor:
#   * VisitingVehicle.transform_plume_mesh  (canonical plume placement)
#   * VisitingVehicle.get_thruster_id / get_thruster_id_map (cached mapping)
#   * VisitingVehicle._cluster_id_for_thruster (multi-digit cluster ids)
#   * pyrpod.util.stl.stl.compose_meshes     (generic mesh composition)
#   * pyrpod.util.stl.stl.transform_mesh     (reusable mesh-object transform)
#
# These tests construct meshes and lightweight VisitingVehicle instances
# directly (no case file I/O) and pin behavior against hand-computed "golden"
# legacy sequences. They prove that the centralized primitives reproduce the
# exact legacy transformation order and concatenation semantics, with a
# floating-point tolerance small enough to detect any change in the transform
# chain.

import numpy as np
import pytest
from stl import mesh

from pyrpod.vehicle.VisitingVehicle import VisitingVehicle
from pyrpod.util.stl.stl import compose_meshes, transform_mesh


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mesh_from_vectors(vectors):
    vectors = np.asarray(vectors, dtype=float)
    data = np.zeros(len(vectors), dtype=mesh.Mesh.dtype)
    m = mesh.Mesh(data, remove_empty_areas=False)
    m.vectors[:] = vectors
    return m


def _triangle(offset=0.0):
    """A single triangle with distinct, asymmetric vertices."""
    return _mesh_from_vectors([[[1.0 + offset, 0.0, 0.0],
                                [0.0, 2.0 + offset, 0.0],
                                [0.0, 0.0, 3.0 + offset]]])


def _thruster(dcm, exit_coord, name):
    """Mirror the in-memory shape produced by process_thruster_def():
    name is a one-element list, exit is [[x,y,z]], dcm is a 3x3 list."""
    return {
        'name': [name],
        'type': ['001'],
        'exit': [list(exit_coord)],
        'dcm': [list(row) for row in np.asarray(dcm, dtype=float)],
    }


def _cluster(dcm, exit_coord, name):
    return {
        'name': [name],
        'exit': [list(exit_coord)],
        'dcm': [list(row) for row in np.asarray(dcm, dtype=float)],
    }


def _make_vv(thruster_data, cluster_data=None, use_clusters=False):
    vv = VisitingVehicle.__new__(VisitingVehicle)
    vv.thruster_data = thruster_data
    vv.use_clusters = use_clusters
    if cluster_data is not None:
        vv.cluster_data = cluster_data
    vv._thruster_id_map = None
    return vv


# A non-trivial (non-identity, non-symmetric) rotation so that transpose and
# ordering mistakes are detectable.
def _rot_z(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


TOL = 1e-9


# --------------------------------------------------------------------------- #
# transform_plume_mesh: local placement (no vehicle/JFH pose)
# --------------------------------------------------------------------------- #
def test_transform_plume_mesh_local_identity_pose():
    """Identity thruster DCM: only the thruster exit translation applies."""
    exit_c = [1.0, 2.0, 3.0]
    vv = _make_vv({'P1T1': _thruster(np.eye(3), exit_c, 'P1T1')})
    m = _triangle()
    original = m.vectors.copy()

    out = vv.transform_plume_mesh('P1T1', m)

    assert np.allclose(out.vectors, original + np.array(exit_c), atol=TOL)


def test_transform_plume_mesh_local_matches_legacy_sequence():
    """Local mode reproduces legacy: rotate(dcm.T) then translate(exit)."""
    dcm = _rot_z(37.0)
    exit_c = [0.5, -1.5, 4.0]
    vv = _make_vv({'P1T1': _thruster(dcm, exit_c, 'P1T1')})

    actual = vv.transform_plume_mesh('P1T1', _triangle())

    golden = _triangle()
    golden.rotate_using_matrix(np.array(dcm).T)
    golden.translate(exit_c)
    assert np.allclose(actual.vectors, golden.vectors, atol=TOL)


def test_transform_plume_mesh_not_applied_twice():
    """A double application of DCM/exit would land somewhere else entirely."""
    dcm = _rot_z(30.0)
    exit_c = [2.0, 0.0, 0.0]
    vv = _make_vv({'P1T1': _thruster(dcm, exit_c, 'P1T1')})

    single = vv.transform_plume_mesh('P1T1', _triangle())

    doubled = _triangle()
    doubled.rotate_using_matrix(np.array(dcm).T)
    doubled.translate(exit_c)
    doubled.rotate_using_matrix(np.array(dcm).T)
    doubled.translate(exit_c)
    assert not np.allclose(single.vectors, doubled.vectors, atol=1e-6)


def test_transform_plume_mesh_mutates_in_place_and_returns_same_object():
    vv = _make_vv({'P1T1': _thruster(np.eye(3), [1.0, 0.0, 0.0], 'P1T1')})
    m = _triangle()
    before = m.vectors.copy()
    out = vv.transform_plume_mesh('P1T1', m)
    assert out is m
    assert not np.allclose(m.vectors, before)


# --------------------------------------------------------------------------- #
# transform_plume_mesh: complete vehicle/JFH placement
# --------------------------------------------------------------------------- #
def test_transform_plume_mesh_full_placement_matches_legacy_no_clusters():
    """Full placement reproduces the exact legacy graph_jfh inner sequence."""
    dcm = _rot_z(20.0)
    vv_dcm = _rot_x(15.0)
    exit_c = [0.3, 0.4, 0.5]
    vv_pos = [10.0, -5.0, 2.0]
    vv = _make_vv({'P1T1': _thruster(dcm, exit_c, 'P1T1')}, use_clusters=False)

    actual = vv.transform_plume_mesh(
        'P1T1', _triangle(), vv_orientation=vv_dcm, vv_position=vv_pos)

    golden = _triangle()
    golden.rotate_using_matrix(np.array(dcm).T)
    golden.rotate_using_matrix(np.array(vv_dcm).T)
    golden.translate(vv_pos)
    golden.translate(exit_c)
    assert np.allclose(actual.vectors, golden.vectors, atol=TOL)


def test_transform_plume_mesh_exact_operation_order():
    """Rotations do not commute; wrong order gives a different mesh, so this
    pins the exact legacy order (thruster DCM, then VV DCM, then translate)."""
    dcm = _rot_z(50.0)
    vv_dcm = _rot_x(70.0)
    exit_c = [1.0, 0.0, 0.0]
    vv_pos = [0.0, 0.0, 0.0]
    vv = _make_vv({'P1T1': _thruster(dcm, exit_c, 'P1T1')})

    actual = vv.transform_plume_mesh(
        'P1T1', _triangle(), vv_orientation=vv_dcm, vv_position=vv_pos)

    # Swapped rotation order -> genuinely different result.
    wrong = _triangle()
    wrong.rotate_using_matrix(np.array(vv_dcm).T)
    wrong.rotate_using_matrix(np.array(dcm).T)
    wrong.translate(vv_pos)
    wrong.translate(exit_c)
    assert not np.allclose(actual.vectors, wrong.vectors, atol=1e-6)


def test_transform_plume_mesh_clusters_disabled_skips_cluster_offset():
    """use_clusters False: cluster offset must not be applied even in full
    placement mode."""
    dcm = np.eye(3)
    exit_c = [1.0, 1.0, 1.0]
    vv = _make_vv(
        {'P1T1': _thruster(dcm, exit_c, 'P1T1')},
        cluster_data={'P1': _cluster(np.eye(3), [100.0, 0.0, 0.0], 'P1')},
        use_clusters=False,
    )
    vv_pos = [0.0, 0.0, 0.0]

    out = vv.transform_plume_mesh(
        'P1T1', _triangle(), vv_orientation=np.eye(3), vv_position=vv_pos)

    golden = _triangle()  # no cluster offset
    golden.translate(vv_pos)
    golden.translate(exit_c)
    assert np.allclose(out.vectors, golden.vectors, atol=TOL)


def test_transform_plume_mesh_clusters_enabled_applies_cluster_offset():
    dcm = np.eye(3)
    exit_c = [1.0, 1.0, 1.0]
    cluster_exit = [7.0, 8.0, 9.0]
    vv = _make_vv(
        {'P1T1': _thruster(dcm, exit_c, 'P1T1')},
        cluster_data={'P1': _cluster(np.eye(3), cluster_exit, 'P1')},
        use_clusters=True,
    )
    vv_pos = [0.0, 0.0, 0.0]

    out = vv.transform_plume_mesh(
        'P1T1', _triangle(), vv_orientation=np.eye(3), vv_position=vv_pos)

    golden = _triangle()
    golden.translate(vv_pos)
    golden.translate(cluster_exit)  # cluster offset before thruster exit
    golden.translate(exit_c)
    assert np.allclose(out.vectors, golden.vectors, atol=TOL)


def test_transform_plume_mesh_multi_digit_cluster_id():
    """P10T1 must resolve to cluster P10 (legacy [0]+[1] would give P1)."""
    exit_c = [0.0, 0.0, 0.0]
    p10_exit = [10.0, 0.0, 0.0]
    p1_exit = [1.0, 0.0, 0.0]
    vv = _make_vv(
        {'P10T1': _thruster(np.eye(3), exit_c, 'P10T1')},
        cluster_data={
            'P1': _cluster(np.eye(3), p1_exit, 'P1'),
            'P10': _cluster(np.eye(3), p10_exit, 'P10'),
        },
        use_clusters=True,
    )
    vv_pos = [0.0, 0.0, 0.0]

    out = vv.transform_plume_mesh(
        'P10T1', _triangle(), vv_orientation=np.eye(3), vv_position=vv_pos)

    right = _triangle()
    right.translate(p10_exit)
    assert np.allclose(out.vectors, right.vectors, atol=TOL)

    wrong = _triangle()
    wrong.translate(p1_exit)
    assert not np.allclose(out.vectors, wrong.vectors, atol=1e-6)


def test_cluster_id_for_thruster_single_digit_matches_legacy():
    """Single-digit ids resolve exactly as the legacy first-two-char parse."""
    vv = _make_vv(
        {'P1T2': _thruster(np.eye(3), [0, 0, 0], 'P1T2')},
        cluster_data={'P1': _cluster(np.eye(3), [0, 0, 0], 'P1')},
        use_clusters=True,
    )
    assert vv._cluster_id_for_thruster('P1T2') == 'P1'


# --------------------------------------------------------------------------- #
# transform_plume_mesh: error handling / validation
# --------------------------------------------------------------------------- #
def test_transform_plume_mesh_unknown_thruster_raises_keyerror():
    vv = _make_vv({'P1T1': _thruster(np.eye(3), [0, 0, 0], 'P1T1')})
    with pytest.raises(KeyError):
        vv.transform_plume_mesh('NOPE', _triangle())


def test_transform_plume_mesh_missing_cluster_raises_keyerror():
    vv = _make_vv(
        {'P9T1': _thruster(np.eye(3), [0, 0, 0], 'P9T1')},
        cluster_data={'P1': _cluster(np.eye(3), [0, 0, 0], 'P1')},
        use_clusters=True,
    )
    with pytest.raises(KeyError):
        vv.transform_plume_mesh(
            'P9T1', _triangle(), vv_orientation=np.eye(3),
            vv_position=[0.0, 0.0, 0.0])


def test_transform_plume_mesh_bad_thruster_dcm_raises_valueerror():
    vv = _make_vv({'P1T1': {'name': ['P1T1'], 'type': ['001'],
                            'exit': [[0, 0, 0]], 'dcm': [[1, 0], [0, 1]]}})
    with pytest.raises(ValueError):
        vv.transform_plume_mesh('P1T1', _triangle())


def test_transform_plume_mesh_bad_vv_orientation_raises_valueerror():
    vv = _make_vv({'P1T1': _thruster(np.eye(3), [0, 0, 0], 'P1T1')})
    with pytest.raises(ValueError):
        vv.transform_plume_mesh(
            'P1T1', _triangle(), vv_orientation=np.eye(2),
            vv_position=[0.0, 0.0, 0.0])


def test_transform_plume_mesh_bad_vv_position_raises_valueerror():
    vv = _make_vv({'P1T1': _thruster(np.eye(3), [0, 0, 0], 'P1T1')})
    with pytest.raises(ValueError):
        vv.transform_plume_mesh(
            'P1T1', _triangle(), vv_orientation=np.eye(3),
            vv_position=[0.0, 0.0])


# --------------------------------------------------------------------------- #
# JFH thruster-index mapping
# --------------------------------------------------------------------------- #
def _three_thruster_vv(order=('P1T1', 'P1T2', 'P2T1')):
    data = {}
    for name in order:
        data[name] = _thruster(np.eye(3), [0, 0, 0], name)
    return _make_vv(data)


def test_thruster_id_map_normal():
    vv = _three_thruster_vv()
    assert vv.get_thruster_id_map() == {'1': 'P1T1', '2': 'P1T2', '3': 'P2T1'}


def test_thruster_id_canonical_output_is_string():
    vv = _three_thruster_vv()
    assert vv.get_thruster_id(1) == 'P1T1'
    assert vv.get_thruster_id('2') == 'P1T2'


def test_thruster_id_raw_name_field_is_one_element_list():
    vv = _three_thruster_vv()
    # Data representation is unchanged (name is still a one-element list) ...
    assert vv.thruster_data['P1T1']['name'] == ['P1T1']
    # ... while the mapping exposes the canonical id directly.
    assert vv.get_thruster_id(1) == 'P1T1'


def test_thruster_id_preserves_insertion_order():
    vv = _three_thruster_vv(order=('P2T1', 'P1T1', 'P1T2'))
    assert vv.get_thruster_id_map() == {'1': 'P2T1', '2': 'P1T1', '3': 'P1T2'}


def test_thruster_id_reordered_config_gives_different_mapping():
    a = _three_thruster_vv(order=('P1T1', 'P1T2', 'P2T1'))
    b = _three_thruster_vv(order=('P2T1', 'P1T2', 'P1T1'))
    assert a.get_thruster_id_map() != b.get_thruster_id_map()


def test_thruster_id_invalid_index_raises_keyerror():
    vv = _three_thruster_vv()
    with pytest.raises(KeyError):
        vv.get_thruster_id(99)


def test_thruster_id_map_is_cached():
    vv = _three_thruster_vv()
    first = vv.get_thruster_id_map()
    assert vv.get_thruster_id_map() is first


def test_thruster_id_map_invalidated_on_set_thruster_config():
    vv = _three_thruster_vv()
    vv.get_thruster_id_map()  # build + cache
    new_data = {'X1': _thruster(np.eye(3), [0, 0, 0], 'X1'),
                'X2': _thruster(np.eye(3), [0, 0, 0], 'X2')}
    vv.set_thruster_config(thruster_data=new_data)
    assert vv.get_thruster_id_map() == {'1': 'X1', '2': 'X2'}
    assert vv.get_thruster_id(1) == 'X1'


# --------------------------------------------------------------------------- #
# compose_meshes
# --------------------------------------------------------------------------- #
def test_compose_meshes_zero_raises():
    with pytest.raises(ValueError):
        compose_meshes([])


def test_compose_meshes_one_returns_same_object():
    m = _triangle()
    assert compose_meshes([m]) is m


def test_compose_meshes_multiple_preserves_order_and_count():
    a, b, c = _triangle(0.0), _triangle(10.0), _triangle(20.0)
    a_v, b_v, c_v = a.vectors.copy(), b.vectors.copy(), c.vectors.copy()

    out = compose_meshes([a, b, c])

    assert len(out.vectors) == 3
    assert np.allclose(out.vectors[0], a_v, atol=TOL)
    assert np.allclose(out.vectors[1], b_v, atol=TOL)
    assert np.allclose(out.vectors[2], c_v, atol=TOL)


def test_compose_meshes_none_element_raises():
    with pytest.raises(ValueError):
        compose_meshes([_triangle(), None])


def test_compose_meshes_does_not_mutate_inputs():
    a, b = _triangle(0.0), _triangle(5.0)
    a_v, b_v = a.vectors.copy(), b.vectors.copy()
    compose_meshes([a, b])
    assert np.allclose(a.vectors, a_v, atol=TOL)
    assert np.allclose(b.vectors, b_v, atol=TOL)


# --------------------------------------------------------------------------- #
# transform_mesh (reusable mesh-object API; must NOT be the shadowed CLI form)
# --------------------------------------------------------------------------- #
def test_transform_mesh_is_mesh_object_api():
    """Regression for the resolved name collision: the imported transform_mesh
    accepts a mesh object plus rotation/scale kwargs, not (input_file, ...)."""
    import inspect
    params = list(inspect.signature(transform_mesh).parameters)
    assert params == ['mesh_obj', 'rotation_matrix',
                      'translation_vector', 'scale_factor']


def test_transform_mesh_noop_when_all_none_returns_same_object():
    m = _triangle()
    before = m.vectors.copy()
    out = transform_mesh(m)
    assert out is m
    assert np.allclose(out.vectors, before, atol=TOL)


def test_transform_mesh_scale_rotate_translate():
    m = _triangle()
    golden = _triangle()
    dcm = _rot_z(25.0)
    translate = [1.0, 2.0, 3.0]

    out = transform_mesh(m, rotation_matrix=dcm,
                         translation_vector=translate, scale_factor=2.0)

    golden.points *= 2.0
    golden.rotate_using_matrix(dcm)
    golden.translate(translate)
    assert np.allclose(out.vectors, golden.vectors, atol=TOL)

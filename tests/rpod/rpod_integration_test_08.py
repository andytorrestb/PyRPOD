# ========================
# PyRPOD: tests/rpod/rpod_integration_test_08.py
# ========================
# End-to-end geometry-equivalence regressions for the PR #116 plume-mesh
# consolidation refactor. These run the real visualization pipeline on the
# lightweight base_case and pin the produced geometry against an inline
# reproduction of the legacy transform/compose sequence, plus artifact counts
# and file names. A separate regression proves graph_jfh_thruster_check applies
# the thruster transform exactly once (no double application).
#
# base_case has clusters disabled; the cluster branch of graph_jfh is covered
# by the existing rpod_verification_test_04 and by rpod_unit_test_04.

import glob
import os
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from stl import mesh

from pyrpod.rpod import JetFiringHistory, PlumeStrikeEstimationStudy
from pyrpod.rpod import PlumeStrikeEstimationStudy as PSES_module
from pyrpod.vehicle import LogisticsModule, TargetVehicle, VisitingVehicle
from pyrpod.mission import MissionEnvironment
from pyrpod.util.io.fs import resolve_asset_path

BASE_CASE = "../case/rpod/base_case/"

# STL files are binary float32; on base_case's O(1-50) coordinates the
# round-trip quantization is ~1e-5, while any transform-order/double-apply
# regression shifts vertices by O(0.1) or more. 1e-3 detects the latter while
# tolerating the former.
STL_TOL = 1e-3


def _build_study(case_dir=BASE_CASE):
    jfh = JetFiringHistory.JetFiringHistory(case_dir)
    jfh.read_jfh()
    tv = TargetVehicle.TargetVehicle(case_dir)
    tv.set_stl()
    lm = LogisticsModule.LogisticsModule(case_dir)
    lm.set_thruster_config()
    me = MissionEnvironment.MissionEnvironment(case_dir)
    study = PlumeStrikeEstimationStudy.PlumeStrikeEstimationStudy(me)
    study.study_init(jfh, tv, lm)
    return study


def _legacy_link(vv):
    link = {}
    i = 1
    for thruster in vv.thruster_data:
        link[str(i)] = vv.thruster_data[thruster]['name']
        i += 1
    return link


def _legacy_firing_mesh(study, firing):
    """Reproduce, inline, the exact pre-refactor graph_jfh geometry for one
    firing of a clusters-disabled case (VV body + concatenated plume cones)."""
    vv = study.vv
    config = study.environment.config
    case_dir = study.environment.case_dir
    link = _legacy_link(vv)

    VVmesh = mesh.Mesh.from_file(
        resolve_asset_path(case_dir, 'stl', config['vv']['stl_lm']))
    vv_orientation = np.array(study.jfh.JFH[firing]['dcm'])
    VVmesh.rotate_using_matrix(vv_orientation.transpose())
    VVmesh.translate(study.jfh.JFH[firing]['xyz'])

    active_cones = None
    for thruster in study.jfh.JFH[firing]['thrusters']:
        thruster_id = link[str(thruster)][0]
        plumeMesh = mesh.Mesh.from_file(
            resolve_asset_path(case_dir, 'stl', config['vv']['stl_thruster']))
        thruster_orientation = np.array(vv.thruster_data[thruster_id]['dcm'])
        plumeMesh.rotate_using_matrix(thruster_orientation.transpose())
        plumeMesh.rotate_using_matrix(vv_orientation.transpose())
        plumeMesh.translate(study.jfh.JFH[firing]['xyz'])
        plumeMesh.translate(vv.thruster_data[thruster_id]['exit'][0])
        if active_cones is None:
            active_cones = plumeMesh
        else:
            active_cones = mesh.Mesh(
                np.concatenate([active_cones.data, plumeMesh.data]))

    if active_cones is None:
        return VVmesh
    return mesh.Mesh(np.concatenate([VVmesh.data, active_cones.data]))


def _jfh_dir(study):
    return os.path.join(study.environment.case_dir, "results", "jfh")


# --------------------------------------------------------------------------- #
# graph_jfh: geometry equivalence + artifact count/names
# --------------------------------------------------------------------------- #
def test_graph_jfh_geometry_matches_legacy_and_artifact_count():
    study = _build_study()
    n_firings = len(study.jfh.JFH)

    jfh_dir = _jfh_dir(study)
    if os.path.isdir(jfh_dir):
        shutil.rmtree(jfh_dir)

    study.graph_jfh()

    # One STL artifact per firing, named firing-{n}.stl.
    produced = sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(jfh_dir, "firing-*.stl")))
    expected_names = sorted(f"firing-{i}.stl" for i in range(n_firings))
    assert produced == expected_names

    # Geometry of each firing matches the legacy sequence exactly.
    for firing in range(n_firings):
        actual = mesh.Mesh.from_file(
            os.path.join(jfh_dir, f"firing-{firing}.stl"))
        expected = _legacy_firing_mesh(study, firing)
        assert len(actual.vectors) == len(expected.vectors)
        assert np.allclose(actual.vectors, expected.vectors, atol=STL_TOL)


def test_graph_jfh_firing_face_count_is_vv_plus_active_plumes():
    """Composition adds exactly VV faces + (n_active x plume faces) per firing,
    with no cluster geometry when clusters are disabled."""
    study = _build_study()
    config = study.environment.config
    case_dir = study.environment.case_dir

    vv_faces = len(mesh.Mesh.from_file(
        resolve_asset_path(case_dir, 'stl', config['vv']['stl_lm'])).vectors)
    plume_faces = len(mesh.Mesh.from_file(
        resolve_asset_path(case_dir, 'stl', config['vv']['stl_thruster'])).vectors)

    jfh_dir = _jfh_dir(study)
    if os.path.isdir(jfh_dir):
        shutil.rmtree(jfh_dir)
    study.graph_jfh()

    for firing in range(len(study.jfh.JFH)):
        n_active = len(study.jfh.JFH[firing]['thrusters'])
        actual = mesh.Mesh.from_file(
            os.path.join(jfh_dir, f"firing-{firing}.stl"))
        assert len(actual.vectors) == vv_faces + n_active * plume_faces


# --------------------------------------------------------------------------- #
# visualize_sweep: produces the same per-firing geometry as graph_jfh
# --------------------------------------------------------------------------- #
def test_visualize_sweep_geometry_matches_legacy():
    study = _build_study()
    study.count = 0  # single-config sweep -> firing-{n}.stl naming

    jfh_dir = _jfh_dir(study)
    if os.path.isdir(jfh_dir):
        shutil.rmtree(jfh_dir)

    study.visualize_sweep(0)

    for firing in range(len(study.jfh.JFH)):
        actual = mesh.Mesh.from_file(
            os.path.join(jfh_dir, f"firing-{firing}.stl"))
        expected = _legacy_firing_mesh(study, firing)
        assert len(actual.vectors) == len(expected.vectors)
        assert np.allclose(actual.vectors, expected.vectors, atol=STL_TOL)


# --------------------------------------------------------------------------- #
# graph_jfh_thruster_check: thruster transform applied exactly once
# --------------------------------------------------------------------------- #
def _one_triangle():
    data = np.zeros(1, dtype=mesh.Mesh.dtype)
    m = mesh.Mesh(data, remove_empty_areas=False)
    m.vectors[:] = np.array([[[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]]])
    return m


def test_graph_jfh_thruster_check_applies_thruster_transform_once(monkeypatch):
    # A visiting vehicle with a single thruster and a non-trivial DCM/exit.
    vv = VisitingVehicle.VisitingVehicle.__new__(VisitingVehicle.VisitingVehicle)
    vv.thruster_data = {
        'P1T1': {
            'name': ['P1T1'], 'type': ['001'],
            'exit': [[0.3, -0.4, 0.5]],
            'dcm': [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
    vv.use_clusters = False
    vv._thruster_id_map = None

    # Spy on the canonical placement method: it must be called exactly once for
    # the single active thruster, and in LOCAL mode (no vehicle/JFH pose), so
    # the thruster DCM/exit are applied a single time.
    calls = []
    real_transform = vv.transform_plume_mesh

    def spy(thruster_id, plumeMesh, vv_orientation=None, vv_position=None):
        calls.append((thruster_id, vv_orientation, vv_position))
        return real_transform(thruster_id, plumeMesh,
                              vv_orientation=vv_orientation,
                              vv_position=vv_position)

    vv.transform_plume_mesh = spy

    study = PlumeStrikeEstimationStudy.PlumeStrikeEstimationStudy.__new__(
        PlumeStrikeEstimationStudy.PlumeStrikeEstimationStudy)
    study.vv = vv
    study.jfh = SimpleNamespace(JFH=[{'thrusters': [1]}])
    study.environment = SimpleNamespace(case_dir="scratch/")
    study.viz = SimpleNamespace(save_figure=lambda figure, path: None)

    # Avoid file I/O and real rendering: stub STL loading and matplotlib.
    monkeypatch.setattr(PSES_module, 'load_stl', lambda path: _one_triangle())
    monkeypatch.setattr(PSES_module, 'plt', MagicMock())
    monkeypatch.setattr(PSES_module, 'mplot3d', MagicMock())

    study.graph_jfh_thruster_check()

    assert calls == [('P1T1', None, None)]

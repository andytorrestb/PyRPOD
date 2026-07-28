from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
import os
import math
import logging
import time
import tracemalloc

from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from pyrpod.vehicle.LogisticsModule import LogisticsModule
from pyrpod.mission.MissionPlanner import MissionPlanner
from pyrpod.plume.RarefiedPlumeGasKinetics import SimplifiedGasKinetics

from pyrpod.util.io.file_print import print_1d_JFH
from pyrpod.util.io.fs import ensure_dir, resolve_asset_path
from pyrpod.util.stl.stl import load_stl, transform_mesh, compose_meshes

from queue import Queue

from pyrpod.logging_utils import (
    get_active_session,
    get_settings,
    log_array_summary,
    performance_logging_enabled,
)
from pyrpod.util.math.transform import rotation_matrix_from_vectors

# New modular imports for refactor
from pyrpod.rpod.approach_maneuvers import (
    ApproachInputs,
    compute_1d_approach,
    validate_n_firings,
)
from pyrpod.rpod.io import ensure_results_dirs, write_jfh
from pyrpod.rpod.PlumeStudyExport import PlumeStudyExport
from pyrpod.plume.PlumeStrikeCalculator import (
    compute_face_centroids,
    compute_plume_strikes,
    extract_plume_params,
    run_parallel_plume_strikes,
)

# Aliased because study_init's parameter names shadow the class names.
from pyrpod.rpod.JetFiringHistory import (
    JetFiringHistory as JetFiringHistoryType,
)
from pyrpod.vehicle.TargetVehicle import (
    TargetVehicle as TargetVehicleType,
)

logger = logging.getLogger(__name__)


def _log_jfh_generation_complete(jfh_path: str, n_firings: int,
                                 gen_start: float) -> None:
    """Log completion of a JFH-generation step (path, size, count, runtime).

    Never raises: JFH generation is a scientific write, so an observability
    failure must not mask a successful (or failed) file write.
    """
    gen_s = time.perf_counter() - gen_start
    try:
        size = os.path.getsize(jfh_path)
    except OSError:
        size = -1
    logger.info("JFH generation completed: firings=%d output=%s "
                "output_size_bytes=%d generation_time_s=%.4f",
                n_firings, os.path.abspath(jfh_path), size, gen_s)


class PlumeStrikeEstimationStudy (MissionPlanner):
    """
        Class responsible for analyzing RPOD performance of visiting vehicles.

        Caculated metrics (outputs) include propellant usage, plume impingement,
        trajectory character, and performance with respect to factors of safety.

        Data Inputs inlcude (redundant? better said in user guide?)
        1. LogisticsModule (LM) object with properly defined RCS configuration
        2. Jet Firing History including LM location and orientation with repsect to the Gateway.
        3. Selected plume models for impingement analysis.
        4. Surface mesh data for target and visiting vehicle.

        Attributes
        ----------

        vv : LogisticsModule
            Visiting vehicle of interest. Includes complete RCS configuration and surface mesh data.

        jfh : JetFiringHistory
            Includes VV location and orientation with respect to the TV.

        plume_model : PlumeModel
            Contains the relevant governing equations selected for analysis.

        Methods
        -------
        study_init(self, JetFiringHistory, Target, Vehicle)
            Designates assets for RPOD analysis.
        
        graph_init_config(self)
            Creates visualization data for initiial configuration of RPOD analysis.

        graph_jfh_thruster_check(self)
            Creates visualization data for initiial configuration of RPOD analysis.
        
        graph_clusters(self, firing, vv_orientation)
            Creates visualization data for the cluster.
        
        graph_jfh(self)
            Creates visualization data for the trajectory of the proposed RPOD analysis.

        update_window_queue(self, window_queue, cur_window, firing_time, window_size)
            Takes the most recent window of time size, and adds the new firing time to the sum, and the window_queue.
            If the new window is larger than the allowed window_size, then earliest firing times are removed
            from the queue and subtracted from the cur_window sum, until the sum fits within the window size.
            A counter for how many firing times are removed and subtracted is recorded.

        update_parameter_queue(self, param_queue, param_window_sum, get_counter)
            Takes the current parameter_queue, and removes the earliest tracked parameters from the front of the queue.
            This occurs "get_counter" times. Each time a parameter is popped from the queue, the sum is also updated,
            as to not track the removed parameter (ie. subtract the value)

        jfh_plume_strikes(self)
            Calculates number of plume strikes according to data provided for RPOD analysis.
            Method does not take any parameters but assumes that study assets are correctly configured.
            These assets include one JetFiringHistory, one TargetVehicle, and one VisitingVehicle.
            A Simple plume model is used. It does not calculate plume physics, only strikes. which
            are determined with a user defined "plume cone" geometry. Simple vector mathematics is
            used to determine if an VTK surface elements is struck by the "plume cone".
        
        print_jfh_1d_approach(v_ida, v_o, r_o)
            Method creates JFH data for axial approach using simpified physics calculations.
    """
    # def __init__(self):
    #     print("Initialized Approach Visualizer")
    # Collaborator state and thruster-group / fuel helpers that this
    # class calls on self but neither it nor MissionPlanner defines.
    # Declared bare, so no class attribute is created at runtime.
    jfh: JetFiringHistoryType
    target: TargetVehicleType
    vv: LogisticsModule
    viz: PlumeStudyExport
    case_key: str
    calc_m_dot_sum: Callable[[str], float]
    calc_v_e: Callable[[str], float]
    calc_delta_v: Callable[[float, float, float, float], float]
    calc_delta_mass_v_e: Callable[[float, float, bool], float]
    # Sweep index set externally by mdao.TradeStudy before visualize_sweep.
    count: int

    def study_init(self, JetFiringHistory: JetFiringHistoryType,
                   Target: TargetVehicleType,
                   Vehicle: LogisticsModule) -> None:
        """
            Designates assets for RPOD analysis.

            Parameters
            ----------
            JetFiringHistory : JetFiringHistory
                Object thruster firing history. It includes VV position, orientation, and IDs for active thrusters.

            Target : TargetVehicle
                Object containing surface mesh and thruster configurations for the Visiting Vehicle.

            Vehicle : VisitingVehicle
                Object containing surface mesh and surfave properties for the Target Vehicle.

            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
            Does the method need to return a status message? or pass similar data?
        """
        self.jfh = JetFiringHistory
        self.target = Target
        self.vv = Vehicle
        # visualization/export helper
        self.viz = PlumeStudyExport(self.environment)

    def graph_init_config(self) -> None:
        """
            Creates visualization data for initiial configuration of RPOD analysis.

            NOTE: Method does not take any parameters. It assumes that self.environment.case_dir
            and self.environment.config are instatiated correctly. Potential defensive programming statements?

            TODO: Needs to be re-factored to save VTK data in proper case directory.

            Returns
            -------
            Method doesn't currently return anything. Simply produces data as needed.
            Does the method need to return a status message? or pass similar data?
        """

        # Save first coordinate in the JFH
        vv_initial_firing = self.jfh.JFH[0]
        # Log initial configuration details for debugging
        logger.debug("Initial VV firing position: %s", vv_initial_firing['xyz'])

        # Translate VV STL to first coordinate
        self.vv.mesh.translate(vv_initial_firing['xyz'])

        # Combine target and VV STLs into one "Mesh" object.
        combined = mesh.Mesh(np.concatenate(
            [self.target.mesh.data, self.vv.mesh.data]
        ))

        figure = plt.figure()
        # axes = mplot3d.Axes3D(figure)
        axes = figure.add_subplot(projection='3d')
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(combined.vectors))
        # axes.quiver(X, Y, Z, U, V, W, color=(0,0,0), length=1, normalize=True)
        lim = 100
        axes.set_xlim([-1 * lim, lim])
        axes.set_ylim([-1 * lim, lim])
        axes.set_zlim([-1 * lim, lim])
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        # figure.suptitle(str(i))
        plt.show()


    def graph_jfh_thruster_check(self) -> None:
        """
            Creates visualization data for initiial configuration of RPOD analysis.

            NOTE: Method does not take any parameters. It assumes that self.environment.case_dir
            and self.environment.config are instatiated correctly. Potential defensive programming statements?

            TODO: Needs to be re-factored to save VTK data in proper case directory.

            Returns
            -------
            Method doesn't currently return anything. Simply produces data as needed.
            Does the method need to return a status message? or pass similar data?
        """

        # JFH numeric thruster indices are resolved to canonical thruster ids
        # via the cached mapping owned by the visiting vehicle
        # (VisitingVehicle.get_thruster_id) rather than rebuilt here.

        # Loop through each firing in the JFH.
        for firing in range(len(self.jfh.JFH)):

            # Save active thrusters for current firing.
            thrusters = self.jfh.JFH[firing]['thrusters']

            # Fixed demo geometry for the RCS sanity check. These paths select
            # dedicated check STLs (cylinder/mold_funnel), NOT the case's
            # configured LM/thruster STLs, so they are intentionally left as
            # literal paths rather than routed through resolve_asset_path.
            VVmesh = load_stl('../stl/cylinder.stl')

            figure = plt.figure()
            # axes = mplot3d.Axes3D(figure)
            axes = figure.add_subplot(projection = '3d')
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(VVmesh.vectors))
 
            # Load and graph STLs of active thrusters.
            for thruster in thrusters:
                # Map thruster ID
                thruster_id = self.vv.get_thruster_id(thruster)

                # Load plume STL in initial configuration.
                plumeMesh = load_stl('../stl/mold_funnel.stl')

                # Tranform plume into initial configuration.
                # TODO: edit mold_funnel.stl to not require these transforms.
                rot_mat = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ])
                plumeMesh = transform_mesh(
                    plumeMesh,
                    rotation_matrix=rot_mat,
                    translation_vector=[0, 0, -50],
                    scale_factor=0.05
                )

                # Place the plume at the thruster (local placement: thruster DCM
                # + thruster exit only -- no vehicle/JFH pose, no cluster).
                # Routed through the canonical transform_plume_mesh so the
                # thruster DCM and exit are applied exactly once each.
                plumeMesh = self.vv.transform_plume_mesh(thruster_id, plumeMesh)

                logger.debug("Thruster %s DCM: %s", thruster_id, self.vv.thruster_data[thruster_id]['dcm'])

                # Add surface to graph.
                surface = mplot3d.art3d.Poly3DCollection(plumeMesh.vectors)
                surface.set_facecolor('orange')
                axes.add_collection3d(surface)

            lim = 7
            axes.set_xlim([-1*lim - 3, lim - 3])
            axes.set_ylim([-1*lim, lim])
            axes.set_zlim([-1*lim, lim])
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel('Z')
            shift=0
            axes.view_init(azim=0, elev=2*shift)


            logger.debug("Completed plotting thruster check for firing %d", firing)

            if firing < 10:
                index = '00' + str(firing)
            elif firing < 100:
                index = '0' + str(firing)
            else:
                # Pre-existing bug: `i` is not defined in this scope, so a
                # 100th-or-later firing raises NameError. Flagged, not
                # fixed (also reported by flake8 as F821).
                index = str(i)  # type: ignore[name-defined]
            # delegate figure saving to export helper
            self.viz.save_figure(figure, os.path.join(self.environment.case_dir, 'img', 'frame' + str(index) + '.png'))

    def graph_clusters(
        self, firing: int, vv_orientation: NDArray[np.float64]
    ) -> mesh.Mesh | None:
        """
            Creates visualization data for the cluster.
            Parameters
            ----------
            firing : int
                Loop iterable over the length of the number of thrusters firing in the JFH.
            
            vv_orientation : np.array
                DCM from the JFH.
            Returns
            -------
            active_clusters : mesh
                Cluster of the current thruster firing.
        """
        clusters_list = []
        for number in range(len(self.vv.cluster_data)):
            cluster_name = 'P' + str(number + 1)
            # print(cluster_name)
            clusters_list.append(cluster_name)

        # print('clusters_list is', clusters_list)
        # Load and graph STLs of active clusters.
        cluster_meshes = []
        for cluster in clusters_list:

            # Load plume STL in initial configuration.
            clusterMesh = mesh.Mesh.from_file(resolve_asset_path(self.environment.case_dir, 'stl', self.environment.config['vv']['stl_cluster']))

            # Transform cluster

            # First, according to DCM of current cluster in CCF
            cluster_orientation = np.array(
                self.vv.cluster_data[cluster]['dcm']
            )
            clusterMesh.rotate_using_matrix(cluster_orientation.transpose())

            # Second, according to DCM of VV in JFH
            clusterMesh.rotate_using_matrix(vv_orientation.transpose())

            # Third, according to position vector of the VV in JFH
            clusterMesh.translate(self.jfh.JFH[firing]['xyz'])

            # Fourth, according to position of current cluster in CCF
            clusterMesh.translate(self.vv.cluster_data[cluster]['exit'][0])
            # print(self.vv.cluster_data[cluster]['exit'][0])

            cluster_meshes.append(clusterMesh)

        # No clusters configured -> no cluster geometry (legacy returned None).
        if not cluster_meshes:
            return None
        return compose_meshes(cluster_meshes)

    def graph_jfh(self, trade_study: bool = False) -> None:
        """
            Creates visualization data for the trajectory of the proposed RPOD analysis.

            This method does NOT calculate plume strikes.

            This utilities allows engineers to visualize the trajectory in the JFH before running
            the full simulation and wasting computation time.
            Returns
            -------
            Method doesn't currently return anything. Simply produces data as needed.
            Does the method need to return a status message? or pass similar data?
        """
        # JFH numeric thruster indices are resolved to canonical thruster ids
        # via the cached mapping owned by the visiting vehicle
        # (VisitingVehicle.get_thruster_id) rather than rebuilt here.

        # Create results directory if it doesn't already exist.
        results_dir = self.environment.case_dir + 'results'
        ensure_dir(results_dir)

        if not trade_study:
            results_dir = results_dir + "/jfh"
            ensure_dir(results_dir)

        if trade_study:
            v_o = ['vo_0', 'vo_1', 'vo_2', 'vo_3', 'vo_4']
            cants = ['cant_0', 'cant_1', 'cant_2', 'cant_3', 'cant_4']
            for v in v_o:
                for cant in cants:
                    results_dir_case = results_dir + "/" + v + '_' + cant
                    ensure_dir(results_dir_case)

        logger.info("JFH visualization export started: firings=%d "
                    "trade_study=%s", len(self.jfh.JFH), trade_study)
        viz_files_written = 0
        viz_failures = 0

        # Save STL surface of target vehicle to local variable.
        target = self.target.mesh

        # Loop through each firing in the JFH.
        for firing in range(len(self.jfh.JFH)):
            # print('firing =', firing+1)

            # Save active thrusters for current firing. 
            thrusters = self.jfh.JFH[firing]['thrusters']
            # print("thrusters", thrusters)
             
            # Load, transform, and, graph STLs of visiting vehicle.  
            VVmesh = mesh.Mesh.from_file(resolve_asset_path(self.environment.case_dir, 'stl', self.environment.config['vv']['stl_lm']))
            vv_orientation = np.array(self.jfh.JFH[firing]['dcm'])
            # print(vv_orientation.transpose())
            VVmesh.rotate_using_matrix(vv_orientation.transpose())
            VVmesh.translate(self.jfh.JFH[firing]['xyz'])

            # Optional per-firing mesh components are initialized explicitly
            # (not probed with locals()) so composition below is unambiguous.
            active_cones = None
            active_clusters = None

            # Load and graph STLs of active clusters.
            if self.vv.use_clusters == True:
                active_clusters = self.graph_clusters(firing, vv_orientation)

            # Load and place the plume STL for each active thruster. The exact
            # legacy placement sequence (thruster DCM, VV orientation, VV
            # position, cluster offset, thruster exit) is owned by
            # VisitingVehicle.transform_plume_mesh.
            plume_meshes = []
            for thruster in thrusters:
                thruster_id = self.vv.get_thruster_id(thruster)

                # Load plume STL in initial configuration.
                plumeMesh = mesh.Mesh.from_file(resolve_asset_path(self.environment.case_dir, 'stl', self.environment.config['vv']['stl_thruster']))

                plumeMesh = self.vv.transform_plume_mesh(
                    thruster_id, plumeMesh,
                    vv_orientation=vv_orientation,
                    vv_position=self.jfh.JFH[firing]['xyz'],
                )

                plume_meshes.append(plumeMesh)

            if plume_meshes:
                active_cones = compose_meshes(plume_meshes)

            # Combine the VV body with the active plume cones and, when clusters
            # are enabled, the cluster geometry. When there are no active cones
            # the VV body is emitted unchanged and cluster geometry is
            # intentionally omitted -- exactly as legacy behavior did.
            components = [VVmesh]
            if active_cones is not None:
                components.append(active_cones)
                if self.vv.use_clusters == True:
                    # graph_clusters returns None only when no clusters are
                    # configured, which use_clusters already excludes.
                    components.append(active_clusters)  # type: ignore[arg-type]
            VVmesh = compose_meshes(components)

            if trade_study == False:
                path_to_stl = os.path.join(self.environment.case_dir, "results", "jfh", f"firing-{firing}.stl")
            elif trade_study == True:
                path_to_stl = os.path.join(self.environment.case_dir, "results", self.get_case_key(), "jfh", f"firing-{firing}.stl")

            # self.vv.convert_stl_to_vtk(path_to_vtk, mesh =VVmesh)
            # The JFH trajectory STLs are an optional visualization aid (viewed
            # in ParaView); a write failure must warn and continue rather than
            # abort the run, since the numerical strike calculation does not
            # depend on them.
            try:
                self.viz.export_firing(VVmesh, path_to_stl)
                viz_files_written += 1
                logger.debug("Wrote JFH firing artifact: %s", path_to_stl)
            except Exception:
                viz_failures += 1
                logger.warning("Optional JFH visualization write failed for "
                               "firing %d (%s); continuing.", firing,
                               path_to_stl, exc_info=True)

        jfh_export_dir = os.path.join(self.environment.case_dir, "results", "jfh")
        logger.info("Completed JFH visualization export: files_written=%d "
                    "failures=%d directory=%s", viz_files_written,
                    viz_failures, jfh_export_dir)

    def visualize_sweep(self, config_iter: int) -> None:
        """
            Creates visualization data for the trajectory of the proposed RPOD analysis.

            This method is valid for SINGLE JFH firings, due to file naming conventions.
            Numbering of files is based on the iteration number of the current configuration provided.

            Method used for mdao_unit_test_02.py

            This utility allows engineers to visualize a configuration sweep before running
            the full simulation and wasting computational resources.
            Parameters
            ----------
            None

            Returns
            -------
            Method doesn't currently return anything. Simply produces data as needed.
            Does the method need to return a status message? or pass similar data?
        """
        # JFH numeric thruster indices are resolved to canonical thruster ids
        # via the cached mapping owned by the visiting vehicle
        # (VisitingVehicle.get_thruster_id) rather than rebuilt here.

        # Create results directory if it doesn't already exist.
        results_dir = self.environment.case_dir + 'results'
        ensure_dir(results_dir)

        results_dir = results_dir + "/jfh"
        ensure_dir(results_dir)

        # Save STL surface of target vehicle to local variable.
        target = self.target.mesh

        # Loop through each firing in the JFH.
        for firing in range(len(self.jfh.JFH)):
            # print('firing =', firing+1)

            # Save active thrusters for current firing.
            thrusters = self.jfh.JFH[firing]['thrusters']
            # print("thrusters is", thrusters)
             
            # Load, transform, and, graph STLs of visiting vehicle.  
            VVmesh = mesh.Mesh.from_file(resolve_asset_path(self.environment.case_dir, 'stl', self.environment.config['vv']['stl_lm']))
            vv_orientation = np.array(self.jfh.JFH[firing]['dcm'])
            # print(vv_orientation.transpose())
            VVmesh.rotate_using_matrix(vv_orientation.transpose())
            VVmesh.translate(self.jfh.JFH[firing]['xyz'])

            # Optional per-firing mesh components are initialized explicitly
            # (not probed with locals()) so composition below is unambiguous.
            active_cones = None
            active_clusters = None

            # Load and graph STLs of active clusters.
            if self.vv.use_clusters == True:
                active_clusters = self.graph_clusters(firing, vv_orientation)

            # Load and place the plume STL for each active thruster. The exact
            # legacy placement sequence (thruster DCM, VV orientation, VV
            # position, cluster offset, thruster exit) is owned by
            # VisitingVehicle.transform_plume_mesh.
            plume_meshes = []
            for thruster in thrusters:
                thruster_id = self.vv.get_thruster_id(thruster)

                # Load plume STL in initial configuration.
                plumeMesh = mesh.Mesh.from_file(resolve_asset_path(self.environment.case_dir, 'stl', self.environment.config['vv']['stl_thruster']))

                plumeMesh = self.vv.transform_plume_mesh(
                    thruster_id, plumeMesh,
                    vv_orientation=vv_orientation,
                    vv_position=self.jfh.JFH[firing]['xyz'],
                )

                plume_meshes.append(plumeMesh)

            if plume_meshes:
                active_cones = compose_meshes(plume_meshes)

            # Combine the VV body with the active plume cones and, when clusters
            # are enabled, the cluster geometry. When there are no active cones
            # the VV body is emitted unchanged and cluster geometry is
            # intentionally omitted -- exactly as legacy behavior did.
            components = [VVmesh]
            if active_cones is not None:
                components.append(active_cones)
                if self.vv.use_clusters == True:
                    # See graph_jfh: None only without configured clusters.
                    components.append(active_clusters)  # type: ignore[arg-type]
            VVmesh = compose_meshes(components)

            if self.count > 0:
                path_to_vtk = os.path.join(self.environment.case_dir, "results", "strikes", f"firing-{self.count}-{firing}")
                path_to_stl = os.path.join(self.environment.case_dir, "results", "jfh", f"firing-{self.count}-{firing}.stl")
            else:
                path_to_vtk = os.path.join(self.environment.case_dir, "results", "jfh", f"firing-{firing}")
                path_to_stl = os.path.join(self.environment.case_dir, "results", "jfh", f"firing-{firing}.stl")
            # self.vv.convert_stl_to_vtk(path_to_vtk, mesh =VVmesh)
            self.viz.export_firing(VVmesh, path_to_stl)

    # def update_window_queue(self, window_queue, cur_window, firing_time, window_size):
    #     """
    #         Takes the most recent window of time size, and adds the new firing time to the sum, and the window_queue.
    #         If the new window is larger than the allowed window_size, then earliest firing times are removed
    #         from the queue and subtracted from the cur_window sum, until the sum fits within the window size.
    #         A counter for how many firing times are removed and subtracted is recorded.

    #         Parameters
    #         ----------
    #         window_queue : Queue
    #                 Queue holding the tracked firing times
    #         cur_window : float
    #                 sum of the tracked firing times (s)
    #         firing_time : float
    #                 length of current firing (s)
    #         window_size : float
    #                 max length of a window of time to track (s)

    #         Returns
    #         -------
    #         Queue
    #             Stores tracked firing times after update
    #         float
    #             sum of tracked firing times after update
    #         int
    #             number of firings removed from the queue 
    #     """
    #     window_queue.put(firing_time)
    #     cur_window += firing_time
    #     get_counter = 0
    #     while cur_window > window_size:
    #         old_firing_time = window_queue.get()
    #         cur_window -= old_firing_time
    #         get_counter +=1
    #     return window_queue, cur_window, get_counter
    
    # def update_parameter_queue(self, param_queue, param_window_sum, get_counter):
    #     """
    #         Takes the current parameter_queue, and removes the earliest tracked parameters from the front of the queue.
    #         This occurs "get_counter" times. Each time a parameter is popped from the queue, the sum is also updated,
    #         as to not track the removed parameter (ie. subtract the value)

    #         Parameters
    #         ----------
    #         param_queue : Queue
    #                 Queue holding the tracked parameters per firing
    #         param_window_sum : float
    #                 sum of the parameters in all the tracked firings
    #         get_counter : int
    #                 number of times to remove a tracked parameter from the front of the queue

    #         Returns
    #         -------
    #         Queue
    #             stores tracked paramters per firing after update
    #         float
    #             sum of tracked parameter after update
    #     """
    #     for i in range(get_counter):
    #         old_param = param_queue.get()
    #         param_window_sum -= old_param
    #     return param_queue, param_window_sum

    # Helper functions for jfh_plume_strikes
    def create_results_dir(self) -> None:
        """
            Creates a results directory and sub-directories if they don't already exist.
        """
        sub_dirs = ['results', 'results/strikes', 'results/jfh']
        for sub_dir in sub_dirs:
            ensure_dir(os.path.join(self.environment.case_dir, sub_dir))

    def set_strike_fields(
        self, target: mesh.Mesh
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64],
              NDArray[np.float64], NDArray[np.float64]]
        | NDArray[np.float64]
    ):
        # Initiate array containing cummulative strikes. 
        cum_strikes = np.zeros(len(target.vectors))

        # using plume physics?
        if self.environment.config['pm']['kinetics'] != 'None':

            # Initiate array containing max pressures induced on each element.
            max_pressures = np.zeros(len(target.vectors))

            # Initiate array containing max shears induced on each element.
            max_shears = np.zeros(len(target.vectors))

            # Initiate array containing cummulative heatflux.
            cum_heat_flux_load = np.zeros(len(target.vectors))

            return cum_strikes, max_pressures, max_shears, cum_heat_flux_load

        return cum_strikes

    def extract_firing_data(
        self, firing: int
    ) -> tuple[list[int], Sequence[float], NDArray[np.float64]]:
        # Save active thrusters for current firing.
        thrusters = self.jfh.JFH[firing]['thrusters']
        # print("thrusters", thrusters)

        # Load visiting vehicle position and orientation
        vv_pos = self.jfh.JFH[firing]['xyz']

        vv_orientation = np.array(self.jfh.JFH[firing]['dcm']).transpose()

        return thrusters, vv_pos, vv_orientation

    def set_plume_strike_fields(
        self, target: mesh.Mesh
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64],
              NDArray[np.float64], NDArray[np.float64],
              NDArray[np.float64]]
        | NDArray[np.float64]
    ):
            # reset strikes for each firing
            strikes = np.zeros(len(target.vectors))

            if self.environment.config['pm']['kinetics'] != 'None':
                # reset pressures for each firing
                pressures = np.zeros(len(target.vectors))

                # reset shear pressures for each firing
                shear_stresses = np.zeros(len(target.vectors))

                # reset heat fluxes for each firing
                heat_flux = np.zeros(len(target.vectors))
                heat_flux_load = np.zeros(len(target.vectors))
                return strikes, pressures, shear_stresses, heat_flux, heat_flux_load
            else:
                return strikes
    
    def set_plume_transformations(
        self, thruster_id: str, vv_orientation: NDArray[np.float64],
        vv_pos: Sequence[float] | NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64]]:
        # Load data to calculate plume transformations

        # First, according to DCM and exit vector using current thruster id in TCD
        thruster_orientation = np.array(
            self.vv.thruster_data[thruster_id]['dcm']
        ).transpose()

        thruster_orientation =  thruster_orientation.dot(vv_orientation)
        # print('DCM: ', self.vv.thruster_data[thruster_id]['dcm'])
        # print('DCM: ', thruster_orientation[0], thruster_orientation[1], thruster_orientation[2])
        plume_normal = np.array(thruster_orientation[0])
        # print("plume normal: ", plume_normal)
        
        # calculate thruster exit coordinate with respect to the Target Vehicle.
        
        # print(self.vv.thruster_data[thruster_id])
        thruster_pos = vv_pos + np.array(self.vv.thruster_data[thruster_id]['exit'])
        thruster_pos = thruster_pos[0]
        # print('thruster position', thruster_pos)

        return plume_normal, thruster_pos, thruster_orientation

    def set_face_centroid(
        self, face: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Calculate centroid for face

        x = np.array(face[0]).mean()
        y = np.array(face[1]).mean()
        z = np.array(face[2]).mean()

        centroid = np.array([x, y, z])

        return centroid

    def set_face_distance(
        self, thruster_pos: NDArray[np.float64],
        centroid: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], Any, NDArray[np.float64]]:
        # Calculate distance vector between face centroid and thruster exit.
        distance = thruster_pos - centroid
        # print('distance vector', distance)
        norm_distance  = np.sqrt(distance[0]**2 + distance[1]**2 + distance[2]**2)

        unit_distance = distance / norm_distance
        # print('distance magnitude', norm_distance)

        return distance, norm_distance, unit_distance

    def _resolve_parallel_options(
        self, parallel: bool | None, workers: int | None, n_firings: int
    ) -> tuple[bool, int]:
        """
            Resolves parallel execution settings for jfh_plume_strikes().

            Precedence: explicit method arguments override the optional
            [exec] config section, which defaults to serial execution.

            Config keys (both optional):
            - [exec] parallel : bool — enable process-based parallelization
              across firings (default false).
            - [exec] workers : int — number of worker processes. Defaults to
              min(os.cpu_count(), n_firings) when parallel is enabled.

            Returns
            -------
            (bool, int)
                (parallel_enabled, workers) — workers is capped at n_firings;
                workers <= 1 resolves to serial execution.
        """
        config = self.environment.config
        if parallel is None:
            try:
                parallel = config.getboolean('exec', 'parallel', fallback=False)
            except ValueError as exc:
                raise ValueError(
                    "Invalid config value for [exec] parallel: expected a "
                    "boolean (true/false)."
                ) from exc
        if workers is None:
            try:
                workers = config.getint('exec', 'workers', fallback=None)
            except ValueError as exc:
                raise ValueError(
                    "Invalid config value for [exec] workers: expected a "
                    "positive integer."
                ) from exc
        if workers is not None and workers < 1:
            raise ValueError(
                f"workers must be a positive integer, got {workers}."
            )

        if not parallel:
            return False, 1

        if workers is None:
            workers = min(os.cpu_count() or 1, n_firings)
        # Never spawn more workers than there are firings to compute.
        workers = min(workers, n_firings)
        if workers <= 1:
            return False, 1
        return True, workers

    def _validate_plume_strike_inputs(self) -> bool:
        """Fail-fast validation of the inputs a plume-strike run requires.

        Logs an ERROR naming the case, the offending section/key, and any
        searched context, then raises a clear exception on the first missing
        required input rather than silently returning ``None``. Conditionally
        required inputs (TDF, surface temperature, sigma, thruster metrics) are
        enforced only when kinetics is enabled.

        Returns
        -------
        bool
            Whether plume kinetics is enabled for this run.
        """
        config = self.environment.config
        case_dir = self.environment.case_dir

        def require_option(section: str, key: str) -> str:
            if not config.has_option(section, key):
                logger.error("Missing required config [%s] %s (case %s)",
                             section, key, case_dir)
                raise KeyError(
                    f"Missing required config [{section}] {key} (case "
                    f"{case_dir})")
            return config[section][key]

        require_option('tv', 'stl')
        require_option('jfh', 'jfh')
        require_option('tcd', 'tcf')
        kinetics = require_option('pm', 'kinetics')
        require_option('plume', 'radius')
        require_option('plume', 'wedge_theta')

        target = getattr(self, 'target', None)
        if target is None or getattr(target, 'mesh', None) is None:
            logger.error("Target vehicle mesh not loaded (case %s)", case_dir)
            raise ValueError(
                f"Target vehicle mesh not loaded (case {case_dir}); call "
                "TargetVehicle.set_stl() before the plume-strike run.")
        if len(target.mesh.vectors) == 0:
            logger.error("Target mesh is empty (case %s)", case_dir)
            raise ValueError(f"Target mesh is empty (case {case_dir}).")

        jfh = getattr(self, 'jfh', None)
        if jfh is None or getattr(jfh, 'JFH', None) is None:
            logger.error("JFH not loaded (case %s)", case_dir)
            raise ValueError(
                f"JFH not loaded (case {case_dir}); call "
                "JetFiringHistory.read_jfh() with a valid [jfh] jfh.")
        if len(jfh.JFH) == 0:
            logger.error("JFH is empty (case %s)", case_dir)
            raise ValueError(f"JFH is empty (case {case_dir}).")

        thruster_data = getattr(self.vv, 'thruster_data', None)
        if not thruster_data:
            logger.error("Thruster configuration not loaded (case %s)", case_dir)
            raise ValueError(
                f"Thruster configuration not loaded (case {case_dir}); call "
                "set_thruster_config() before the plume-strike run.")

        # Every JFH thruster index must map to a configured thruster; the
        # numeric index is 1-based over the thruster configuration order.
        n_configured = len(thruster_data)
        for firing_idx, step in enumerate(jfh.JFH):
            for thr in step['thrusters']:
                if not (1 <= int(thr) <= n_configured):
                    logger.error(
                        "JFH firing %d references invalid thruster index %s "
                        "(configured thrusters: %d, case %s)",
                        firing_idx, thr, n_configured, case_dir)
                    raise ValueError(
                        f"JFH firing {firing_idx} references invalid thruster "
                        f"index {thr}; case {case_dir} configures "
                        f"{n_configured} thrusters.")

        kinetics_on = kinetics != 'None'
        if kinetics_on:
            require_option('tcd', 'tdf')
            require_option('tv', 'surface_temp')
            require_option('tv', 'sigma')
            if getattr(self.vv, 'thruster_metrics', None) is None:
                logger.error("Kinetics enabled but thruster metrics (TDF) not "
                             "loaded (case %s)", case_dir)
                raise ValueError(
                    f"Kinetics enabled but thruster metrics not loaded (case "
                    f"{case_dir}); call set_thruster_metrics().")
        return kinetics_on

    def jfh_plume_strikes(
        self, parallel: bool | None = None, workers: int | None = None
    ) -> dict[str, dict[str, NDArray[np.float64]]]:
        """
            Calculates number of plume strikes according to data provided for RPOD analysis.
            Method assumes that study assets are correctly configured.
            These assets include one JetFiringHistory, one TargetVehicle, and one VisitingVehicle.
            A Simple plume model is used. It does not calculate plume physics, only strikes. which
            are determined with a user defined "plume cone" geometry. Simple vector mathematics is
            used to determine if an VTK surface elements is struck by the "plume cone".

            Parameters
            ----------
            parallel : bool, optional
                Enable process-based parallelization across firings. Defaults
                to None, meaning "use the optional [exec] parallel config key",
                which itself defaults to false (serial, legacy behavior).
            workers : int, optional
                Number of worker processes when parallel execution is enabled.
                Defaults to None, meaning "use the optional [exec] workers
                config key", falling back to min(os.cpu_count(), n_firings).

            Notes
            -----
            Only independent per-firing strike arrays are computed in worker
            processes; cumulative arrays (cum_strikes, max_pressures,
            max_shears, cum_heat_flux_load) are accumulated serially in
            original firing order, and VTK output is written serially by the
            parent process. If the parallel path fails to initialize or run,
            the method logs a warning and falls back to serial execution.

            Returns
            -------
            dict
                firing_data keyed by firing number ('1'..'N'), each holding
                per-face arrays: strikes, cum_strikes, and, when kinetics is
                enabled, pressures, max_pressures, shear_stress, max_shears,
                heat_flux_rate, heat_flux_load, cum_heat_flux_load.
        """
        # Fail-fast validation of every required input before doing any work
        # or writing any output. Returns whether kinetics is enabled.
        kinetics_on = self._validate_plume_strike_inputs()

        # Prepare results directories and target data
        self.create_results_dir()
        target = self.target.mesh
        target_normals = target.get_unit_normals()
        # The target is stationary for the whole run, so face centroids are
        # computed once here and reused for every firing.
        target_centroids = compute_face_centroids(target.vectors)
        log_array_summary(logger, "target_unit_normals", target_normals)
        log_array_summary(logger, "target_face_centroids", target_centroids)

        # Initialize cumulative arrays
        if kinetics_on:
            cum_strikes, max_pressures, max_shears, cum_heat_flux_load = self.set_strike_fields(target)
        else:
            cum_strikes = self.set_strike_fields(target)

        firing_data: dict[str, dict[str, NDArray[np.float64]]] = {}

        n_firings = len(self.jfh.JFH)

        # Build serializable step dicts once; shared by serial and parallel paths.
        steps = []
        for firing in range(n_firings):
            steps.append({
                'thrusters': self.jfh.JFH[firing]['thrusters'],
                'xyz': np.array(self.jfh.JFH[firing]['xyz']),
                'dcm': np.array(self.jfh.JFH[firing]['dcm']),
                't': float(self.jfh.JFH[firing]['t'])
            })

        parallel_enabled, n_workers = self._resolve_parallel_options(parallel, workers, n_firings)

        # Run-scoped logging settings (progress cadence, performance toggle).
        settings = get_settings()
        progress_every = max(1, settings.progress_every_n_firings)
        plume_model = self.environment.config['pm']['kinetics']
        plume_radius = float(self.environment.config['plume']['radius'])
        wedge_theta = float(self.environment.config['plume']['wedge_theta'])
        exec_mode = 'parallel' if parallel_enabled else 'serial'

        logger.info(
            "Plume-strike run started: firings=%d target_faces=%d kinetics=%s "
            "plume_model=%s plume_radius=%s wedge_theta=%s mode=%s workers=%d "
            "progress_every_n_firings=%d",
            n_firings, len(target.vectors), kinetics_on, plume_model,
            plume_radius, wedge_theta, exec_mode,
            n_workers if parallel_enabled else 1, progress_every,
        )

        # Performance/memory tracking (standard library only). tracemalloc is
        # engaged solely for an explicitly configured run, never merely because
        # this method was called, so unit tests are not slowed.
        perf_on = performance_logging_enabled()
        tracemalloc_started_here = False
        if perf_on and not tracemalloc.is_tracing():
            tracemalloc.start()
            tracemalloc_started_here = True
        run_wall_start = time.perf_counter()
        run_cpu_start = time.process_time()
        fallback_occurred = False
        worker_meta = None
        overall_max_heat_flux_rate = 0.0

        # Optionally compute independent per-firing results in worker
        # processes. Workers receive only plain serializable inputs (arrays,
        # dicts, config scalars) — never the study/vehicle/environment objects.
        per_firing_results = None
        if parallel_enabled:
            try:
                per_firing_results, worker_meta = run_parallel_plume_strikes(
                    jfh_steps=steps,
                    face_centroids=target_centroids,
                    target_unit_normals=target_normals,
                    thruster_data=self.vv.thruster_data,
                    thruster_metrics=getattr(self.vv, 'thruster_metrics', None),
                    plume_params=extract_plume_params(self.environment),
                    workers=n_workers,
                    return_meta=True,
                )
                logger.info("Parallel execution completed across %d workers.",
                            n_workers)
            except Exception as exc:
                logger.warning(
                    "Parallel plume strike execution failed (%s: %s); "
                    "falling back to serial execution.",
                    type(exc).__name__, exc, exc_info=True,
                )
                per_firing_results = None
                worker_meta = None
                fallback_occurred = True

        # Loop through each firing in the JFH and delegate to impingement module
        for firing in range(n_firings):
            firing_wall_start = time.perf_counter()
            step = steps[firing]

            if per_firing_results is not None:
                result = per_firing_results[firing]
            else:
                result = compute_plume_strikes(
                    target_mesh=target,
                    target_unit_normals=target_normals,
                    vv=self.vv,
                    jfh_step=step,
                    environment=self.environment,
                    face_centroids=target_centroids,
                )

            strikes = result["strikes"]
            cum_strikes = cum_strikes + strikes

            cellData = {
                "strikes": strikes,
                "cum_strikes": cum_strikes.copy(),
            }

            if kinetics_on:
                # dict.get() reports these as Optional; the enclosing
                # kinetics_on branch is what guarantees the keys exist, an
                # invariant the result dict's type cannot carry.
                pressures: Any = result.get("pressures")
                shear_stresses: Any = result.get("shear_stress")
                heat_flux_rate: Any = result.get("heat_flux_rate")
                heat_flux_load: Any = result.get("heat_flux_load")

                max_pressures = np.maximum(max_pressures, pressures)
                max_shears = np.maximum(max_shears, shear_stresses)
                cum_heat_flux_load = cum_heat_flux_load + heat_flux_load

                cellData.update({
                    "pressures": pressures,
                    "max_pressures": max_pressures,
                    "shear_stress": shear_stresses,
                    "max_shears": max_shears,
                    "heat_flux_rate": heat_flux_rate,
                    "heat_flux_load": heat_flux_load,
                    "cum_heat_flux_load": cum_heat_flux_load,
                })

            firing_data[str(firing+1)] = cellData

            # if checking constraints:
            # save all new pressure and heat flux values into each cell's respective queue
            # add pressure and heat_flux into each cell's window sum
            # update the window queues based on their max sizes, and the firing times
            # update the parameter queues based on the updates made on the window queues 
            # if self.environment.config['pm']['kinetics'] != 'None' and checking_constraints:
            #     for i in range(len(pressures)):
            #         pressure_queues[i].put(float(pressures[i]))
            #         pressure_window_sums[i] += pressures[i]

            #         heat_flux_queues[i].put(float(heat_flux_load[i]))
            #         heat_flux_window_sums[i] += heat_flux_load[i]
            
            #     pressure_window_queue, pressure_cur_window, pressure_get_counter = self.update_window_queue(
            #         pressure_window_queue, pressure_cur_window, firing_time, pressure_window_size)

            #     heat_flux_window_queue, heat_flux_cur_window, heat_flux_get_counter = self.update_window_queue(
            #         heat_flux_window_queue, heat_flux_cur_window, firing_time, heat_flux_window_size)

            #     for queue_index in range(len(pressure_queues)):
            #         if not pressure_queues[queue_index].empty():
            #             pressure_queues[queue_index], pressure_window_sums[queue_index] = self.update_parameter_queue(pressure_queues[queue_index], pressure_window_sums[queue_index], pressure_get_counter)

            #         if not heat_flux_queues[queue_index].empty() and heat_flux_window_sums[queue_index] != 0:
            #             heat_flux_queues[queue_index], heat_flux_window_sums[queue_index] = self.update_parameter_queue(heat_flux_queues[queue_index], heat_flux_window_sums[queue_index], heat_flux_get_counter)
                
            #         #check if instantaneous constraints are broken
            #         if pressures[queue_index] > pressure_constraint and not failed_constraints:
            #             constraint_file.write(f"Pressure constraint failed at cell #{i}.\n")
            #             constraint_file.write(f"Pressure reached {pressures[queue_index]}.\n\n")
            #             failed_constraints = 1
            #         if shear_stresses[queue_index] > shear_constraint and not failed_constraints:
            #             constraint_file.write(f"Shear constraint failed at cell #{i}.\n")
            #             constraint_file.write(f"Shear reached {shear_stresses[queue_index]}.\n\n")
            #             failed_constraints = 1
            #         if heat_flux[queue_index] > heat_flux_constraint and not failed_constraints:
            #             constraint_file.write(f"Heat flux constraint failed at cell #{i}.\n")
            #             constraint_file.write(f"Heat flux reached {heat_flux[queue_index]}.\n\n")
            #             failed_constraints = 1

            #         # check if window constraints are broken, and report
            #         if pressure_window_sums[queue_index] > pressure_window_constraint and not failed_constraints:
            #             constraint_file.write(f"Pressure window constraint failed at cell #{queue_index}.\n")
            #             constraint_file.write(f"Pressure winodw reached {pressure_window_sums[queue_index]}.\n\n")
            #             failed_constraints = 1
            #         if heat_flux_window_sums[queue_index] > heat_flux_window_constraint and not failed_constraints:
            #             constraint_file.write(f"Heat flux window constraint failed at cell #{queue_index}.\n")
            #             constraint_file.write(f"Heat flux load reached {heat_flux_window_sums[queue_index]}.\n\n")
            #             failed_constraints = 1


            # convert_stl_to_vtk_strikes() prepends case_dir + "results/", so
            # pass a path relative to that. Previously the full case_dir path
            # was passed here and prepended again, producing a doubled
            # case_dir/results/case_dir/results/strikes/ output tree.
            path_to_vtk = "strikes/firing-" + str(firing)

            self.target.convert_stl_to_vtk_strikes(path_to_vtk, cellData.copy(), target)
            logger.debug("Wrote firing artifact: %s", os.path.join(
                self.environment.case_dir, "results", path_to_vtk + ".vtu"))

            # Detailed per-firing array diagnostics (DEBUG only).
            log_array_summary(logger, f"firing_{firing + 1}_strikes", strikes)

            # Track overall max heat-flux RATE for the completion summary. This
            # is a scalar used only for logging; it is never written to output
            # nor added to firing_data, so numerical results are unchanged.
            if kinetics_on:
                overall_max_heat_flux_rate = max(
                    overall_max_heat_flux_rate, float(heat_flux_rate.max()))

            # Configured progress reporting: every Nth firing and the final one.
            if logger.isEnabledFor(logging.INFO) and (
                    (firing + 1) % progress_every == 0 or firing == n_firings - 1):
                firing_wall_s = time.perf_counter() - firing_wall_start
                struck_faces = int(np.count_nonzero(strikes))
                pct = 100.0 * (firing + 1) / n_firings
                fields = [
                    f"progress_pct={pct:.1f}",
                    f"jfh_id={self.jfh.JFH[firing]['nt']}",
                    f"sim_time_s={step['t']}",
                    f"thrusters={list(step['thrusters'])}",
                    f"struck_faces={struck_faces}",
                ]
                if kinetics_on:
                    fields += [
                        f"max_pressure_pa={float(result['pressures'].max()):.6g}",
                        f"max_shear_pa={float(result['shear_stress'].max()):.6g}",
                        f"max_heat_flux_w_m2={float(result['heat_flux_rate'].max()):.6g}",
                        f"max_heat_flux_load_j_m2={float(result['heat_flux_load'].max()):.6g}",
                    ]
                if worker_meta is not None and worker_meta[firing] is not None:
                    wm = worker_meta[firing]
                    fields += [f"worker_pid={wm['pid']}",
                               f"wall_time_s={wm['wall_time_s']:.4f}"]
                else:
                    fields += [f"worker_pid={os.getpid()}",
                               f"wall_time_s={firing_wall_s:.4f}"]
                tracked_bytes = cum_strikes.nbytes
                if kinetics_on:
                    tracked_bytes += (max_pressures.nbytes + max_shears.nbytes
                                      + cum_heat_flux_load.nbytes)
                fields.append(
                    f"tracked_array_memory_mb={tracked_bytes / 1e6:.4f}")
                logger.info("Firing %d/%d completed: %s",
                            firing + 1, n_firings, " ".join(fields))

        # if self.environment.config['pm']['kinetics'] != 'None' and checking_constraints:
        #     if not failed_constraints:
        #         constraint_file.write(f"All impingement constraints met.")
        #     # constraint_file.close()

        # Completion summary (INFO).
        run_wall_s = time.perf_counter() - run_wall_start
        run_cpu_s = time.process_time() - run_cpu_start
        peak_memory_mb = None
        if perf_on and tracemalloc.is_tracing():
            _current, peak_bytes = tracemalloc.get_traced_memory()
            peak_memory_mb = peak_bytes / 1e6
            if tracemalloc_started_here:
                tracemalloc.stop()

        tracked_bytes = cum_strikes.nbytes
        if kinetics_on:
            tracked_bytes += (max_pressures.nbytes + max_shears.nbytes
                              + cum_heat_flux_load.nbytes)
        run_status = 'successful_with_warning' if fallback_occurred else 'successful'
        summary = {
            'firings_completed': n_firings,
            'total_struck_face_events': float(cum_strikes.sum()),
            'unique_struck_faces': int(np.count_nonzero(cum_strikes)),
            'wall_time_s': round(run_wall_s, 3),
            'cpu_time_s': round(run_cpu_s, 3),
            'avg_wall_time_s_per_firing': (
                round(run_wall_s / n_firings, 4) if n_firings else 0.0),
            'tracked_array_memory_mb': round(tracked_bytes / 1e6, 4),
            'output_artifacts': n_firings,
            'execution_mode': (
                'parallel' if (parallel_enabled and not fallback_occurred)
                else 'serial'),
            'fallback_occurred': fallback_occurred,
            'run_status': run_status,
        }
        if kinetics_on:
            summary['max_pressure_pa'] = f"{float(max_pressures.max()):.6g}"
            summary['max_shear_pa'] = f"{float(max_shears.max()):.6g}"
            summary['max_heat_flux_w_m2'] = f"{overall_max_heat_flux_rate:.6g}"
            summary['max_heat_flux_load_j_m2'] = (
                f"{float(cum_heat_flux_load.max()):.6g}")
        if peak_memory_mb is not None:
            summary['python_peak_memory_mb'] = round(peak_memory_mb, 3)
        logger.info("Plume-strike run completed: %s",
                    " ".join(f"{k}={v}" for k, v in summary.items()))

        return firing_data

    def calc_time_multiplier(self, v_ida: float, v_o: float,
                             r_o: float) -> float:
        # Determine thruster configuration characterstics.
        # The JFH only contains firings done by the neg_x group
        m_dot_sum = self.calc_m_dot_sum('neg_x')
        # print('m_dot_sum is', m_dot_sum)
        # thruster_metrics is Optional (None when the case has no [tcd] tdf);
        # this legacy path has never guarded it.
        MIB = self.vv.thruster_metrics[  # type: ignore[index]
            self.vv.thruster_data[self.vv.rcs_groups['neg_x'][0]]['type'][0]]['MIB']
        # print('MIB is', MIB)
        F_thruster = self.vv.thruster_metrics[  # type: ignore[index]
            self.vv.thruster_data[self.vv.rcs_groups['neg_x'][0]]['type'][0]]['F']
        F = F_thruster * np.cos(self.vv.decel_cant)
        n_thrusters = len(self.vv.rcs_groups['neg_x'])
        F = F * n_thrusters

        # Defining a multiplier reduce time steps and make running faster
        dt = (MIB / F_thruster)
        # print('dt is', dt)
        dm_firing = m_dot_sum * dt
        # print('dm_firing is', dm_firing)
        docking_mass = self.vv.mass
        # print('docking mass is', docking_mass)

        # Calculate required change in velocity.
        dv_req = v_o - v_ida
        v_e = self.calc_v_e('neg_x')

        # Calculate propellant used for docking and changes in mass.
        forward_propagation = False
        delta_mass_jfh = self.calc_delta_mass_v_e(dv_req, v_e, forward_propagation)
        # print('delta_mass_docking is',delta_mass_jfh)

        pre_approach_mass = self.vv.mass
        # print('pre-approach mass is', pre_approach_mass)

        # Instantiate data structure to hold JFH data + physics data.
        # Initializing position
        x = [r_o]
        y = [0]
        z = [0]

        # Initializing empty tracking lists
        dx = [0]
        t = [0]
        dv: list[float] = [0]

        # Initializing inertial state
        dxdt = [v_o]

        # Initializing initial mass
        mass = [pre_approach_mass]

        # Initializing list to later sum propellant expenditure
        dm_total = [dm_firing]

        # Firing number
        n = [1]

        # Create dummy rotation matrices.
        x1 = [1, 0, 0]
        y1 = [1, 0, 0]

        rot = [np.array(rotation_matrix_from_vectors(x1, y1))]

        # Calculate JFH and 1D physics data for required firings.
        while (dv_req > 0):
            # print('dv_req', round(dv_req, 4), 'n firings', n[i])

            # Grab last value in the JFH arrays (initial conditions for current time step)
            # print('x, dx, dt, t, dxdt, mass, dm_total')
            # print(x[-1], dx[-1], dt_vals[-1], t[-1], dxdt[-1], mass[-1], dm_total[-1])

            # Update VV mass per firing
            mass_o = mass[-1]
            mass.append(mass_o - dm_firing)
            mass_f = mass[-1]
            # print('mass_f is', mass_f)

            # Calculate velocity change per firing.
            dv_firing = self.calc_delta_v(dt, v_e, m_dot_sum, mass_o)
            dv.append(dv_firing)
            # print('dv is', dv_firing)
            # print(round(dxdt[-1] - dv_firing, 2))
            # input()
            dxdt.append(dxdt[-1] - dv_firing)

            # Calculate distance traveled per firing
            # print(dxdt[-1], dxdt[-2]) # last and second to last element.
            v_avg = 0.5 * (dxdt[-1] + dxdt[-2])
            dx.append(v_avg * dt)
            x.append(x[-1] - v_avg*dt)
            y.append(0)
            z.append(0)

            # Calculate left over v_req (TERMINATES LOOP)
            dv_req -= dv_firing

            # Calculate mass expended up to this point.
            dm_total.append(dm_total[-1] + dm_firing)

            # Calculate current firing.
            n.append(n[-1]+1)

            # Add time data.
            t.append(t[-1] + dt)

            rot.append(np.array(rotation_matrix_from_vectors(x1, y1)))

        # After simulating, estimate a coarse time multiplier based on number of firings
        n_firings = len(t)
        time_multiplier = 10 / n_firings if n_firings > 0 else 1
        # print('n firings:', len(t), 'time multiplier:', time_multiplier)
        return (1 / time_multiplier)

        # one_d_results = {
        #     'n_firings': n,
        #     'x': x,
        #     'dx': dx,
        #     't': t,
        #     'dv': dv,
        #     'v': dxdt,
        #     'mass': mass,
        #     'delta_mass': dm_total
        # }

        # print(one_d_results)

    def print_jfh_1d_approach_n_fire(self, v_ida: float, v_o: float,
                                     r_o: float, n_firings: int,
                                     trade_study: bool = False) -> None:
        """Write a 1D-approach JFH holding EXACTLY n_firings entries.

        ``n_firings`` is the number of entries written to the Jet Firing
        History. It is validated up front and enforced by
        ``compute_1d_approach``; previously it was accepted and then ignored,
        so the file's length was whatever the deceleration simulation
        happened to produce.
        """
        n_firings = validate_n_firings(n_firings)
        # Delegate to approach_maneuvers.compute_1d_approach and rpod.io.write_jfh
        tm = self.calc_time_multiplier(v_ida, v_o, r_o)
        inputs = ApproachInputs(v_ida=float(v_ida), v_o=float(v_o), r_o=float(r_o), group='neg_x')
        # Adapter for grouping methods if not explicitly available as a module
        class _GroupingAdapter:
            def __init__(self, outer: 'PlumeStrikeEstimationStudy') -> None:
                self._outer = outer
            def calc_m_dot_sum(self, group: str) -> float:
                return self._outer.calc_m_dot_sum(group)
            def calc_v_e(self, group: str) -> float:
                return self._outer.calc_v_e(group)

        results = compute_1d_approach(
            inputs=inputs,
            vv=self.vv,
            fuel_mgr=self,
            grouping=_GroupingAdapter(self),
            cant_rad=self.vv.decel_cant,
            dt_strategy={"multiplier": tm},
            n_firings=n_firings,
        )

        r = [results["x"], results["y"], results["z"]]
        t_values = results["t"]
        rot = results["rot"]

        # Build output path as before
        if not trade_study:
            jfh_path = self.environment.case_dir + 'jfh/' + self.environment.config['jfh']['jfh']
        else:
            jfh_path = self.environment.case_dir + 'jfh/' + self.get_case_key() + '.A'

        logger.info("JFH generation started (1D n-fire approach): "
                    "v_ida_m_s=%s v_o_m_s=%s r_o_m=%s n_firings=%s",
                    v_ida, v_o, r_o, n_firings)
        gen_start = time.perf_counter()
        os.makedirs(os.path.dirname(jfh_path), exist_ok=True)
        write_jfh(t_values, r, rot, jfh_path, mode="1d")
        if len(t_values) != n_firings:
            raise ValueError(
                f"JFH {jfh_path} was written with {len(t_values)} entries but "
                f"{n_firings} firings were requested")
        _log_jfh_generation_complete(jfh_path, len(t_values), gen_start)


    def print_jfh_1d_approach(self, v_ida: float, v_o: float, r_o: float,
                              trade_study: bool = False) -> None:
        """
            Method creates JFH data for axial approach using simpified physics calculations.

            This approach models one continuous firing.

            Kinematics and mass changes are discretized according to the thruster's minimum firing time.

            Parameters
            ----------
            v_ida : float
                VisitingVehicle docking velocity (determined by international docking adapter)

            v_o : float
                VisitingVehicle incoming axial velocity.

            r_o : float
                Initial distance to docking port.

            Returns
            -------
            Method doesn't currently return anything. Simply prints data to a files as needed.
            Does the method need to return a status message? or pass similar data?

        """
        # Delegate to approach_maneuvers with a fixed multiplier similar to legacy
        inputs = ApproachInputs(v_ida=float(v_ida), v_o=float(v_o), r_o=float(r_o), group='neg_x')
        class _GroupingAdapter:
            def __init__(self, outer: 'PlumeStrikeEstimationStudy') -> None:
                self._outer = outer
            def calc_m_dot_sum(self, group: str) -> float:
                return self._outer.calc_m_dot_sum(group)
            def calc_v_e(self, group: str) -> float:
                return self._outer.calc_v_e(group)

        results = compute_1d_approach(
            inputs=inputs,
            vv=self.vv,
            fuel_mgr=self,
            grouping=_GroupingAdapter(self),
            cant_rad=self.vv.decel_cant,
            dt_strategy={"multiplier": 60.0},
        )

        r = [results["x"], results["y"], results["z"]]
        t_values = results["t"]
        rot = results["rot"]

        if not trade_study:
            jfh_path = self.environment.case_dir + 'jfh/' + self.environment.config['jfh']['jfh']
        else:
            jfh_path = self.environment.case_dir + 'jfh/' + self.get_case_key() + '.A'

        logger.info("JFH generation started (1D approach): "
                    "v_ida_m_s=%s v_o_m_s=%s r_o_m=%s", v_ida, v_o, r_o)
        gen_start = time.perf_counter()
        os.makedirs(os.path.dirname(jfh_path), exist_ok=True)
        write_jfh(t_values, r, rot, jfh_path, mode="1d")
        _log_jfh_generation_complete(jfh_path, len(t_values), gen_start)
        return

    def edit_1d_JFH(self, t_values: NDArray[np.float64],
                    r: Sequence[NDArray[np.float64]],
                    rot: NDArray[np.float64]) -> None:
        """
            Helper function to RPOD.calc_jfh_1d_approach() that is responsible for
            modifying the JFH attribute in memory with the values calculated.

            Parameters
            ----------
            t_values : np.array
                Array containing time step data for each firing in the JFH.

            r : np.array
                Array containing positional data for each firing in the JFH.

            rot : np.array
                Array containing rotational data for each firing in the JFH.

            Returns
            -------
            Method doesn't currently return anything. Simply prints data to a files as needed.
            Does the method need to return a status message? or pass similar data?

        """
        # Printing out the empty JFH and checking its size
        # print('self.jfh.JFH is', self.jfh.JFH)
        # print('len(self.jfh.JFH) is', len(self.jfh.JFH))

        # Scrap
        # print('self.jfh.JFH[0] is', self.jfh.JFH[0])
        # print('self.jfh.JFH[1] is', self.jfh.JFH[1])
        # self.jfh.JFH.append({'nt': '1', 'dt': '0.000', 't': '1', 'dcm': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 'xyz': [-10.0, 0.0, 0.0], 'uf': 1.0, 'thrusters': [1, 2, 5, 6, 9, 10, 13, 14]})
        # print('self.jfh.JFH[0] is', self.jfh.JFH[0])

        # Checking the time step
        # print('t_values is', t_values)
        # print('len(t_values) is', len(t_values))

        # Changing the number of firings to be evaluated
        self.jfh.nt = len(t_values)
        # print('self.jfh.nt is', self.jfh.nt)
        # print('self.jfh.nt after is', self.jfh.nt)
        # print("self.jfh.JFH[0]['nt'] is", self.jfh.JFH[0]['nt'])

        # Confirming the time step is correct
        # print('str(t_values[1]) is', str(t_values[1]))

        # Verifying the format of the rot array
        # print('rot[0] is', rot[0])
        # print('[list(rot[0][0]), list(rot[0][1]), list(rot[0][2])] is', [list(rot[0][0]), list(rot[0][1]), list(rot[0][2])])
        # print('[list(rot[1][0]), list(rot[1][1]), list(rot[1][2])] is', [list(rot[1][0]), list(rot[1][1]), list(rot[1][2])])

        # Verifying the format of the r array
        # print('r is', r)
        # print('[r[0][0], r[1][0], r[2][0]] is', [r[0][0], r[1][0], r[2][0]])
        # print('[r[0][1], r[1][1], r[2][1]] is', [r[0][1], r[1][1], r[2][1]])

        for i in range(len(t_values)):
            # NOTE: 'thrusters' is currently hardcoded
            # NOTE: all the 'xyz' values are still negative
            # Start from the required distance to slow down and approach zero
            # print('-r[0][-1] + r[0][i] is', -r[0][-1] + r[0][i])

            
            # NOTE: The thrusters are currently hardcoded, but should be changed to the thrusters indicated in the neg_x group in the tgf
            self.jfh.JFH.append({'nt': str(i + 1), 'dt': str(t_values[i]), 't': str(t_values[1]), 'dcm': [list(rot[i][0]), list(rot[i][1]), list(rot[i][2])], 'xyz': [-r[0][-1] + r[0][i] - 0.5, -r[1][i], -r[2][i]], 'uf': 1.0, 'thrusters': [1, 2, 5, 6, 9, 10, 13, 14]})

        # Printing out the populated JFH and checking its size
        # print('self.jfh.JFH is', self.jfh.JFH)
        # print('len(self.jfh.JFH) is', len(self.jfh.JFH))

    def calc_jfh_1d_approach(self, v_ida: float, v_o: float,
                             cant: float) -> None:
        """
            The x-position represents the distance required to reach a velocity of zero.    
        
            Method calculates JFH data for 1D approach using simpified physics calculations.

            This approach models one continuous firing.

            Kinematics and mass changes are discretized according to the thruster's minimum firing time.

            Parameters
            ----------
            v_ida : float
                VisitingVehicle docking velocity (determined by international docking adapter)

            v_o : float
                VisitingVehicle incoming axial velocity.

            cant : float
                Angling of all deceleration thrusters in degrees that 

            Returns
            -------
            Method doesn't currently return anything. Simply prints data to a files as needed.
            Does the method need to return a status message? or pass similar data?

        """
        # Delegate to compute-first API, then update in-memory JFH via existing helper
        MissionPlanner.cant = np.radians(cant)
        inputs = ApproachInputs(v_ida=float(v_ida), v_o=float(v_o), r_o=0.0, group='neg_x')

        class _GroupingAdapter:
            def __init__(self, outer: 'PlumeStrikeEstimationStudy') -> None:
                self._outer = outer
            def calc_m_dot_sum(self, group: str) -> float:
                return self._outer.calc_m_dot_sum(group)
            def calc_v_e(self, group: str) -> float:
                return self._outer.calc_v_e(group)

        results = compute_1d_approach(
            inputs=inputs,
            vv=self.vv,
            fuel_mgr=self,
            grouping=_GroupingAdapter(self),
            cant_rad=MissionPlanner.cant,
            dt_strategy={"multiplier": 100.0},
        )

        t_values = results["t"]
        r = [results["x"], results["y"], results["z"]]
        rot = results["rot"]
        self.edit_1d_JFH(t_values, r, rot)

    def get_case_key(self) -> str:
        return self.case_key

    def set_case_key(self, v0_iter: Any, cant_iter: Any) -> None:

        self.case_key = 'vo_' + str(v0_iter) + '_cant_' + str(cant_iter)

        return

    # Declared without self and never called; invoking it on an instance
    # would raise TypeError. Adding self would change the signature,
    # which #103 excludes, so the defect is flagged here instead.
    def make_test_jfh() -> None:  # type: ignore[misc]

        return
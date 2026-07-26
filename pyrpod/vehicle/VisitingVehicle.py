import pandas as pd

from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import numpy as np
import math
import os
import re
import logging

from pyrpod.vehicle.Vehicle import Vehicle
from pyrpod.mdao import SweepConfig
from pyrpod.logging_utils import log_asset, log_array_summary
from pyrpod.util.io.fs import resolve_asset_path

logger = logging.getLogger(__name__)

# Adapted from
# https://stackoverflow.com/questions/54616049/converting-a-rotation-matrix-to-euler-angles-and-back-special-case
def rot2eul(R):
    beta = -np.arcsin(R[2][0])
    alpha = np.arctan2(R[2][1]/np.cos(beta),R[2][2]/np.cos(beta))
    gamma = np.arctan2(R[1][0]/np.cos(beta),R[0][0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

# Helper functions for constructer. 
def process_coordinates(str_coord):
    # Split str at spaces
    str_list = str_coord.split(' ')
    # Return as list of floats
    return [float(x) for x in str_list]

# Process definition of an individual thruster.
def process_thruster_def(str_thruster):
    columns = ['name', 'type', 'exit', 'dcm']
    # thruster = pd.DataFrame(columns = columns)
    # print(thruster.dtypes)
    thruster = {}
    # Remove new line char (last char) and split at any space char.    
    str_list = str_thruster[:-1].split(' ')
    # str_list = str_thruster.split(' ')
    # print(str_list)
    # Save name and type of thruster
    thruster["name"] = [str_list.pop(0)]
    thruster['type'] = [str_list.pop(0)]
    # print(thruster['name'])
    # Save coordinate for center of exit plane.
    coord = []
    for i in range(3):
        coord.append(float(str_list.pop(0)))
    thruster['exit'] = [coord]

    # Save direction cosine matrix of thruster relative to the vehicle    
    drm = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(float(str_list.pop(0)))
        drm.append(row)
    thruster['dcm'] = drm
    # thruster = pd.DataFrame(thruster)
    # print(thruster)
    # return pd.DataFrame(thruster)
    return thruster

# Wrapper function
def process_str_thrusters(str_thrusters):
    # dcm = direction cosine matrix
    columns = ['name', 'type', 'exit', 'dcm']
    thrusters_data = {}
    for thruster in str_thrusters:
        name = str(thruster.split(' ')[0])
        thrusters_data[name] = process_thruster_def(thruster)
        # print(process_thruster_def(thruster))
        # thrusters_data = pd.concat([thrusters_data,process_thruster_def(thruster)], ignore_index = True)
        # print(thrusters_data.dtypes)

    return thrusters_data

# Process definition of an individual cluster.
def process_cluster_def(str_cluster):
    columns = ['name', 'exit', 'dcm']
    cluster = {}
    # Remove new line char (last char) and split at any space char.    
    str_list = str_cluster[:-1].split(' ')
    # Save name of cluster
    cluster["name"] = [str_list.pop(0)]
    # Save coordinate for center of cluster.
    coord = []
    for i in range(3):
        coord.append(float(str_list.pop(0)))
    cluster['exit'] = [coord]

    # Save direction cosine matrix of cluster relative to the vehicle    
    drm = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(float(str_list.pop(0)))
        drm.append(row)
    cluster['dcm'] = drm
    return cluster

# Wrapper function
def process_str_clusters(str_clusters):
    # dcm = direction cosine matrix
    columns = ['name', 'exit', 'dcm']
    clusters_data = {}
    for cluster in str_clusters:
        name = str(cluster.split(' ')[0])
        clusters_data[name] = process_cluster_def(cluster)

    return clusters_data

class VisitingVehicle(Vehicle):
    """
        Class responsible for handling visiting vehicle data.

        Includes surface mesh and thruster configuration data.

        Attributes
        ----------
        num_thrusters : int
            Total number of thrusters in RCS configuration.

        thruster_units : str
            Units for thruster coordinates.

        cog : float
            Center of Gravity for the Visiting Vehicle.

        grapple : float
            Grappling coordinate for the Visiting Vehicle.

        thruster_data : dictionary
            Dictionary holding the main thruster configuration data.

        cluster_data : dictionary
            Dictionary holding the main cluster configuration data.

        jet_interactions : float
            Can be ignored for now.

        Methods
        -------
        set_stl()
            Reads in Vehicle surface mesh from STL file.
        
        set_thruster_config()
            Reads the thruster configuration file from the config.ini for the Visiting Vehicle and saves it as class members.

        change_cluster_config()
            Alters cluster configuration data using OpenMDAO inputs.


        set_cluster_config()
            Read in cluster configuration data from the provided file path.

        set_thruster_metrics()
            Reads the thruster data file to gather thruster-specific performance parameters for the configuration from a .csv file
            and saves it in a list of dictionaries. These dictionaries are then saved into each thruster in the configuration.    

        print_info()
            Simple method to format printing of vehicle info.

        initiate_plume_mesh()
            Helper method that reads in surface mesh for plume clone.

        transform_plume_mesh(thruster_id, plumeMesh, vv_orientation=None, vv_position=None)
            Canonical plume placement. Local mode (no pose) applies the thruster
            DCM and exit; complete mode (JFH pose supplied) additionally applies
            the vehicle orientation, vehicle position, and cluster exit offset.

        get_thruster_id_map() / get_thruster_id(jfh_index)
            Cached JFH-index -> canonical-thruster-id mapping (insertion order);
            invalidated when set_thruster_config() replaces the configuration.

        initiate_plume_normal(thruster_id)
            Collects plume normal vectors data for visualization.

        plot_vv_and_thruster(plumeMesh, thruster_id, normal, i)
            Plots Visiting Vehicle and plume cone for provided thruster id.

        check_thruster_configuration()
            Plots visiting vehicle and all thrusters in RCS configuration.
    """

    def print_info(self):
        """
            Simple method to format printing of vehicle info.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        logger.info('number of thrusters: %s', self.num_thrusters)
        logger.info('thruster units: %s', self.thruster_units)
        logger.info('center of gravity: %s', self.cog)
        logger.info('grapple coordinate: %s', self.grapple)
        logger.info('number of dual jet interactions: %s', self.jet_interactions)
        return
    
    def set_stl(self):
        """
            Reads in Vehicle surface mesh from STL file.

            Parameters
            ----------
            None

            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
            Does the method need to return a status message? or pass similar data?
        """
        path_to_stl = resolve_asset_path(self.case_dir, 'stl', self.config['vv']['stl_lm'])
        self.mesh = mesh.Mesh.from_file(path_to_stl)
        self.path_to_stl = path_to_stl
        log_asset("visiting-vehicle STL", self.config['vv']['stl_lm'],
                  path_to_stl, self.case_dir, logger=logger)
        logger.info("Visiting vehicle geometry loaded: mesh_faces=%d",
                    len(self.mesh.vectors))
        log_array_summary(logger, "vv_mesh_vectors", self.mesh.vectors)
        return

    def get_thruster_cant(self, thruster_name):
        """
            Finds the cant angle defined as angle from the LM surface tangent.
            Takes the thruster's DCM, undoes the frame transformation
            ie. the frame made by the surface tangent and the line from the LM's 
            axial surface to the exit coordinate, is rotated about x to match the universal YZ axes.
            Then the DCM is decomposed to grab the cant angling.

            Parameters
            ----------
            thruster_name : string
                name of the thruster of interest
            
            Returns
            -------
            float
                cant angle in rad
        """
        # find frame rotation (Tx)
        # taken directly from SweepConfig.SweepDecelAngles.calculate_frame_rot()
        exit_coords = self.thruster_data[thruster_name]['exit'][0]
        y = exit_coords[1]
        z = exit_coords[2]

        if y == 0 and z > 0:
            theta = np.pi/2
        elif y == 0 and z < 0:
            theta = -np.pi/2
        else:
            theta = np.arctan2(z, y)

        Tx = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

        # Find the inverse of Tx
        inv_Tx = np.linalg.inv(Tx)

        # undo the frame rotation
        DCM = self.thruster_data[thruster_name]['dcm']
        Rz = np.dot(inv_Tx, DCM)

        # resulting matrix representes the rotation of DCM about z-axis
        # Rz -> cant angle
        cant = np.arccos(Rz[0][0])

        return cant

    def set_thruster_config(self, thruster_data=None):
        """
            Reads the thruster configuration file from the config.ini for the Visiting Vehicle and saves it as class members.

            If thruster data IS passed, simple overwrite self.thruster_data.
            This use is intended to occur only after a notional use of this method.
            (ie a method call without thruster_data, using the tcf file path instead.)

            Parameters
            ----------
            None

            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
            Does the method need to return a status message? or pass similar data?
        """
        if thruster_data is None:
            try:
                tcf_name = self.config['tcd']['tcf']
            except KeyError:
                logger.warning("No [tcd] tcf configured for case %s; thruster "
                               "configuration not loaded.", self.case_dir)
                return
            path_to_tcf = resolve_asset_path(self.case_dir, 'tcd', tcf_name)
            # Simple program, reading text from a file.
            with open(path_to_tcf, 'r') as f:
                lines = f.readlines()

                # Parse through first few lines, save relevant information. 
                self.num_thrusters = int(lines.pop(0))
                self.thruster_units = lines.pop(0)[0] # dont want '\n'
                self.cog = process_coordinates(lines.pop(0))
                self.grapple = process_coordinates(lines.pop(0))

                # Save all strings containing thruster data in a list
                str_thrusters = []
                for i in range(self.num_thrusters):
                    str_thrusters.append(lines.pop(0))

                # Parse through strings and save data in a dictionary
                self.thruster_data = process_str_thrusters(str_thrusters)

                self.jet_interactions = lines.pop(0)

            log_asset("thruster config (TCF)", tcf_name, path_to_tcf,
                      self.case_dir, logger=logger)

        else:
            logger.debug("Thruster configuration overwritten in-memory "
                         "(%d thrusters).", len(thruster_data))
            self.thruster_data = thruster_data

        # The thruster configuration was (re)loaded or replaced; drop any cached
        # JFH-index -> thruster-id mapping so it is rebuilt on next use.
        self._thruster_id_map = None

        self.use_clusters = False

        n_types = len({self.thruster_data[t]['type'][0]
                       for t in self.thruster_data})
        logger.info("Thruster configuration loaded: thrusters=%d "
                    "thruster_types=%d", len(self.thruster_data), n_types)
        if logger.isEnabledFor(logging.DEBUG):
            exits = np.array([self.thruster_data[t]['exit'][0]
                              for t in self.thruster_data])
            log_array_summary(logger, "thruster_exit_coords", exits)

        return
    
    def change_cluster_config(self, x):
        """
            Alters cluster configuration data using OpenMDAO inputs.

            Parameters
            ----------
            x : array
                Axial position (along the x axis) of the nozzle exit with respect to the LM's docking adapter.
            
            Returns
            -------
            Method doesn't currently return anything.
        """
        # print('len(self.cluster_data) is', len(self.cluster_data))
        # print('float(x) is', float(x))
        for cluster in self.cluster_data:
            # print('cluster is', cluster)
            self.cluster_data[cluster]["exit"][0][0] = float(x)

    def set_cluster_config(self):
        """
            Read in cluster configuration data from the provided file path.
            Gathers cluster configuration data for the Visiting Vehicle from a .dat file
            and saves it as class members.
            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
        """

        path_to_ccf = resolve_asset_path(self.case_dir, 'tcd', self.config['tcd']['ccf'])

        # Simple program, reading text from a file.
        with open(path_to_ccf, 'r') as f:
            lines = f.readlines()

            # Parse through first few lines, save relevant information. 
            self.num_clusters = int(lines.pop(0))
            self.cluster_units = lines.pop(0)[0] # dont want '\n'

            # Save all strings containing cluster data in a list
            str_clusters = []
            for i in range(self.num_clusters):
                str_clusters.append(lines.pop(0))

            # Parse through strings and save data in a dictionary
            self.cluster_data = process_str_clusters(str_clusters)

        self.use_clusters = True

        log_asset("cluster config (CCF)", self.config['tcd']['ccf'],
                  path_to_ccf, self.case_dir, logger=logger)
        logger.info("Cluster configuration loaded: clusters=%d units=%s",
                    self.num_clusters, self.cluster_units)

        return

    def set_thruster_metrics(self):
        """
            Reads the csv thruster data file to gather thruster-specific performance parameters for the configuration
            and saves it in a list of dictionaries. These dictionaries are then saved into each thruster in the configuration.

            Parameters
            ----------
            None

            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
            Does the method need to return a status message? or pass similar data?
        """

        # Read in path for thruster metric data.
        try:
            tdf_name = self.config['tcd']['tdf']
        except KeyError:
            logger.warning("No [tcd] tdf configured for case %s; thruster "
                           "performance metrics not loaded.", self.case_dir)
            self.thruster_metrics = None
            return
        path_to_thruster_metrics = resolve_asset_path(self.case_dir, 'tcd', tdf_name)

        # specify columns to be read as strings.
        str_cols = ['#']
        dict_types = {x: 'str' for x in str_cols}

        # read csv into a pd dataframe
        thruster_metrics = pd.read_csv(path_to_thruster_metrics, dtype=dict_types)
        # print(thruster_characteristics)

        # convert the dataframe into a list of dictionaries
        thruster_metrics_list = thruster_metrics.to_dict(orient='records')

        self.thruster_metrics = {}

        for thruster in thruster_metrics_list:

            # Seperate thruster metrics to form new key value pairs.
            thruster_id = thruster['#']
            thruster_metrics = thruster.pop('#')

            # Save thruster metrics
            self.thruster_metrics[thruster_id] = thruster

        log_asset("thruster metrics (TDF)", tdf_name, path_to_thruster_metrics,
                  self.case_dir, logger=logger)
        logger.info("Thruster metrics loaded: thruster_types=%d "
                    "kinetics_performance_data=available",
                    len(self.thruster_metrics))

        return

    def initiate_plume_mesh(self):
        """
            Helper method that reads in surface mesh for plume clone.

            Parameters
            ----------
            None for now. Should/could include cone sizing parameters according to plume physics.
            This is easy. Simply produce a "unit cone" ahead of time, and scale the coordinates
            using numpy-stl. Cone half-angle can also be pre-programmed.

            Returns
            -------
            plumeMesh : mesh.Mesh
                Surface mesh constructed from STL file.
        """
        # TODO: use STL that is already oriented correctly.
        plumeMesh = mesh.Mesh.from_file('../data/stl/mold_funnel.stl')
        plumeMesh.translate([0, 0, -50])
        plumeMesh.rotate([1, 0, 0], math.radians(180))
        plumeMesh.points = 0.035 * plumeMesh.points
        return plumeMesh

    # Leading "P<n>" cluster prefix of a thruster id (e.g. "P1T2" -> "P1",
    # "P10T1" -> "P10"). Reproduces the legacy first-two-character parse for
    # single-digit clusters while additionally supporting multi-digit ones.
    _CLUSTER_ID_RE = re.compile(r'^(P\d+)')

    def _cluster_id_for_thruster(self, thruster_id):
        """Resolve the cluster a thruster belongs to from its id.

        The cluster association is encoded in the thruster naming convention:
        a thruster id such as ``P1T2`` (or ``P10T1``) belongs to cluster
        ``P1`` (``P10``). Legacy code extracted only the first two characters
        (``thruster_id[0] + thruster_id[1]``), which is correct only for
        single-digit cluster numbers; this parses the full ``P<n>`` prefix so
        multi-digit clusters resolve correctly while single-digit ids behave
        identically.

        Raises
        ------
        KeyError
            If the id does not encode a ``P<n>`` prefix, or the resolved
            cluster is absent from the loaded cluster configuration. Clusters
            are never silently omitted.
        """
        match = self._CLUSTER_ID_RE.match(str(thruster_id))
        if match is None:
            raise KeyError(
                f"Thruster id {thruster_id!r} does not encode a 'P<n>' cluster "
                f"prefix; cannot resolve its cluster offset.")
        cluster_id = match.group(1)
        if cluster_id not in self.cluster_data:
            raise KeyError(
                f"Cluster {cluster_id!r} required by thruster {thruster_id!r} "
                f"is not present in the loaded cluster configuration "
                f"(available: {sorted(self.cluster_data)}).")
        return cluster_id

    def transform_plume_mesh(self, thruster_id, plumeMesh,
                             vv_orientation=None, vv_position=None):
        """Place a plume mesh for a thruster, mutating it in place.

        Canonical owner of plume-placement geometry. Applies the legacy
        transform sequence used by the RPOD visualization workflows, in this
        exact order (rotations first, about the origin, before any translation
        away from the rotation axes):

        1. Thruster DCM (transposed) — thruster frame -> visiting-vehicle frame.
        2. Visiting-vehicle DCM (transposed) — VV frame -> target/JFH frame.
           Applied only when ``vv_orientation`` is supplied.
        3. Visiting-vehicle position in the target/JFH frame. Applied only when
           ``vv_position`` is supplied.
        4. Cluster exit offset, when clusters are enabled and a vehicle pose is
           supplied (see :meth:`_cluster_id_for_thruster`).
        5. Thruster exit offset — always applied.

        Frame conventions (documentation only; do not "correct" the order):
        the stored thruster DCM maps the thruster frame to the VV frame, and
        the JFH/vehicle DCM maps the VV frame to the target frame. DCMs are
        passed exactly as stored; transposition is applied internally.

        Two placement modes:

        - Local thruster placement (``vv_orientation``/``vv_position`` both
          omitted): applies only steps 1 and 5 — the historical behavior of
          this method, used by ``check_thruster_configuration`` and
          ``LogisticsModule.plot_thruster_group``. No cluster offset is applied
          in this mode.
        - Complete vehicle/JFH placement (pose supplied): applies steps 1-5,
          the sequence used by ``graph_jfh``/``visualize_sweep``.

        Parameters
        ----------
        thruster_id : str
            Canonical thruster id (a key of ``self.thruster_data``). An unknown
            id raises ``KeyError`` via the ``thruster_data`` lookup.
        plumeMesh : stl.mesh.Mesh
            Plume mesh in its initial orientation. Mutated in place (numpy-stl
            transforms mutate); the same object is returned for convenience.
        vv_orientation : array-like, optional
            3x3 visiting-vehicle DCM as stored in the JFH (untransposed). When
            supplied, enables steps 2 and 4.
        vv_position : array-like, optional
            Length-3 visiting-vehicle position in the target/JFH frame. When
            supplied, enables step 3.

        Returns
        -------
        stl.mesh.Mesh
            The same ``plumeMesh`` object, transformed in place.

        Raises
        ------
        KeyError
            Unknown ``thruster_id``; or, when clusters are enabled, a missing
            required cluster.
        """
        full_placement = vv_orientation is not None

        # Step 1: thruster DCM (transposed). Always applied. The stored DCM is
        # expected to be 3x3; validating here fails fast on malformed
        # configuration data rather than producing silently wrong geometry.
        thruster_dcm = np.asarray(self.thruster_data[thruster_id]['dcm'])
        if thruster_dcm.shape != (3, 3):
            raise ValueError(
                f"Thruster {thruster_id!r} DCM must be 3x3, got shape "
                f"{thruster_dcm.shape}.")
        plumeMesh.rotate_using_matrix(thruster_dcm.T)

        # Step 2: visiting-vehicle DCM (transposed), when a pose is supplied.
        if vv_orientation is not None:
            vv_dcm = np.asarray(vv_orientation)
            if vv_dcm.shape != (3, 3):
                raise ValueError(
                    f"vv_orientation must be a 3x3 DCM, got shape "
                    f"{vv_dcm.shape}.")
            plumeMesh.rotate_using_matrix(vv_dcm.T)

        # Step 3: visiting-vehicle position in the target/JFH frame.
        if vv_position is not None:
            vv_pos = np.asarray(vv_position)
            if vv_pos.shape != (3,):
                raise ValueError(
                    f"vv_position must be a length-3 vector, got shape "
                    f"{vv_pos.shape}.")
            plumeMesh.translate(vv_position)

        # Step 4: cluster exit offset (legacy semantics), only for complete
        # vehicle/JFH placement with clusters enabled.
        if full_placement and getattr(self, 'use_clusters', False):
            cluster_id = self._cluster_id_for_thruster(thruster_id)
            plumeMesh.translate(self.cluster_data[cluster_id]['exit'][0])

        # Step 5: thruster exit offset. Always applied.
        plumeMesh.translate(self.thruster_data[thruster_id]['exit'][0])
        return plumeMesh

    def get_thruster_id_map(self):
        """Return the cached JFH-index -> canonical-thruster-id mapping.

        The JFH references thrusters by 1-based numeric index into the thruster
        configuration order; this maps ``'1' -> first thruster id``, and so on,
        preserving the insertion order of ``self.thruster_data`` (the legacy
        ordering rule). Values are canonical thruster ids (the string in the
        one-element ``['name']`` field), so callers no longer index a
        one-element list per lookup.

        The mapping is built lazily on first use and cached; it is invalidated
        by :meth:`set_thruster_config` whenever the configuration is replaced.
        """
        cached = getattr(self, '_thruster_id_map', None)
        if cached is None:
            cached = {}
            for index, thruster in enumerate(self.thruster_data, start=1):
                cached[str(index)] = self.thruster_data[thruster]['name'][0]
            self._thruster_id_map = cached
        return cached

    def get_thruster_id(self, jfh_index):
        """Return the canonical thruster id for a 1-based JFH thruster index.

        Parameters
        ----------
        jfh_index : int or str
            Thruster index as referenced by the JFH (1-based).

        Returns
        -------
        str
            Canonical thruster id (a key of ``self.thruster_data``).

        Raises
        ------
        KeyError
            If the index is outside the configured thruster range.
        """
        mapping = self.get_thruster_id_map()
        key = str(jfh_index)
        try:
            return mapping[key]
        except KeyError:
            raise KeyError(
                f"JFH references thruster index {jfh_index} outside the "
                f"configured range 1..{len(mapping)}.") from None

    def initiate_plume_normal(self, thruster_id):
        """
            Collects plume normal vectors data for visualization.

            Parameters
            ----------
            thruster_id : str
                String to access thruster via a unique ID.

            Returns
            -------
            [X,Y,Z,U,V,W] : 2D List
                2D list contains vector data for plume normal. This is janky but convenient for plotting.

        """

        X = []
        Y = []
        Z = []

        U = []
        V = []
        W = []

        # add position vectors to a list.
        position = self.thruster_data[thruster_id]['exit'][0]
        X.append(position[0])
        Y.append(position[1])
        Z.append(position[2])


        # add normal vectors to a list
        dcm = self.thruster_data[thruster_id]['dcm']
        U.append(dcm[0][2])
        V.append(dcm[1][2])
        W.append(dcm[2][2])


        return [X,Y,Z,U,V,W]

    def plot_vv_and_thruster(self, plumeMesh, thruster_id, normal, i):
        """
            Plots Visiting Vehicle and plume cone for provided thruster id.

            This is useful for a quick sanity check of STL file coordinates.

            Parameters
            ----------
            plumeMesh : mesh.Mesh
                Surface mesh constructed from STL file in transformed orientation.

            thruster_id : str
                String to access thruster via a unique ID.

            normal : 2D List
                2D list contains vector data for plume normal. This is janky but convenient for plotting.

            Returns
            -------
            i : int
                Integer is passed to the wrapper function for saving images with a sequential naming scheme.

        """

        # Set up nominal configuration for thruster
        VVmesh = self.mesh

        # graph vehicle and vectors.
        combined = mesh.Mesh(np.concatenate([VVmesh.data, plumeMesh.data]))

        # Instantiate data str to hold visual plots.
        figure = plt.figure()
        axes = figure.add_subplot(projection = '3d')
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(VVmesh.vectors))

        surface = mplot3d.art3d.Poly3DCollection(plumeMesh.vectors)
        surface.set_facecolor('orange')

        axes.add_collection3d(surface)
        axes.quiver(normal[0], normal[1], normal[2], normal[3], normal[4], normal[5], color = (0,0,0), length=4, normalize=True)

        lim = 7
        axes.set_xlim([-1*lim - 3, lim - 3])
        axes.set_ylim([-1*lim, lim])
        axes.set_zlim([-1*lim, lim])

        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')

        figure.suptitle(self.thruster_data[thruster_id]['name'][0])

        shift = 0

        if i < 4:
            axes.view_init(azim=0, elev=2*shift)
        elif i < 8:
            axes.view_init(azim=0, elev=2*shift)
        elif i < 12:
            axes.view_init(azim=0, elev=2*shift)
        else:
            axes.view_init(azim=0, elev=2*shift)

        if i < 10:
            index = '00' + str(i)
        elif i < 100:
            index = '0' + str(i)
        else:
            index = str(i)
        # screen_shot = vpl.screenshot_fig()
        # vpl.save_fig('img/frame' + str(index) + '.png')
        plt.savefig('img/frame' + str(index) + '.png')
        return i + 1

    def check_thruster_configuration(self):
        """
            Plots visiting vehicle and all thrusters in RCS configuration.

            Methods loads STL file of VV and turn on all thrusters to check locations + orientations.

            It is useful for a quick sanity check of the RCS configuration.
        """

        # Loop through each thruster, graphing normal vecotr and rotated plume cone.
        i = 0
        for thruster_id in self.thruster_data:

            # transform plume mesh to notional position.
            plumeMesh = self.initiate_plume_mesh()

            # transform plume mesh according to dcm data of current thruster.
            plumeMesh = self.transform_plume_mesh(thruster_id, plumeMesh)

            if not os.path.isdir('stl/tcd/'):
                os.system('mkdir stl/tcd')

            plumeMesh.save('stl/tcd/' + str(i) + '.stl')

            normal = self.initiate_plume_normal(thruster_id)

            i = self.plot_vv_and_thruster(plumeMesh, thruster_id, normal, i)

        return
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from pyrpod.mission.SubModule import SubModule
from pyrpod.mission.six_dof_dynamics import SixDOFDynamics
from pyrpod.util.io.fs import resolve_asset_path

# This one MUST stay type-only. MissionPlanner imports FlightEvaluator at
# module level, so a real import here closes the cycle and raises ImportError
# on a partially initialized module in either direction.
if TYPE_CHECKING:
    from pyrpod.mission.MissionPlanner import MissionPlanner

class FlightEvaluator(SubModule, SixDOFDynamics):
        # self.maneuvers = []

    # Set by read_flight_plan; None when the case configures no flight plan.
    flight_plan: pd.DataFrame | None

    def load_plan(self) -> None:
        # Parse CSV or other source
        pass

    def execute(self, mission_planner: MissionPlanner) -> None:
        # Loop over maneuvers and update planner
        pass


    def read_flight_plan(self, vv: Any) -> None:
        """
            Reads in VV flight as specified using CSV format.

            NOTE: Method assumes that self.case_dir and self.config are instantiated
            correctly. Potential defensive programming statements?

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        # Reads and parses through flight plan CSV file.

        self.vv = vv

        try:
            path_to_file = resolve_asset_path(
                self.case_dir, 'jfh', self.config['jfh']['flight_plan'], shared_subdir='flight_plan'
            )
        except KeyError:
            # print("WARNING: flight plan not set")
            self.flight_plan = None
            return
        self.flight_plan = pd.read_csv(path_to_file)
        # print(self.flight_plan)

        return

    def calc_flight_performance(self) -> None:
        """
            Calculates 6DOF performance for all firings specified in the flight plan.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        for firing in self.flight_plan.iterrows():  # type: ignore[union-attr]

                    # Convert firing data to numpy arra for easier data manipulation.
                    firing_array = np.array(firing[1])

                    # save firing ID
                    nth_firing = np.array(firing[1].iloc[0])
                    # print('Firing number', nth_firing)

                    # calculate required change in translational velcoity
                    v1 = firing_array[4:7]
                    v0 = firing_array[1:4]
                    dv = v1 - v0

                    # calculate required change in translational velcoity
                    w1 = firing_array[10:13]
                    w0 = firing_array[7:10]
                    dw = w1 - w0
                    # print(nth_firing, dv, dw)

                    self.set_current_6dof_state(v0, w0)
                    self.set_desired_6dof_state(v1, w1)

                    self.calc_6dof_performance()
                    # print('======================================')
        return


    def set_current_6dof_state(self, v: Sequence[float] | NDArray[Any] = [0, 0, 0],
                               w: Sequence[float] | NDArray[Any] = [0, 0, 0]) -> None:
        """
            Sets current inertial state for the VV. Can be done manually or read from flight plan.

            Parameters
            ----------
            v : 3 element list
                Contains vector components of translational velocity.

            w : 3 element list
                Contains vector components of rotational velocity.

            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
            Does the method need to return a status message? or pass similar data?
        """
        self.v_current = np.array(v)
        self.w_current = np.array(w)
        return

    def set_desired_6dof_state(self, v: Sequence[float] | NDArray[Any] = [0, 0, 0],
                               w: Sequence[float] | NDArray[Any] = [0, 0, 0]) -> None:
        """
            Sets desired inertial state for the VV. Can be done manually or read from flight plan.

            Parameters
            ----------
            v : 3 element list
                Contains vector components of translational velocity.

            w : 3 element list
                Contains vector components of rotational velocity.

            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
            Does the method need to return a status message? or pass similar data?
        """
        self.v_desired = np.array(v)
        self.w_desired = np.array(w)
        return

    def get_deltas(self) -> tuple[NDArray[Any], NDArray[Any]]:
        return self.v_desired - self.v_current, self.w_desired - self.w_current


    # calc_trans_performance is inherited from SixDOFDynamics

    def calc_6dof_performance(self) -> None:
        """
            Wrapper method used to calculate performance for translation and rotational maneuvers.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        # Wrapper function that sets up data for 6DOF performance
        dv = self.v_desired - self.v_current
        dw = self.w_desired - self.w_current

        # print('Required changes in 6DOF state')
        # print('dv', dv, 'm/s, dw', dw, 'm/s')
        # print()

        # Calculate performance for translation maneuvers
        # and assess directionality as needed
        translations = ['x', 'y', 'z']
        for i, v in enumerate(dv):
            if v ==0:
                pass
            elif v > 0:
                motion = '+' + translations[i]
                self.calc_trans_performance(motion, v)
            else:
                motion = '-' + translations[i]
                self.calc_trans_performance(motion, v)
        # print()

        # # Calculate performance for rotational maneuvers
        # # and assess directionality as needed
        # rotations = ['pitch', 'roll', 'yaw']
        # for i, v in enumerate(dv):
        #     if v ==0:
        #         pass
        #     elif v > 0:
        #         motion = '+' + rotations[i]
        #         self.calc_rot_performance(motion)
        #     else:
        #         motion = '-' + rotations[i]
        #         self.calc_rot_performance(motion)
        return

    # Byte-for-byte duplicate of the definition above; this one unconditionally
    # shadows it at class-creation time. Removing the earlier copy is a code
    # change, which #103 excludes, so the shadowing is flagged here instead
    # (see the report's deferred observations).
    def calc_flight_performance(self) -> None:  # type: ignore[no-redef]
        """
            Calculates 6DOF performance for all firings specified in the flight plan.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        for firing in self.flight_plan.iterrows():  # type: ignore[union-attr]

                    # Convert firing data to numpy arra for easier data manipulation.
                    firing_array = np.array(firing[1])

                    # save firing ID
                    nth_firing = np.array(firing[1].iloc[0])
                    # print('Firing number', nth_firing)

                    # calculate required change in translational velcoity
                    v1 = firing_array[4:7]
                    v0 = firing_array[1:4]
                    dv = v1 - v0

                    # calculate required change in translational velcoity
                    w1 = firing_array[10:13]
                    w0 = firing_array[7:10]
                    dw = w1 - w0
                    # print(nth_firing, dv, dw)

                    self.set_current_6dof_state(v0, w0)
                    self.set_desired_6dof_state(v1, w1)

                    self.calc_6dof_performance()
                    # print('======================================')
        return

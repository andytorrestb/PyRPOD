from __future__ import annotations

from typing import Any


class SixDOFDynamics:
    # def __init__(self):
        # self.thruster_model = thruster_model

    # Supplied by whichever object mixes this class in (FlightEvaluator sets
    # it in read_flight_plan); SixDOFDynamics never constructs it itself.
    vv: Any

    def evaluate_translational(self, dv: Any) -> None:
        # Placeholder logic for translational maneuver
        pass

    def evaluate_rotational(self, dw: Any) -> None:
        # Placeholder logic for rotational maneuver
        pass

    def evaluate(self, dv: Any, dw: Any) -> None:
        # Combined 6DOF evaluation logic
        self.evaluate_translational(dv)
        self.evaluate_rotational(dw)

    def calc_trans_performance(
        self, motion: str, dv: float
    ) -> tuple[float, float, float] | None:
        """
            Calculates RCS performance according to thruster working groups for a direction of motion.

            This method assumes constant mass, which needs to be addressed.

            Needs better name?

            Parameters
            ----------
            dv : float
                Speficied change in velocity value.

            motion : str
                Directionality of motion. Used to select active thrusters.

            Returns
            -------
            time : float
                Burn time elapsed.

            distance : float
                Distance covered during burn time.

            propellant_used : float
                Propellant used during burn time.
        """
        # Calculate RCS performance according to thrusters grouped to be in the direction.
        # WIP: Initial code executes simple 1DOF calculations
        # print(type(self.vv))
        # print(self.vv)
        if self.vv.rcs_groups == None:
            # print("WARNING: Thruster Grouping File not Set")
            # mypy asks for an explicit `return None` whenever the declared
            # return type is not plain None, even when None is part of the
            # union. A bare return and `return None` compile identically, so
            # the source is left as-is rather than edited for the checker.
            return  # type: ignore[return-value]

        n_thrusters = len(self.vv.rcs_groups[motion])
        total_thrust = n_thrusters * self.vv.thrust
        acceleration = total_thrust / self.vv.mass
        # print(acceleration)
        time = abs(dv) / acceleration
        distance = 0.5 * abs(dv) * time
        m_dot = total_thrust / self.vv.isp
        propellant_used = m_dot * time

        # Print info to screen (TODO: write this to a data structure)
        p = 2 # how many decimals places to print
        # print('Total thrust produced', round(total_thrust, p), 'N')
        # print('Resulting accelration', round(acceleration, p), 'm / s ^ 2')
        # print('Time required', round(time, p), 's')
        # print('Distance Covered', round(distance, p), 'm')
        # print('Total propellant used', round(propellant_used, p), 'kg')

        return time, distance, propellant_used
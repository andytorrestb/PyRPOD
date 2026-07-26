from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import matplotlib.pyplot as plt
import logging

if TYPE_CHECKING:
    # Type-only: this module does no pandas work itself, it only reads a
    # DataFrame handed to it by the owning object.
    import pandas as pd

logger = logging.getLogger(__name__)


class PostProcessor:
    # Collaborator state and helpers the plotting methods read but this class
    # never defines; they come from the object that owns the post-processor
    # (see MissionPlanner, FuelManager and SixDOFDynamics). Declared bare so no
    # class attribute is created at runtime.
    vv: Any
    flight_plan: pd.DataFrame
    calc_burn_time: Callable[[float, float, float], float]
    calc_delta_mass: Callable[[float, float], float]
    calc_trans_performance: Callable[[str, Any], tuple[float, float, float]]

    def __init__(self) -> None:
        pass

    def plot_burn_profiles(self) -> None:
        # Stub plot for burn times
        pass

    def plot_mass_usage(self) -> None:
        # Stub plot for fuel usage
        pass
    def plot_burn_time(self, dv: float) -> None:
        """
            Plots burn time for a given dv and isp value. Varies thrust according to user inputs.

            TODO: Add ISP value as a parameter. Remove isp_vals, add isp as a parameter to the function.
            Test code.

            TODO: Integrate with establsihed configuration file framework.

            Parameters
            ----------
            dv : float
                Specified change in velocity value.

            isp : float
                Specified specific impulse value.

            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
            Does the method need to return a status message? or pass similar data?

        """
        isp_vals = [50, 200, 300, 400, 500]
        thrust_range = np.linspace(50, 600, 5000)
        # These plotting accumulators start as a Python list and are then
        # rebound to the NumPy array built from them. A list|NDArray union does
        # not survive mypy's loop binder (it widens back at the loop edge and
        # rejects .append), so the rebound locals in this module stay dynamic.
        burn_time: Any = []

        isp = 200
        for thrust in thrust_range:
            burn_time.append(abs(self.calc_burn_time(dv, isp, thrust)))
        burn_time = np.array(burn_time)

        fig, ax = plt.subplots()


        ax.plot(thrust_range, burn_time)
        ax.set(xlabel='Thrust (s)', ylabel='Burn-time (s)',
            title='Thrust vs Burn-time Required (' + str(abs(dv)) + ')')
        ax.grid()
        ax.legend()
        # plt.xscale("log")
        # plt.yscale("log")
        fig.savefig("test.png")

    def plot_burn_time_contour(self, dv: float) -> None:
        """
            Plots burn time for a given dv by varrying thrust values. Graph is contoured using ISP values.

            TODO: Add isp_vals as a parameter. Integrate with configuration file framework. Test Code.

            Parameters
            ----------
            dv : float
                Specified change in velocity value.

            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
            Does the method need to return a status message? or pass similar data?
        """
        isp_vals = [300]
        thrust_range = np.linspace(1, 1000, 5000)

        fig, ax = plt.subplots()

        for isp in isp_vals:
            burn_time: Any = []
            for thrust in thrust_range:
                burn_time.append(abs(self.calc_burn_time(dv, isp, thrust)))
            burn_time = np.array(burn_time) / (3600*24)
            ax.plot(thrust_range, burn_time, label = 'ISP = (' + str(abs(isp)) + ' s)')

        ax.set(xlabel='Thrust (N)', ylabel='Burn-time (days)',
            title='Thrust vs Burn-time Required (Δv = ' + str(abs(dv)) + ' m/s)')
        ax.grid()
        ax.legend()
        plt.xscale("log")
        # plt.yscale("log")
        fig.savefig("test.png")
        return

    def plot_burn_time_flight_plan(self) -> None:
        """
            Plots burn time for all dv maneuvers in the specified flight plan.

            Parameters
            ----------
            None

            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
            Does the method need to return a status message? or pass similar data?
        """
        isp = 300
        thrust_range = np.linspace(10, 100, 5000)

        fig, ax = plt.subplots()

        dv = self.flight_plan.iterrows()

        for v in dv:
            # print(type(v[1]))
            dv = v[1][1]
            logger.debug('Processing dv entry in flight plan')
            burn_time: Any = []
            for thrust in thrust_range:
                burn_time.append(abs(self.calc_burn_time(dv, isp, thrust)))
            burn_time = np.array(burn_time) / (3600*24)
            ax.plot(thrust_range, burn_time, label = 'Δv = (' + str(abs(dv)) + ' m/s)')

        ax.set(xlabel='Thrust (N)', ylabel='Burn-time (days)',
            title='Burn-time Required vs Thrust (ISP = ' + str(abs(300)) + ' s)')
        ax.grid()
        ax.legend()
        # plt.xscale("log")
        # plt.yscale("log")
        fig.savefig("test.png")

        return

    def plot_delta_mass(self, dv: float) -> None:
        """
            Plots propellant usage for a given dv requirements by varying ISP according to user inputs.

            Parameters
            ----------
            dv : float
                Speficied change in velocity value.

            Returns
            -------
            Method doesn't currently return anything. Simply sets class members as needed.
            Does the method need to return a status message? or pass similar data?
        """
        isp_range = np.linspace(100, 600, 5000)
        delta_mass: Any = []

        for isp in isp_range:
            delta_mass.append(abs(self.calc_delta_mass(dv, isp)))
        delta_mass = np.array(delta_mass)
        # for i, isp, in enumerate(isp_range):
        #     print(isp_range[i], delta_mass[i])

        thrust_tech = {
            'electro thermal': [50, 185],
            # 'hall-effect': [800, 1950],
            'cold-warm-gas': [30, 110],
            'mono-bi-propellants': [160, 310]
        }

        fig, ax = plt.subplots()
        for tech in thrust_tech:
            # print(tech)

            y_vals = np.array([delta_mass.max(), delta_mass.mean(), delta_mass.min()])
            isp_val = thrust_tech[tech][1]
            isp_line = np.array([isp_val, isp_val, isp_val])

            ax.plot(isp_line, y_vals, label=tech)

        ax.plot(isp_range, delta_mass)
        ax.set(xlabel='ISP (s)', ylabel='mass (kg)',
            title='Max ISP vs Propellant Mass Required (' + str(abs(dv)) + ' m/s)')
        ax.grid()
        ax.legend()
        # plt.xscale("log")
        # plt.yscale("log")
        fig.savefig("test.png")

    def plot_delta_mass_contour(self) -> None:
    
        """
            Co-Plots propellant usage for all dv maneuvers in the specified flight plan.

            TODO: Add isp_range as a parameter. Integrate with configuration file framework. Test Code.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        #creat plotting object.
        fig, ax = plt.subplots()

        delta_mass_min: float = 10e9
        delta_mass_max: float = 0

        # Step through all planned firings in the flight plan
        for firing in self.flight_plan.iterrows():
            # save delta v requirement to a local variable.
            dv = firing[1][1]

            # Calculate change in mass for a given range of ISP values.
            isp_range = np.linspace(50, 400, 5000)
            delta_mass: Any = []

            for isp in isp_range:
                delta_mass.append(abs(self.calc_delta_mass(dv, isp)))
            delta_mass = np.array(delta_mass)

            # Save absolute min and max data for plotting.
            if delta_mass.max() > delta_mass_max:
                delta_mass_max = delta_mass.max()

            if delta_mass.min() < delta_mass_min:
                delta_mass_min = delta_mass.min()

            # Plot data.
            ax.plot(isp_range, delta_mass, label='( Δv =' + str(abs(dv)) + ' m/s)')

        # thrust_tech = {
        #     # 'electro thermal': [50, 185],
        #     # 'hall-effect': [800, 1950],
        #     'cold-warm-gas': [30, 110],
        #     'mono-bi-propellants': [160, 310]
        # }

        # for tech in thrust_tech:
        #     print(tech)

        #     y_vals = np.array([delta_mass_max, 0.5*(delta_mass_max + delta_mass_min), delta_mass_min])
        #     isp_val = thrust_tech[tech][1]
        #     isp_line = np.array([isp_val, isp_val, isp_val])

        #     ax.plot(isp_line, y_vals, label=tech, linestyle='dotted')

        # Set plot display parameters.
        ax.set(xlabel='Specific Impulse (s)', ylabel='Propellant Mass Required (kg)',
            title='Propellant Mass Required vs Specific Impulse')
        ax.grid()
        ax.legend()
        # plt.xscale("log")
        # plt.yscale("log")
        fig.tight_layout(pad=1.8)

        # Save to file
        fig.savefig("test.png")
        return

    def plot_thrust_envelope(self) -> None:
        """
            Plots operational envelope relating burn time to thrust required for all firings in the flight plan.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        # print(self.vv)
        # print(self.flight_plan)

        for firing in self.flight_plan.iterrows():
            # Parse flight plan data.
            # print(firing)
            firing_array = np.array(firing[1])
            # print(firing_array)

            # calculate required change in translational velcoity
            v1 = firing_array[4:7]
            v0 = firing_array[1:4]
            dv = v1 - v0

            # Create lists to hold data for plotting
            time_req = []
            distance_req = []

            # Create range of thrust values to claculate.
            thrust_vals = np.linspace(10, 1000, 100)

            for thrust in thrust_vals:

                self.vv.add_thruster_performance(thrust, 100)
                time, distance, propellant_used = self.calc_trans_performance('+x', dv)
                
                time_req.append(time)
                distance_req.append(distance)

            fig, ax = plt.subplots()
            ax.plot(time_req, thrust_vals)


            ax.set(xlabel='time (s)', ylabel='thrust (N)',
                title='Thrust vs Time Required (' + str(abs(dv[0])) + ')')
            
            ax.grid()
            plt.xscale("log")
            # plt.yscale("log")


            fig.savefig("test" + str(firing[0]) + ".png")
            plt.show()
        return
    
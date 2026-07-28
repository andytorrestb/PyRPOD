"""
Trade studies over RPOD / plume-impingement configurations.

Two entry points live here:

* the legacy dynamics-driven sweeps (``run_axial_overshoot_sweep``,
  ``run_surface_cant_sweep``, ``run_multi_var_sweep``), which vary an
  approach velocity and/or a thruster cant angle and report a mission
  summary per configuration;
* the package-level PRESCRIBED validation API,

      >>> study = TradeStudy.from_config('flat_plate_baseline.yaml')
      >>> results = study.run()

  which runs a YAML-configured plume/target validation sweep through
  :class:`pyrpod.mdao.plume_validation.PlumeValidationStudy` and returns
  structured results. The heavy lifting lives in the small modules of
  ``pyrpod.mdao`` (study configuration, firing plan, surface-load
  integration, results, reference comparison, plots), so this class stays a
  thin façade over them.
"""

from __future__ import annotations

import os
import csv
import time
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt

from pyrpod.rpod import JetFiringHistory, PlumeStrikeEstimationStudy
from pyrpod.mdao import SweepConfig
from pyrpod.mdao.parameter_sweep import ParameterSweepStudy
from pyrpod.mdao.plume_validation import PlumeValidationStudy
from pyrpod.mdao.reference_data import (
    ComparisonReport,
    ReferenceDataset,
    load_reference_dataset,
)
from pyrpod.mdao.study_config import StudyConfig
from pyrpod.mdao.study_results import StudyResults
from pyrpod.mission import MissionEnvironment
from pyrpod.util.io.fs import ensure_dir
import configparser

from pyrpod.vehicle.LogisticsModule import LogisticsModule
from pyrpod.vehicle.TargetVehicle import TargetVehicle

logger = logging.getLogger(__name__)


def _engine_for(
    study_config: StudyConfig,
) -> PlumeValidationStudy | ParameterSweepStudy:
    """Select the study engine named by the configuration's sweep mode.

    ``per_case`` (the default) gives every angle-distance combination its own
    Jet Firing History and strike run; ``single_jfh`` runs the whole sweep as
    one history, so every firing's strikes come from the same pipeline run
    and the cumulative fields form a sweep envelope.
    """
    if study_config.sweep.mode == "single_jfh":
        return ParameterSweepStudy(study_config)
    return PlumeValidationStudy(study_config)


class TradeStudy():
    # Built by init_trade_study and read by every sweep; max_v0 is recorded
    # by the sweeps and read back by print_mission_report. Declared bare so
    # no class attribute is created at runtime.
    rpod: Any
    max_v0: Any

    def __init__(self, case_dir: str,
                 study_config: StudyConfig | None = None) -> None:
        self.case_dir = case_dir
        config = configparser.ConfigParser()
        config.read(self.case_dir + "config.ini")
        self.config = config
        # Present only for a study built by from_config(); the legacy
        # constructor keeps its original single-argument behavior. The engine
        # is selected by the configuration's sweep mode: one Jet Firing
        # History per case, or one spanning the whole sweep.
        self.study_config = study_config
        self._validation_study = (_engine_for(study_config)
                                  if study_config is not None else None)

    # ------------------------------------------------------- package API
    @classmethod
    def from_config(cls, path: str | os.PathLike[str],
                    output_dir: str | os.PathLike[str] | None = None,
                    ) -> "TradeStudy":
        """Build a prescribed validation study from a YAML configuration.

        Parameters
        ----------
        path : str or path-like
            Path to a study configuration file (see
            :class:`pyrpod.mdao.study_config.StudyConfig` and
            ``docs/plume_validation_study.md``).
        output_dir : str or path-like, optional
            Overrides the configured output directory, so the same committed
            configuration can be run into a scratch location (this is what
            the automated tests do).

        Returns
        -------
        TradeStudy
            A study bound to the parsed configuration; call :meth:`run`.
            Which engine backs it follows ``sweep.mode``:
            ``per_case`` (default) binds
            :class:`~pyrpod.mdao.plume_validation.PlumeValidationStudy`,
            ``single_jfh`` binds
            :class:`~pyrpod.mdao.parameter_sweep.ParameterSweepStudy`.
        """
        config = StudyConfig.from_yaml(path)
        if output_dir is not None:
            config = config.with_output_dir(os.fspath(output_dir))
        return cls(config.case_dir, study_config=config)

    @property
    def validation_study(self) -> PlumeValidationStudy | ParameterSweepStudy:
        """The bound study engine (requires from_config)."""
        if self._validation_study is None:
            raise RuntimeError(
                "this TradeStudy was not built from a study configuration; "
                "use TradeStudy.from_config(path) for prescribed plume "
                "validation sweeps")
        return self._validation_study

    def run(self, write_outputs: bool = True) -> StudyResults:
        """Run the configured sweep through its selected engine."""
        return self.validation_study.run(write_outputs=write_outputs)

    def compare(self, reference: ReferenceDataset | str | os.PathLike[str]
                ) -> ComparisonReport:
        """Compare the last run against external reference data.

        The reference may be a loaded :class:`ReferenceDataset` or a path to
        a CSV / JSON / YAML file in the generic reference format; where the
        data came from (DSMC, analytical, experiment) is irrelevant here.
        """
        if not isinstance(reference, ReferenceDataset):
            reference = load_reference_dataset(reference)
        return self.validation_study.compare(reference)

    def plot(self) -> list[str]:
        """Generate the optional sweep trend plots for the last run."""
        return self.validation_study.plot()

    # ------------------------------------------------------ legacy sweeps
    def init_trade_study(self, lm: LogisticsModule, tv: TargetVehicle) -> None:
        """
        Organizes data needed to kick off an RPOD trade study.

        Mainly done by properly configuring a plume-strike study object.

        """
        # Save variable name for readability.
        case_dir = self.case_dir

        # Instantiate JetFiringHistory object.
        jfh = JetFiringHistory.JetFiringHistory(case_dir)

        # Instantiate the plume-strike study object. It takes a
        # MissionEnvironment (MissionPlanner's constructor argument), not a
        # case directory; the previous call named a class that does not
        # exist in the module and raised AttributeError before any sweep
        # could start.
        environment = MissionEnvironment.MissionEnvironment(case_dir)
        rpod = PlumeStrikeEstimationStudy.PlumeStrikeEstimationStudy(
            environment)
        rpod.study_init(jfh, tv, lm)
        self.environment = environment
        self.rpod = rpod

    def init_trade_study_case(self) -> None:
        """
        Resets JFH data according to current case key.

        Case key is a unique identified for a specific configuration within the trade study. 
        """

        # Instantiate JetFiringHistory object.
        jfh = JetFiringHistory.JetFiringHistory(self.case_dir)
        jfh.config['jfh']['jfh'] = self.rpod.get_case_key() + ".A"
        jfh.read_jfh()

        self.rpod.jfh = jfh

    @staticmethod
    def summarize_firing_data(
        firing_data: Mapping[str, Mapping[str, NDArray[np.float64]]]
    ) -> dict[str, float]:
        """Reduce a plume-strike run's per-firing arrays to case maxima.

        ``jfh_plume_strikes`` returns the per-face arrays for every firing
        (including the running cumulative ones); the mission report wants one
        number per quantity per configuration, so they are reduced here
        instead of being read off study attributes that the plume-strike
        study never sets.
        """
        summary = {"max_pressure": 0.0, "max_shear": 0.0,
                   "max_heat_rate": 0.0, "max_heat_load": 0.0,
                   "max_cum_heat_load": 0.0}
        for fields in firing_data.values():
            for key, field_name in (("max_pressure", "pressures"),
                                    ("max_shear", "shear_stress"),
                                    ("max_heat_rate", "heat_flux_rate"),
                                    ("max_heat_load", "heat_flux_load"),
                                    ("max_cum_heat_load",
                                     "cum_heat_flux_load")):
                values = fields.get(field_name)
                if values is None or len(values) == 0:
                    continue
                summary[key] = max(summary[key], float(np.max(values)))
        return summary

    def print_mission_report(
        self,
        firing_data: Mapping[str, Mapping[str, NDArray[np.float64]]] | None
        = None,
    ) -> None:
        """Append one configuration's summary row to results/MissionReport.csv.

        ``firing_data`` is the mapping returned by
        ``PlumeStrikeEstimationStudy.jfh_plume_strikes()``; when supplied the
        impingement maxima are reduced from it. It is optional so an existing
        caller that sets the maxima on the study object itself keeps working.
        """

        case_key = self.rpod.get_case_key()

        max_v0 = self.max_v0

        fuel_mass = getattr(self.rpod, 'fuel_mass', '')

        if firing_data is not None:
            summary = self.summarize_firing_data(firing_data)
            max_pressure = summary['max_pressure']
            max_shear = summary['max_shear']
            max_heat_rate = summary['max_heat_rate']
            max_heat_load = summary['max_heat_load']
            max_cum_heat_load = summary['max_cum_heat_load']
        else:
            max_pressure = self.rpod.max_pressure
            max_shear = self.rpod.max_shear
            max_heat_rate = self.rpod.max_heat_rate
            max_heat_load = self.rpod.max_heat_load
            max_cum_heat_load = self.rpod.max_cum_heat_load

        # Check if the file exists
        report_path = self.case_dir + 'results/MissionReport.csv'
        file_exists = os.path.isfile(report_path)

        # Open CSV file in append mode
        with open(report_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write header if the file is newly created
            if not file_exists:
                csv_writer.writerow(['CaseKey', 'MaxV0', 'FuelMass', 'MaxPressure', 'MaxShear', 'MaxHeatRate', 'MaxHeatLoad', 'MaxCumulativeHeatLoad'])

            # Write data row
            csv_writer.writerow([
                case_key,
                max_v0,
                fuel_mass,
                max_pressure,
                max_shear,
                max_heat_rate,
                max_heat_load,
                max_cum_heat_load
            ])

    def graph_mission_report(self, report_results: pd.DataFrame) -> None:
        """
        """
        # Extract the first column (assuming it's the parameter you want to plot against)
        first_column = report_results.columns[0]
        # Convert DataFrame columns to NumPy arrays for indexing
        x_values = report_results[first_column].to_numpy()
        # Do not log full DataFrames at INFO; a compact DEBUG summary suffices.
        logger.debug("Mission report DataFrame: shape=%s columns=%s",
                     report_results.shape, list(report_results.columns))
        # Plot each parameter against the first parameter
        for column in report_results.columns[1:8]:
            y_values = report_results[column].to_numpy()
            plt.figure()  # Create a new figure for each plot
            plt.plot(x_values,y_values, label=column)
            plt.xlabel(first_column)
            plt.ylabel(column)
            plt.title(f'{column} vs {first_column}')
            plt.legend()
            plt.grid(True)

        # Show all plots
        plt.show()


    def interpret_mission_report(self) -> None:
        """
        """
        report_path = self.case_dir + 'results/MissionReport.csv'
        report_results = pd.read_csv(report_path)

        # The plume-strike study holds its configuration on its
        # MissionEnvironment; `self.rpod.config` never existed and raised
        # AttributeError here.
        config = self.rpod.environment.config
        max_pressure = float(config['tv']['normal_pressure'])
        max_shear = float(config['tv']['shear_pressure'])
        max_heat_rate = float(config['tv']['heat_flux'])
        # max_heat_load = float(config['tv'][]) no such constraint
        max_cum_heat_load = float(config['tv']['heat_flux_load'])

        plume_status = []
        plume_failure_mode = []

        for i, row in report_results.iterrows():
            if row['MaxPressure'] > max_pressure:
                plume_status.append('fail')
                plume_failure_mode.append('pressure')
            elif row['MaxShear'] > max_shear:
                plume_status.append('fail')
                plume_failure_mode.append('shear')
            elif row['MaxHeatRate'] > max_heat_rate:
                plume_status.append('fail')
                plume_failure_mode.append('heat_flux')
            # elif report_results['MaxHeatLoad'] > max_heat_load:  no such constraint
            #     plume_status.append('fail')
            elif row['MaxCumulativeHeatLoad'] > max_cum_heat_load:
                plume_status.append('fail')
                plume_failure_mode.append('cumulative_heat_flux_load')
            else:
                plume_status.append('pass')
                plume_failure_mode.append('none')

        report_results['PlumeStatus'] = plume_status
        report_results['PlumeFailureMode'] = plume_failure_mode
        self.graph_mission_report(report_results)

    def run_axial_overshoot_sweep(self, sweep_vars: Mapping[str, Any],
                                  lm: LogisticsModule,
                                  tv: TargetVehicle) -> None:
        """
        Simple variable sweep study that assesses RCS performance for a given set of axial overshoot velocity values.
        """

        # Organize variables to sweet over.
        axial_overshoot = sweep_vars['axial_overshoot']
        self.max_v0 = np.max(axial_overshoot)

        logger.info("Trade study initialized: sweep_type=axial_overshoot "
                    "case_dir=%s n_configurations=%d", self.case_dir,
                    len(axial_overshoot))
        study_start = time.perf_counter()

        # Link elements for RPOD analysis.
        self.init_trade_study(lm, tv)

        # Create results directory if necessary.
        results_dir = self.case_dir + 'results'
        ensure_dir(results_dir)

        # Loop through over shoot velocities to test.
        for i, v_o in enumerate(axial_overshoot):

            self.rpod.set_case_key(i, 0)
            config_start = time.perf_counter()
            logger.info("Trade-study config %d/%d: case_key=%s "
                        "axial_overshoot_m_s=%s", i + 1, len(axial_overshoot),
                        self.rpod.get_case_key(), v_o)

            # Create JFH for a given velocity.
            self.rpod.print_jfh_1d_approach_n_fire(
                                    tv.v_ida,
                                    v_o,
                                    tv.r_o,
                                    n_firings = 100,
                                    trade_study = True
                                )

            # Reset JFH according to specific case.
            self.init_trade_study_case()
            self.rpod.graph_jfh(trade_study= True)
            # jfh_plume_strikes takes (parallel, workers); the previous
            # trade_study keyword raised TypeError. The per-firing arrays it
            # returns are what the mission report summarizes.
            firing_data = self.rpod.jfh_plume_strikes()

            self.print_mission_report(firing_data)
            logger.info("Trade-study config %d/%d completed: case_key=%s "
                        "config_time_s=%.3f", i + 1, len(axial_overshoot),
                        self.rpod.get_case_key(),
                        time.perf_counter() - config_start)
        report_path = self.case_dir + 'results/MissionReport.csv'
        logger.info("Trade study completed: sweep_type=axial_overshoot "
                    "configurations=%d mission_report=%s total_time_s=%.3f "
                    "status=successful", len(axial_overshoot), report_path,
                    time.perf_counter() - study_start)
        self.interpret_mission_report()

    def run_surface_cant_sweep(self, sweep_vars: Mapping[str, Any],
                               lm: LogisticsModule,
                               tv: TargetVehicle) -> None:
        """
        """

        # Organize variables to sweet over.
        surface_cant_angles = sweep_vars['surface_cant_angles']
        v_o = sweep_vars['axial_overshoot']
        self.max_v0 = np.max(v_o)

        # Link elements for RPOD analysis.
        self.init_trade_study(lm, tv)

        angle_sweep = SweepConfig.SweepDecelAngles(lm.thruster_data, lm.rcs_groups)

        # Create results directory if necessary.
        results_dir = self.case_dir + 'results'
        ensure_dir(results_dir)

        # Loop through over shoot velocities to test.
        for i, cant in enumerate(surface_cant_angles):

            cant = np.deg2rad(cant)
            lm.decel_cant = cant

            new_tcd = angle_sweep.cant_decel_thrusters(cant)   
            lm.set_thruster_config(new_tcd)

            # print(i, v_o)
            # # Set unique case identifier within trade study.
            self.rpod.set_case_key(0, i)

            # Create JFH for a given velocity.
            self.rpod.print_jfh_1d_approach_n_fire(
                                    tv.v_ida,
                                    v_o,
                                    tv.r_o,
                                    n_firings = 100,
                                    trade_study = True
                                )

            # Reset JFH according to specific case.
            self.init_trade_study_case()              
    
            self.rpod.graph_jfh(trade_study= True)
            firing_data = self.rpod.jfh_plume_strikes()

            self.print_mission_report(firing_data)
        self.interpret_mission_report()

    def run_multi_var_sweep(self, sweep_vars: Mapping[str, Any],
                            lm: LogisticsModule,
                            tv: TargetVehicle) -> None:
        """
        """
        # Organize variables to sweet over.
        axial_overshoot = sweep_vars['axial_overshoot']
        surface_cant_angles = sweep_vars['surface_cant_angles']
        self.max_v0 = np.max(axial_overshoot)

        # Link elements for RPOD analysis.
        self.init_trade_study(lm, tv)

        angle_sweep = SweepConfig.SweepDecelAngles(lm.thruster_data, lm.rcs_groups)

        # Create results directory if necessary.
        results_dir = self.case_dir + 'results'
        ensure_dir(results_dir)

        # Loop through over shoot velocities to test.
        for i, v_o in enumerate(axial_overshoot):
            for j, cant in enumerate(surface_cant_angles):

                lm.decel_cant = cant

                new_tcd = angle_sweep.cant_decel_thrusters(cant)   
                lm.set_thruster_config(new_tcd)

                # print(i, v_o)
                # # Set unique case identifier within trade study.
                self.rpod.set_case_key(i, j)

            # Create JFH for a given velocity.
                self.rpod.print_jfh_1d_approach_n_fire(
                                    tv.v_ida,
                                    v_o,
                                    tv.r_o,
                                    n_firings = 10,
                                    trade_study = True
                                )

                # Reset JFH according to specific case.
                self.init_trade_study_case()              
        
                self.rpod.graph_jfh(trade_study= True)
                firing_data = self.rpod.jfh_plume_strikes()

                self.print_mission_report(firing_data)
        self.interpret_mission_report() 
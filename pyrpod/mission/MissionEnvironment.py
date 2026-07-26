from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
import numpy as np

import configparser


from pyrpod.vehicle import VisitingVehicle, TargetVehicle
from pyrpod.rpod import JetFiringHistory

@dataclass
class MissionEnvironment:

    def __init__(self, case_dir: str) -> None:
        self.case_dir = case_dir
        config = configparser.ConfigParser()
        config.read(self.case_dir + "config.ini")
        self.config = config

        self.jfh = JetFiringHistory.JetFiringHistory(self.case_dir)

        self.vv = VisitingVehicle.VisitingVehicle(self.case_dir)
        self.tv = TargetVehicle.TargetVehicle(self.case_dir)

    # --- Configuration and Setup ---
    # config: configparser.ConfigParser()  # Replace with specific config type if defined
    flight_plan: Any  # Replace with a structured FlightPlan class
    # jfh: JetFiringHistory.JetFiringHistory(self.case_dir)  # JetFiringHistory object
    # plume_model: Any  # Object encapsulating plume equations/parameters
    # thruster_data: Any  # Thruster performance characteristics

    # # --- Vehicle Models ---
    # vv: VisitingVehicle.VisitingVehicle()  # VisitingVehicle
    # tv: TargetVehicle.TargetVehicle()  # TargetVehicle

    # --- Kinematic State ---
    v_current: Optional[np.ndarray] = None
    w_current: Optional[np.ndarray] = None
    v_desired: Optional[np.ndarray] = None
    w_desired: Optional[np.ndarray] = None

    # --- Runtime Results / Logs ---
    delta_v_log: List[Dict[str, Any]] = field(default_factory=list)
    burn_log: List[Dict[str, Any]] = field(default_factory=list)
    impingement_log: List[Dict[str, Any]] = field(default_factory=list)
    mass_profile: List[float] = field(default_factory=list)

    # --- Accessors and Mutators ---
    # The four kinematic fields default to None and nothing in this class
    # populates them, so the returned mapping is Optional-valued. The previous
    # Dict[str, np.ndarray] annotation was wrong on a freshly built
    # environment, which is the only state PyRPOD ever puts one in today.
    def get_current_state(self) -> Dict[str, Optional[np.ndarray]]:
        return {
            "v_current": self.v_current,
            "w_current": self.w_current,
            "v_desired": self.v_desired,
            "w_desired": self.w_desired,
        }

    def update_state(self, v: np.ndarray, w: np.ndarray) -> None:
        self.v_current = v
        self.w_current = w

    def set_desired_state(self, v_des: np.ndarray, w_des: np.ndarray) -> None:
        self.v_desired = v_des
        self.w_desired = w_des

    def get_jfh_segment(self, t: float) -> Any:
        # As the comment says, this assumes a query interface that
        # JetFiringHistory does not implement; the accessor has never worked.
        # Flagged rather than fixed (see the report's deferred observations).
        # Assumes jfh provides a query interface by time
        return self.jfh.query(t)  # type: ignore[attr-defined]

    def get_thruster_by_id(self, thruster_id: str) -> Any:
        # thruster_data is never assigned: neither __init__ nor any dataclass
        # field defines it, and the commented-out declaration above was never
        # reinstated. Declaring it here is not an option -- this is a dataclass
        # with fields that already carry defaults, so a bare annotation in the
        # class body would add a field and break class creation. Flagged rather
        # than fixed; see the report's deferred observations.
        return self.thruster_data.get(thruster_id)  # type: ignore[attr-defined]

    def log_impingement(self, strike_data: Dict[str, Any]) -> None:
        self.impingement_log.append(strike_data)

    def log_burn(self, burn_data: Dict[str, Any]) -> None:
        self.burn_log.append(burn_data)

    def log_delta_v(self, dv_data: Dict[str, Any]) -> None:
        self.delta_v_log.append(dv_data)

    def update_mass_profile(self, mass: float) -> None:
        self.mass_profile.append(mass)

    # Was annotated "MissionContext", a class that does not exist anywhere in
    # the repository; copy.deepcopy(self) returns this class.
    def clone(self) -> "MissionEnvironment":
        # Deep copy method for simulation branching (if needed)
        import copy
        return copy.deepcopy(self)

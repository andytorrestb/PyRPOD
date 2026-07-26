from __future__ import annotations

from pyrpod.mission.MissionEnvironment import MissionEnvironment


class SubModule:
    def __init__(self, environment: MissionEnvironment) -> None:
        self.environment = environment
        self.case_dir = environment.case_dir
        self.config = environment.config
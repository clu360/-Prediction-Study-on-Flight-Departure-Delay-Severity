"""Validated, reproducible preprocessing policy."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import yaml


@dataclass(frozen=True)
class Config:
    max_turnaround_minutes: int = 1440
    sensitivity_turnaround_minutes: int = 720
    prediction_offset_minutes: int = 1
    clock_tolerance_minutes: int = 1
    evaluation_airports: tuple[str, ...] = ("ATL", "DFW", "DEN", "ORD", "LAX")

    def __post_init__(self) -> None:
        if self.prediction_offset_minutes != 1:
            raise ValueError("The locked forecast proxy is takeoff + 1 minute")
        if not 0 < self.sensitivity_turnaround_minutes <= self.max_turnaround_minutes:
            raise ValueError("Invalid turnaround intervals")
        if not 0 <= self.clock_tolerance_minutes <= 2:
            raise ValueError("Clock tolerance must remain small (0..2 minutes)")

    @classmethod
    def load(cls, path: Path | None) -> "Config":
        values = yaml.safe_load(path.read_text()) if path else {}
        if "evaluation_airports" in values:
            values["evaluation_airports"] = tuple(values["evaluation_airports"])
        return cls(**values)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

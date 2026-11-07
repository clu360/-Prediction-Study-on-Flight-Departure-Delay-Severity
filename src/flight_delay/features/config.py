from dataclasses import asdict, dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class FeatureConfig:
    history_days: int = 28
    schedule_days: int = 7
    alpha: float = 20.0
    history_regime: str = "causal_online"
    matrix_batch_rows: int = 50000
    evaluation_airports: tuple[str, ...] = ("ATL", "DFW", "DEN", "ORD", "LAX")

    def __post_init__(self) -> None:
        if (self.history_days, self.schedule_days, self.alpha, self.history_regime) != (28, 7, 20, "causal_online"):
            raise ValueError("Phase II locks 28d outcomes, 7d schedules, alpha=20 and causal online histories")
        if self.matrix_batch_rows < 1:
            raise ValueError("matrix_batch_rows must be positive")

    @classmethod
    def load(cls, path: Path | None = None) -> "FeatureConfig":
        values = yaml.safe_load(path.read_text()) if path else {}
        if "evaluation_airports" in values:
            values["evaluation_airports"] = tuple(values["evaluation_airports"])
        return cls(**values)

    def dictionary(self) -> dict:
        return asdict(self)

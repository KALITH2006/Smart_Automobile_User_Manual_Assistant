"""Shared types for the ADAPT-Diag adaptive retrieval pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Mode(str, Enum):
    M0 = "M0"  # exact structured DTC lookup
    M1 = "M1"  # filtered sparse (BM25) retrieval
    M2 = "M2"  # hybrid sparse + dense with rank fusion
    M3 = "M3"  # hybrid, iterative with query rewriting


@dataclass(frozen=True)
class Hit:
    """One retrievable section/chunk."""

    id: int
    page: int
    text: str
    score: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VehicleProfile:
    make: str | None = None
    model: str | None = None
    year: int | None = None
    engine: str | None = None


@dataclass(frozen=True)
class DiagnosticQuery:
    """q = (t, C, v, x) from the paper."""

    text: str = ""
    dtcs: tuple[str, ...] = ()
    vehicle: VehicleProfile | None = None
    sensors: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteDecision:
    mode: Mode
    confidence: float
    rule: str
    llm_calls: int = 0


@dataclass(frozen=True)
class AdaptiveConfig:
    """Tunable parameters (Table II of the paper)."""

    rrf_c: int = 60
    n_evidence: int = 5
    depth: int = 50  # candidates per retriever before fusion
    t_max: int = 3
    alpha: float = 0.5  # weight of relevance
    beta: float = 0.2  # weight of DTC coverage
    gamma: float = 0.3  # weight of score margin
    tau: float = 0.5  # sufficiency threshold
    # Sparse-only retrieval (M1) scores sufficiency on a different scale (query-term coverage) than
    # rank-fusion modes, so it can carry its own threshold. None -> use `tau`.
    tau_m1: float | None = None
    theta_r: float = 0.5  # router confidence threshold
    sensor_margin: float = 0.1  # fraction of the normal range a reading must exceed
    # Ablation switches
    use_router: bool = True
    use_encoder: bool = True
    allow_abstain: bool = True

    def tau_for(self, mode: "Mode") -> float:
        return self.tau_m1 if mode is Mode.M1 and self.tau_m1 is not None else self.tau


@dataclass(frozen=True)
class Step:
    round: int
    mode: Mode
    sufficiency: float
    query: str


@dataclass
class Outcome:
    abstained: bool
    sufficiency: float
    mode: Mode  # mode of the final retrieval round
    route: RouteDecision
    evidence: list[Hit]
    steps: list[Step]
    sensor_tokens: list[str]
    final_query: str
    llm_calls: int

"""Context encoder: turn vehicle profile, DTCs and sensor data into filters and symptom tokens."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .types import DiagnosticQuery

DTC_RE = re.compile(r"\b([PBCU][0-3][0-9A-F]{3})\b", re.IGNORECASE)
SYSTEM_BY_PREFIX = {"P": "powertrain", "B": "body", "C": "chassis", "U": "network"}

DTC_TABLE_PATH = Path(__file__).parent / "data" / "dtc_generic.json"


def load_dtc_table(path: Path = DTC_TABLE_PATH) -> dict[str, dict]:
    """code -> {"description": str, "system": str}"""
    return json.loads(path.read_text(encoding="utf8"))


def extract_dtcs(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in DTC_RE.finditer(text):
        seen.setdefault(m.group(1).upper(), None)
    return list(seen)


@dataclass(frozen=True)
class SensorRange:
    key: str
    low: float
    high: float
    low_token: str
    high_token: str


# Configurable normal ranges; tokens are only emitted when a reading is outside by a margin.
DEFAULT_SENSOR_RANGES: tuple[SensorRange, ...] = (
    SensorRange("ltft_pct", -10.0, 10.0, "low long-term fuel trim rich condition", "high long-term fuel trim lean condition"),
    SensorRange("stft_pct", -10.0, 10.0, "low short-term fuel trim rich condition", "high short-term fuel trim lean condition"),
    SensorRange("coolant_temp_c", 80.0, 105.0, "engine runs cold low coolant temperature", "engine overheating high coolant temperature"),
    SensorRange("battery_voltage", 12.4, 14.8, "low battery voltage", "high charging voltage"),
    SensorRange("idle_rpm", 600.0, 900.0, "low idle speed", "high idle speed"),
)


@dataclass
class EncodedContext:
    text: str  # t' = complaint text + confirmed-able symptom tokens
    filter: dict = field(default_factory=dict)  # F
    dtcs: list[str] = field(default_factory=list)
    systems: set[str] = field(default_factory=set)
    sensor_tokens: list[str] = field(default_factory=list)  # shown to the user for confirmation


def sensor_tokens(
    sensors: dict[str, float],
    ranges: tuple[SensorRange, ...] = DEFAULT_SENSOR_RANGES,
    margin: float = 0.1,
) -> list[str]:
    out: list[str] = []
    for r in ranges:
        if r.key not in sensors:
            continue
        v, slack = sensors[r.key], margin * (r.high - r.low)
        if v > r.high + slack:
            out.append(r.high_token)
        elif v < r.low - slack:
            out.append(r.low_token)
    return out


def encode(query: DiagnosticQuery, margin: float = 0.1) -> EncodedContext:
    dtcs = list(dict.fromkeys([c.upper() for c in query.dtcs] + extract_dtcs(query.text)))
    systems = {SYSTEM_BY_PREFIX[c[0]] for c in dtcs}

    flt: dict = {}
    v = query.vehicle
    if v:
        for key in ("make", "model", "year", "engine"):
            val = getattr(v, key)
            if val is not None:
                flt[key] = val
    if systems:
        flt["system"] = systems

    tokens = sensor_tokens(query.sensors, margin=margin)
    text = " ".join([query.text.strip(), *tokens]).strip()
    return EncodedContext(text=text, filter=flt, dtcs=dtcs, systems=systems, sensor_tokens=tokens)


def metadata_matches(meta: dict, flt: dict) -> bool:
    """Apply filter F to a section's metadata.

    A section that does not carry a given metadata key is kept: a single-vehicle owner's manual
    has no per-section make/model metadata, and dropping everything would make retrieval empty.
    """
    for key, want in flt.items():
        if key == "year":
            lo, hi = meta.get("year_from"), meta.get("year_to")
            if lo is not None and hi is not None and not (lo <= want <= hi):
                return False
        elif key == "system":
            have = meta.get("system")
            if have is not None and have not in want:
                return False
        else:
            have = meta.get(key)
            if have is not None and str(have).lower() != str(want).lower():
                return False
    return True

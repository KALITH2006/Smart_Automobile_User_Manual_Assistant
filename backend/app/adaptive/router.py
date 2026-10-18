"""Adaptive router (Table I): rules first, optional LLM classifier for the rest."""
from __future__ import annotations

import re
from typing import Callable, Optional

from .context import EncodedContext
from .types import DiagnosticQuery, Mode, RouteDecision

# part-name tokens -> vehicle system, used for the "identifier or part-name" and "one system" checks
PART_SYSTEMS: dict[str, str] = {
    "spark": "powertrain", "plug": "powertrain", "injector": "powertrain", "throttle": "powertrain",
    "alternator": "electrical", "battery": "electrical", "fuse": "electrical", "starter": "electrical",
    "wiper": "body", "washer": "body", "window": "body", "mirror": "body", "headlight": "lighting",
    "bulb": "lighting", "brake": "brakes", "abs": "brakes", "rotor": "brakes", "caliper": "brakes",
    "tire": "wheels", "tyre": "wheels", "wheel": "wheels", "jack": "wheels", "tpms": "wheels",
    "radiator": "cooling", "coolant": "cooling", "thermostat": "cooling", "oil": "lubrication",
    "dipstick": "lubrication", "filter": "intake", "clutch": "transmission", "gearbox": "transmission",
    "cvt": "transmission", "airbag": "safety", "seatbelt": "safety", "fuel": "fuel", "cap": "fuel",
}

IDENTIFIER_RE = re.compile(r"\b[A-Za-z]{1,3}-?\d{2,}[A-Za-z]?\b")

SYMPTOM_GROUPS: dict[str, tuple[str, ...]] = {
    "misfire": ("misfire", "rough idle", "shak", "vibrat", "judder"),
    "stall": ("stall", "dies", "cuts out", "cut out"),
    "noise": ("noise", "rattle", "knock", "squeal", "grind", "whine", "click", "hiss"),
    "smell": ("smell", "odor", "odour", "fumes"),
    "smoke": ("smoke", "steam"),
    "overheat": ("overheat", "running hot"),
    "power": ("hesitat", "lack of power", "sluggish", "surge", "loss of power"),
    "leak": ("leak", "drip", "puddle"),
    "starting": ("won't start", "wont start", "hard to start", "no start", "slow crank", "doesn't start"),
    "pull": ("pulls to", "spongy", "pulsat", "squeak"),
    "light": ("warning light", "check engine", "indicator", "telltale", "light came on", "light is on"),
}

# (mode, confidence) classifier for queries the rules cannot resolve
Classifier = Callable[[str], "tuple[Mode, float]"]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def count_symptoms(text: str) -> int:
    low = text.lower()
    return sum(any(k in low for k in keys) for keys in SYMPTOM_GROUPS.values())


def part_systems(text: str) -> set[str]:
    return {PART_SYSTEMS[w] for w in _words(text) if w in PART_SYSTEMS}


class Router:
    """Rules are evaluated in the order M0, M3, M1, then the M2 default.

    Table I of the paper tests M1 before M3. That sends a multi-symptom complaint that merely
    contains a part word ("engine shakes and there is a fuel smell") to a single-retriever BM25
    lookup, so the multi-symptom/multi-system rule is evaluated first here.
    """

    def __init__(
        self,
        dtc_table: dict[str, dict] | None = None,
        classifier: Optional[Classifier] = None,
        theta_r: float = 0.5,
        max_m1_tokens: int = 12,
    ):
        self.dtc_table = dtc_table or {}
        self.classifier = classifier
        self.theta_r = theta_r
        self.max_m1_tokens = max_m1_tokens

    def route(self, query: DiagnosticQuery, ctx: EncodedContext) -> RouteDecision:
        complaint = query.text.strip()
        dtcs = ctx.dtcs

        # M0: exactly one code, no complaint text, code present in the table
        if len(dtcs) == 1 and not _strip_codes(complaint) and dtcs[0] in self.dtc_table:
            return RouteDecision(Mode.M0, 1.0, "single DTC found in table")

        words = _words(complaint)
        parts = part_systems(complaint)
        has_identifier = bool(IDENTIFIER_RE.search(complaint)) or bool(parts)
        systems = ctx.systems | parts

        # M3: several codes across systems, or several distinct symptoms
        if (len(dtcs) >= 2 and len(ctx.systems) >= 2) or count_symptoms(complaint) >= 2:
            return RouteDecision(Mode.M3, 0.9, "multi-system codes or multiple symptoms")

        # M1: identifier / part-name tokens, a single system, short query
        if has_identifier and len(systems) <= 1 and 0 < len(words) <= self.max_m1_tokens:
            return RouteDecision(Mode.M1, 0.9, "short part/identifier query in one system")

        # M2 default; an optional LLM classifier may override and low confidence escalates to M3
        if self.classifier is not None:
            mode, conf = self.classifier(complaint)
            if conf < self.theta_r:
                return RouteDecision(Mode.M3, conf, "classifier low confidence", llm_calls=1)
            return RouteDecision(mode, conf, "classifier", llm_calls=1)
        return RouteDecision(Mode.M2, 1.0, "default free-text query")


def _strip_codes(text: str) -> str:
    from .context import DTC_RE

    return DTC_RE.sub("", text).strip(" ,.;:-")


def escalate(mode: Mode) -> Mode:
    """M1 -> M2 -> M3; M0 (no table hit) -> M2; M3 stays M3."""
    return {Mode.M0: Mode.M2, Mode.M1: Mode.M2, Mode.M2: Mode.M3, Mode.M3: Mode.M3}[mode]

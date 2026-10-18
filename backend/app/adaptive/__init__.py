"""ADAPT-Diag: adaptive, context-aware retrieval for vehicle fault diagnosis."""
from .context import encode, extract_dtcs, load_dtc_table
from .controller import run_adaptive
from .router import Router
from .searchers import InMemorySearcher, chroma_searcher
from .types import AdaptiveConfig, DiagnosticQuery, Hit, Mode, Outcome, VehicleProfile

__all__ = [
    "AdaptiveConfig", "DiagnosticQuery", "Hit", "InMemorySearcher", "Mode", "Outcome", "Router",
    "VehicleProfile", "chroma_searcher", "encode", "extract_dtcs", "load_dtc_table", "run_adaptive",
]

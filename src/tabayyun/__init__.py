"""Tabayyun-Cert: certified, corroboration-gated quarantine for poisoned
test-time memory in federated, agentic AI-native networks."""
from .certify import DirectionalDrift, action_certified, apriori_drift, half_life, persistence_bound, promotion_prob
from .gate import GateConfig, tabayyun_gate
from .memory import Gates, LinearMemory, MLPMemory, make_memory
from .tiered import TabayyunConfig, TabayyunMemory

__version__ = "0.1.0"
__all__ = [
    "Gates", "LinearMemory", "MLPMemory", "make_memory",
    "GateConfig", "tabayyun_gate",
    "TabayyunConfig", "TabayyunMemory",
    "DirectionalDrift", "action_certified", "apriori_drift", "half_life",
    "persistence_bound", "promotion_prob",
]

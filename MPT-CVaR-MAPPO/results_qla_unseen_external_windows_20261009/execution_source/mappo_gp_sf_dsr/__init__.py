"""MAPPO-GP-SF-DSR extension for risk-averse edge offloading.

This package is intentionally separate from ``repro_risk_averse_edge`` so the
paper-faithful reproduction remains available as a baseline.
"""

from .config import MAPPOConfig, MAPPOSystemConfig

__all__ = ["MAPPOConfig", "MAPPOSystemConfig"]

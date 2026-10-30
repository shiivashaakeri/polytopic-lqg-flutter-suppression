"""
Polytopic Receding-Horizon Policy Gradient (P-RHPG) synthesis.

Implementation of Algorithm 1 from the CDC 2026 paper:
"Receding-Horizon Policy Gradient for Polytopic Controller Synthesis."

Public API:
    PRHPGProblem    -- class-based interface
    prhpg_struct    -- functional API: build problem
    prhpg_solve     -- functional API: solve problem
"""

from tptool.prhpg.synthesis import (
    PRHPGProblem,
    prhpg_solve,
    prhpg_struct,
)
from tptool.prhpg.observer import (
    ObserverPRHPGProblem,
    observer_struct,
    observer_solve,
)

__all__ = [
    "PRHPGProblem",
    "prhpg_solve",
    "prhpg_struct",
    "ObserverPRHPGProblem",
    "observer_struct",
    "observer_solve",
]

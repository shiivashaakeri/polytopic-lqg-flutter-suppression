"""
LMI synthesis for TP-form polytopic LPV systems.

Continuous-time formulation following Tanaka & Wang (2002).
Requires cvxpy (install via: pip install tptool[lmi]).
"""

from tptool.lmi.synthesis import (
    LMIProblem,
    LMIObserverProblem,
    lmi_struct,
    lmi_asym,
    lmi_asym_decay,
    lmi_input,
    lmi_solve,
)

__all__ = [
    "LMIProblem",
    "lmi_asym",
    "lmi_asym_decay",
    "lmi_input",
    "lmi_solve",
    "lmi_struct",
    "LMIObserverProblem",
]

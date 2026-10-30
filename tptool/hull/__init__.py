"""
Convex hull decompositions for polytopic LPV representations.
"""

from tptool.hull.box import box_decomp
from tptool.hull.close import close_decomp, opt_decomp
from tptool.hull.cno import cno
from tptool.hull.decomp import decomp, genhull
from tptool.hull.ino import ino, rnodiff, rnoino
from tptool.hull.manip import hull_manip
from tptool.hull.snnn import snnn_basis, snnn_decomp

__all__ = [
    "box_decomp",
    "close_decomp",
    "cno",
    "decomp",
    "genhull",
    "hull_manip",
    "ino",
    "opt_decomp",
    "rnodiff",
    "rnoino",
    "snnn_basis",
    "snnn_decomp",
]

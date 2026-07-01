"""flatpack: flatten 3D backpack shells into 2D fabric panels for MYOG patterns."""

from flatpack.distortion import DistortionReport, distortion_report
from flatpack.fabric import Fabric, FabricFit, FABRICS, fabric_fit
from flatpack.flatten import FlattenResult, flatten, lscm
from flatpack.seams import Panel, SeamSpec, load_seam_spec, split_mesh

__all__ = [
    "DistortionReport",
    "distortion_report",
    "Fabric",
    "FabricFit",
    "FABRICS",
    "fabric_fit",
    "FlattenResult",
    "flatten",
    "lscm",
    "Panel",
    "SeamSpec",
    "load_seam_spec",
    "split_mesh",
]

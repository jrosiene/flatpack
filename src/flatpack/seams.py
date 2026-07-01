"""Seam definition: cut a shell mesh into panels along user-chosen edge paths.

Seams are defined manually in a small YAML file (automatic seam-finding is
out of scope for now). A seam is a path of vertex indices; every
consecutive pair must be an actual mesh edge. Cutting removes the
face-to-face connection across those edges, and the resulting connected
components of the face graph become panels.

Example seam file:

    units: mm
    seam_allowance: 10
    seams:
      - name: side
        path: [3, 18, 33, 48]
    panels:
      - name: front
        anchor_face: 12          # any face index inside this panel
        fabric: ultrastretch
        stretch_axis_deg: 0      # stretch axis, measured from the grainline
        grain: [3, 48]           # vertex pair defining the grainline
        notches: [18]            # boundary vertices to mark with notches

Vertex and face indices refer to the *original* mesh; each Panel keeps the
mapping back to original indices so notch/grain references stay valid
after splitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import trimesh
import yaml

from flatpack.meshutil import unique_edges


@dataclass
class PanelSpec:
    """User-provided metadata for one panel (all fields optional but name)."""

    name: str
    anchor_face: int | None = None
    fabric: str = "rigid"
    stretch_axis_deg: float = 0.0
    grain: tuple[int, int] | None = None  # original vertex indices
    notches: list[int] = field(default_factory=list)  # original vertex indices


@dataclass
class SeamSpec:
    """Parsed seam file."""

    seams: list[list[int]]
    panels: list[PanelSpec]
    units: str = "mm"
    seam_allowance: float = 10.0


@dataclass
class Panel:
    """A connected patch cut out of the shell, ready to flatten."""

    name: str
    vertices: np.ndarray  # (n, 3), panel-local
    faces: np.ndarray  # (t, 3), panel-local indices
    orig_vertex_index: np.ndarray  # panel-local -> original mesh vertex index
    spec: PanelSpec

    def local_index(self, orig_vertex: int) -> int:
        """Translate an original-mesh vertex index to this panel's indexing."""
        matches = np.nonzero(self.orig_vertex_index == orig_vertex)[0]
        if len(matches) == 0:
            raise KeyError(
                f"vertex {orig_vertex} is not part of panel {self.name!r}"
            )
        return int(matches[0])


def load_seam_spec(path: str | Path) -> SeamSpec:
    return spec_from_dict(yaml.safe_load(Path(path).read_text()))


def spec_from_dict(data: dict) -> SeamSpec:
    """Build a SeamSpec from parsed YAML / JSON data (see module docstring)."""
    seams = [list(map(int, seam["path"])) for seam in data.get("seams", [])]
    panels = [
        PanelSpec(
            name=str(p["name"]),
            anchor_face=int(p["anchor_face"]) if "anchor_face" in p else None,
            fabric=str(p.get("fabric", "rigid")),
            stretch_axis_deg=float(p.get("stretch_axis_deg", 0.0)),
            grain=tuple(map(int, p["grain"])) if "grain" in p else None,
            notches=[int(v) for v in p.get("notches", [])],
        )
        for p in data.get("panels", [])
    ]
    return SeamSpec(
        seams=seams,
        panels=panels,
        units=str(data.get("units", "mm")),
        seam_allowance=float(data.get("seam_allowance", 10.0)),
    )


def face_labels(mesh: trimesh.Trimesh, seams: list[list[int]]) -> tuple[int, np.ndarray]:
    """Connected-component label per face after cutting along the seams.

    Returns (number of components, per-face label array). This is the
    split preview: the GUI colours faces by label before committing.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    seam_edges = _seam_edge_set(faces, seams)

    # Face adjacency graph, minus adjacency across seam edges.
    adjacency = np.asarray(mesh.face_adjacency)  # (k, 2) face index pairs
    shared = np.sort(np.asarray(mesh.face_adjacency_edges), axis=1)  # (k, 2) vertices
    keep = np.array(
        [tuple(edge) not in seam_edges for edge in shared], dtype=bool
    )
    kept = adjacency[keep]

    graph = scipy.sparse.coo_matrix(
        (np.ones(len(kept)), (kept[:, 0], kept[:, 1])),
        shape=(len(faces), len(faces)),
    )
    return scipy.sparse.csgraph.connected_components(graph, directed=False)


def split_mesh(mesh: trimesh.Trimesh, spec: SeamSpec) -> list[Panel]:
    """Cut the mesh along the seam paths and return one Panel per component.

    Components containing a panel spec's anchor_face get that spec; others
    get an auto-generated name and default metadata.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)

    n_components, labels = face_labels(mesh, spec.seams)

    panels = []
    for component in range(n_components):
        face_subset = faces[labels == component]
        orig_vertex_index = np.unique(face_subset)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[orig_vertex_index] = np.arange(len(orig_vertex_index))

        panel_spec = _spec_for_component(spec, labels, component)
        panels.append(
            Panel(
                name=panel_spec.name,
                vertices=vertices[orig_vertex_index],
                faces=remap[face_subset],
                orig_vertex_index=orig_vertex_index,
                spec=panel_spec,
            )
        )
    panels.sort(key=lambda p: p.name)
    return panels


def _seam_edge_set(faces: np.ndarray, seams: list[list[int]]) -> set[tuple[int, int]]:
    """Validate seam paths and collect their edges as sorted vertex pairs."""
    mesh_edges = {tuple(edge) for edge in unique_edges(faces)}
    seam_edges: set[tuple[int, int]] = set()
    for seam_index, path in enumerate(seams):
        if len(path) < 2:
            raise ValueError(f"seam {seam_index} needs at least two vertices")
        for a, b in zip(path[:-1], path[1:]):
            edge = (min(a, b), max(a, b))
            if edge not in mesh_edges:
                raise ValueError(
                    f"seam {seam_index}: vertices {a} and {b} are consecutive in "
                    "the seam path but are not connected by a mesh edge"
                )
            seam_edges.add(edge)
    return seam_edges


def _spec_for_component(
    spec: SeamSpec, labels: np.ndarray, component: int
) -> PanelSpec:
    matching = [
        p
        for p in spec.panels
        if p.anchor_face is not None and labels[p.anchor_face] == component
    ]
    if len(matching) > 1:
        names = ", ".join(p.name for p in matching)
        raise ValueError(
            f"panels {names} have anchor faces in the same component; "
            "check your seam paths actually separate them"
        )
    if matching:
        return matching[0]
    return PanelSpec(name=f"panel_{component}")

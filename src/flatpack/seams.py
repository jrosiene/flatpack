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

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import trimesh
import yaml

from flatpack.meshutil import boundary_edges, unique_edges


@dataclass
class PanelSpec:
    """User-provided metadata for one panel (all fields optional but name)."""

    name: str
    anchor_face: int | None = None
    fabric: str = "rigid"
    stretch_axis_deg: float = 0.0
    grain: tuple[int, int] | None = None  # original vertex indices
    notches: list[int] = field(default_factory=list)  # original vertex indices


@dataclass(frozen=True)
class Mark:
    """A sewing annotation anchored to a mesh vertex.

    type: "bartack" (drawn as a thick bar, oriented toward `toward` if
    given) or "attach" (a circle-cross target for webbing, buckles,
    zipper stops, ...). The label is printed next to the symbol.
    """

    vertex: int
    type: str = "attach"
    label: str = ""
    toward: int | None = None


@dataclass
class SeamSpec:
    """Parsed seam file."""

    seams: list[list[int]]
    panels: list[PanelSpec]
    units: str = "mm"
    seam_allowance: float = 10.0
    # Darts are slit seams: a vertex path from a boundary vertex (the dart
    # mouth) inward to the apex. Opening the slit lets the flattening
    # spread it into a V; the V is the dart intake. A path with both ends
    # interior gives a fisheye dart.
    darts: list[list[int]] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)
    # Print each straight boundary edge's length on the pattern:
    # "none", "cm" or "in".
    edge_labels: str = "none"
    # Auto registration ticks along seams to help line up mating panels.
    seam_markers: bool = True


@dataclass
class Panel:
    """A connected patch cut out of the shell, ready to flatten."""

    name: str
    vertices: np.ndarray  # (n, 3), panel-local
    faces: np.ndarray  # (t, 3), panel-local indices
    orig_vertex_index: np.ndarray  # panel-local -> original mesh vertex index
    spec: PanelSpec
    darts: list[list[int]] = field(default_factory=list)  # original indices
    marks: list[Mark] = field(default_factory=list)
    # Panel-local boundary edges that are true seams (get sewn to another
    # panel), as sorted (a, b) pairs — used to place alignment ticks.
    seam_edges: set = field(default_factory=set)

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
    darts = [list(map(int, dart["path"])) for dart in data.get("darts", [])]
    marks = [
        Mark(
            vertex=int(m["vertex"]),
            type=str(m.get("type", "attach")),
            label=str(m.get("label", "")),
            toward=int(m["toward"]) if m.get("toward") is not None else None,
        )
        for m in data.get("marks", [])
    ]
    edge_labels = str(data.get("edge_labels", "none"))
    if edge_labels not in ("none", "cm", "in"):
        raise ValueError(
            f"edge_labels must be 'none', 'cm' or 'in', not {edge_labels!r}"
        )
    return SeamSpec(
        seams=seams,
        panels=panels,
        units=str(data.get("units", "mm")),
        seam_allowance=float(data.get("seam_allowance", 10.0)),
        darts=darts,
        marks=marks,
        edge_labels=edge_labels,
        seam_markers=bool(data.get("seam_markers", True)),
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

    The seams are genuinely *opened*: vertices along them are duplicated so
    each side keeps its own copy and the seam becomes boundary. This matters
    for seams that do not separate the surface — a single seam up the side
    of a tube-shaped pack body yields one panel that unrolls flat, instead
    of a panel that is still topologically a closed tube.

    Components containing a panel spec's anchor_face get that spec; others
    get an auto-generated name and default metadata.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    # Darts are slits: they get opened exactly like seams, but they never
    # separate components (they end inside the surface).
    darts = _normalized_darts(faces, spec)
    seam_edges = _seam_edge_set(faces, spec.seams + darts)
    # Only real panel-joining seams (not darts) carry alignment ticks.
    join_edges = _seam_edge_set(faces, spec.seams)

    vertices, faces, orig_map = open_seams(mesh, seam_edges)

    n_components, labels = face_labels(mesh, spec.seams)

    panels = []
    for component in range(n_components):
        face_subset = faces[labels == component]
        vertex_subset = np.unique(face_subset)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[vertex_subset] = np.arange(len(vertex_subset))
        local_faces = remap[face_subset]

        panel_spec = _spec_for_component(spec, labels, component)
        panel_origs = set(orig_map[vertex_subset].tolist())
        local_orig = orig_map[vertex_subset]
        panel_seam_edges = {
            edge
            for edge in boundary_edges(local_faces)
            if (
                min(int(local_orig[edge[0]]), int(local_orig[edge[1]])),
                max(int(local_orig[edge[0]]), int(local_orig[edge[1]])),
            )
            in join_edges
        }
        panels.append(
            Panel(
                name=panel_spec.name,
                vertices=vertices[vertex_subset],
                faces=local_faces,
                orig_vertex_index=local_orig,
                spec=panel_spec,
                darts=[d for d in darts if set(d) <= panel_origs],
                marks=[m for m in spec.marks if m.vertex in panel_origs],
                seam_edges=panel_seam_edges,
            )
        )
    panels.sort(key=lambda p: p.name)
    return panels


def open_seams(
    mesh: trimesh.Trimesh, seam_edges: set[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate vertices along seam edges so the surface is actually cut.

    Around each vertex touched by a seam, the incident faces are grouped
    into wedges separated by seam edges; every wedge beyond the first gets
    its own copy of the vertex. After this, faces on opposite sides of a
    seam share no vertices, so the seam is real boundary. Seam endpoints
    interior to the surface keep a single vertex (the cut just stops
    there, like scissors partway into a sheet).

    Returns (vertices, faces, orig_map) where orig_map[i] is the original
    index each (possibly duplicated) vertex came from. Original vertices
    keep their indices; copies are appended.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    orig_map = list(range(len(vertices)))
    new_positions: list[np.ndarray] = []

    incident: dict[int, list[int]] = defaultdict(list)
    for fi, face in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
        for v in face:
            incident[int(v)].append(fi)

    seam_vertices = sorted({v for edge in seam_edges for v in edge})
    for v in seam_vertices:
        wedges = _wedges_around(v, incident[v], faces, orig_map, seam_edges)
        for wedge in wedges[1:]:
            copy = len(vertices) + len(new_positions)
            new_positions.append(vertices[v])
            orig_map.append(v)
            for fi in wedge:
                faces[fi][faces[fi] == v] = copy

    if new_positions:
        vertices = np.vstack([vertices, new_positions])
    return vertices, faces, np.array(orig_map, dtype=np.int64)


def _wedges_around(
    v: int,
    face_indices: list[int],
    faces: np.ndarray,
    orig_map: list[int],
    seam_edges: set[tuple[int, int]],
) -> list[list[int]]:
    """Group the faces around vertex v into components separated by seams.

    Two faces are in the same wedge when they share a non-seam edge at v.
    Edges are matched by current vertex ids (duplicates made at other seam
    vertices split the fan exactly where they should) but tested against
    the seam set by original ids.
    """
    by_edge: dict[int, list[int]] = defaultdict(list)
    for fi in face_indices:
        for u in faces[fi]:
            u = int(u)
            if u == v:
                continue
            a, b = sorted((v, orig_map[u]))
            if (a, b) in seam_edges:
                continue
            by_edge[u].append(fi)

    neighbours: dict[int, set[int]] = defaultdict(set)
    for shared in by_edge.values():
        for a in shared:
            for b in shared:
                neighbours[a].add(b)

    wedges = []
    remaining = set(face_indices)
    while remaining:
        seed = remaining.pop()
        wedge = [seed]
        queue = [seed]
        while queue:
            for other in neighbours[queue.pop()]:
                if other in remaining:
                    remaining.remove(other)
                    wedge.append(other)
                    queue.append(other)
        wedges.append(wedge)
    return wedges


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


def _normalized_darts(faces: np.ndarray, spec: SeamSpec) -> list[list[int]]:
    """Validate darts and orient every path mouth-first, apex-last.

    The mouth must sit on a boundary or seam vertex: a fully interior
    slit would turn the panel into an annulus, which a conformal map
    cannot flatten — reject it with a useful message instead of failing
    later with a topology error.
    """
    if not spec.darts:
        return []
    from flatpack.meshutil import boundary_loops

    reachable = {int(v) for loop in boundary_loops(faces) for v in loop}
    for path in spec.seams:
        reachable.update(int(v) for v in path)

    darts = []
    for i, dart in enumerate(spec.darts):
        if dart[0] in reachable:
            darts.append(list(dart))
        elif dart[-1] in reachable:
            darts.append(list(dart)[::-1])
        else:
            raise ValueError(
                f"dart {i} must start at a boundary or seam vertex; a fully "
                "interior (fisheye) dart cannot be flattened - extend it to "
                "a boundary or run a seam through it"
            )
    return darts


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

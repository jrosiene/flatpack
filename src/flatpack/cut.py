"""Cut a mesh across faces, for seams that don't follow existing edges.

Seam paths normally follow mesh edges (seams.py), which makes a
"diagonal" seam staircase across the triangulation. cut_between() slices
the surface instead: it takes two vertices, puts a plane through them
(oriented along the local surface normal, so the cut runs across the
surface rather than through it), finds where that plane crosses the
mesh between the two points, inserts vertices on the crossed edges, and
retriangulates the crossed faces.

The result is a new mesh in which the cut is an ordinary chain of edges,
so seam splitting and flattening work on it unchanged. Existing vertex
indices are preserved (new vertices are appended), which keeps previously
drawn seams, notches and grainlines valid; face indices do change.

Intersections closer than `snap` (fraction of edge length) to an existing
vertex snap to it, avoiding sliver triangles.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import trimesh

# A node on the cut curve: ("v", vertex_index) for an existing vertex,
# ("e", (a, b)) for a point on the edge between vertices a < b.
Node = tuple


@dataclass
class PlaneCutResult:
    mesh: trimesh.Trimesh
    path: list[int]  # vertex chain along the cut (indices into .mesh)


def insert_vertex_on_edge(
    mesh: trimesh.Trimesh, face_index: int, point: np.ndarray, snap: float = 0.1
) -> tuple[trimesh.Trimesh, int, tuple[int, int] | None]:
    """Insert a vertex on the mesh edge nearest to `point` on the given face.

    For when there is simply no vertex where a seam needs to start or end.
    The point is projected onto the closest of the face's three edges;
    both faces sharing that edge are split so the new vertex is a proper
    mesh vertex usable in seam paths. If the projection lands within
    `snap` (fraction of edge length) of an existing vertex, that vertex is
    returned instead and the mesh is unchanged.

    Returns (mesh, vertex_index, split_edge); existing vertex indices are
    preserved. split_edge is the (a, b) edge that was subdivided, or None
    if the point snapped to an existing vertex (mesh unchanged) — it lets
    the caller record the edit for later replay.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not 0 <= face_index < len(faces):
        raise ValueError(f"face index {face_index} out of range")
    point = np.asarray(point, dtype=float)

    v0, v1, v2 = (int(v) for v in faces[face_index])
    best = None
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        edge = vertices[b] - vertices[a]
        t = float(np.clip((point - vertices[a]) @ edge / (edge @ edge), 0.0, 1.0))
        distance = float(np.linalg.norm(vertices[a] + t * edge - point))
        if best is None or distance < best[0]:
            best = (distance, a, b, t)
    _, a, b, t = best

    if t < snap:
        return mesh, a, None
    if t > 1.0 - snap:
        return mesh, b, None

    new_index = len(vertices)
    new_position = vertices[a] + t * (vertices[b] - vertices[a])
    all_vertices = np.vstack([vertices, [new_position]])

    edge = (min(a, b), max(a, b))
    split = {edge: new_index}
    face_list: list = []
    for face in faces:
        if _face_split_points(face, split):
            face_list.extend(_split_face(face, split, all_vertices))
        else:
            face_list.append(tuple(int(v) for v in face))

    return (
        trimesh.Trimesh(vertices=all_vertices, faces=np.array(face_list), process=False),
        new_index,
        edge,
    )


def replay_edits(mesh: trimesh.Trimesh, edits: list[dict]) -> trimesh.Trimesh:
    """Re-apply a recorded list of mesh edits, in order, to a fresh mesh.

    Lets a saved seams file rebuild the exact working mesh (added vertices,
    cuts, scaling) on top of the original OBJ, so seam indices line up
    again. Each edit is one of:

      {"op": "scale", "factor": f}
      {"op": "cut", "start": i, "end": j}
      {"op": "add_vertex", "edge": [a, b], "point": [x, y, z]}

    The operations are deterministic, so replaying them reproduces the same
    vertex numbering the original session produced.
    """
    for edit in edits:
        mesh = _apply_edit(mesh, edit)
    return mesh


def _apply_edit(mesh: trimesh.Trimesh, edit: dict) -> trimesh.Trimesh:
    op = edit.get("op")
    if op == "scale":
        factor = float(edit["factor"])
        return trimesh.Trimesh(
            np.asarray(mesh.vertices, dtype=float) * factor, mesh.faces, process=False
        )
    if op == "cut":
        return cut_between(mesh, int(edit["start"]), int(edit["end"])).mesh
    if op == "add_vertex":
        a, b = (int(v) for v in edit["edge"])
        faces = np.asarray(mesh.faces, dtype=np.int64)
        face = next(
            (i for i, f in enumerate(faces) if a in f and b in f), None
        )
        if face is None:
            raise ValueError(
                f"recorded edit references edge ({a}, {b}), which is not in "
                "the mesh - the base OBJ does not match this seam file"
            )
        new_mesh, _, _ = insert_vertex_on_edge(
            mesh, face, np.asarray(edit["point"], dtype=float)
        )
        return new_mesh
    raise ValueError(f"unknown mesh edit op {op!r}")


def cut_between(
    mesh: trimesh.Trimesh, start: int, end: int, snap: float = 0.05
) -> PlaneCutResult:
    """Cut the surface along the plane through vertices start and end."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if start == end:
        raise ValueError("cut needs two different vertices")

    d = _plane_distances(mesh, vertices, start, end)

    node_pos, edge_node = _node_helpers(vertices, d, snap)
    graph = _build_cut_graph(faces, d, node_pos, edge_node)
    path_nodes = _shortest_cut(graph, ("v", start), ("v", end))

    return _apply_cut(vertices, faces, path_nodes, node_pos)


def _plane_distances(
    mesh: trimesh.Trimesh, vertices: np.ndarray, start: int, end: int
) -> np.ndarray:
    """Signed distance of every vertex to the cutting plane.

    The plane contains both cut points and the average of their surface
    normals, so it slices across the shell rather than skimming along it.
    """
    chord = vertices[end] - vertices[start]
    scale = np.linalg.norm(chord)
    if scale < 1e-12:
        raise ValueError("cut vertices coincide")
    normals = np.asarray(mesh.vertex_normals)
    plane_normal = np.cross(chord, normals[start] + normals[end])
    norm = np.linalg.norm(plane_normal)
    if norm < 1e-9 * scale:
        raise ValueError(
            "cannot orient the cutting plane (cut direction is parallel "
            "to the surface normal); pick different points"
        )
    d = (vertices - vertices[start]) @ (plane_normal / norm)
    d[np.abs(d) < 1e-9 * scale] = 0.0
    d[start] = 0.0
    d[end] = 0.0
    return d


def _node_helpers(vertices: np.ndarray, d: np.ndarray, snap: float):
    """Position lookup and (cached, snap-aware) edge-crossing node factory."""
    cache: dict[tuple[int, int], Node] = {}

    def crossing_t(key: tuple[int, int]) -> float:
        da, db = d[key[0]], d[key[1]]
        return da / (da - db)

    def node_pos(node: Node) -> np.ndarray:
        if node[0] == "v":
            return vertices[node[1]]
        a, b = node[1]
        return vertices[a] + crossing_t(node[1]) * (vertices[b] - vertices[a])

    def edge_node(a: int, b: int) -> Node:
        key = (min(a, b), max(a, b))
        if key not in cache:
            t = crossing_t(key)
            if t < snap:
                cache[key] = ("v", key[0])
            elif t > 1.0 - snap:
                cache[key] = ("v", key[1])
            else:
                cache[key] = ("e", key)
        return cache[key]

    return node_pos, edge_node


def _build_cut_graph(faces, d, node_pos, edge_node):
    """Per-face plane-intersection segments, as a graph over cut nodes."""
    graph: dict[Node, list] = defaultdict(list)

    for v0, v1, v2 in faces:
        nodes = [("v", int(v)) for v in (v0, v1, v2) if d[v] == 0.0]
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            if d[a] * d[b] < 0.0:
                nodes.append(edge_node(int(a), int(b)))
        unique = list(dict.fromkeys(nodes))
        if len(unique) < 2:
            continue  # plane only grazes a corner of this face
        if len(unique) > 2:
            raise ValueError(
                "the cutting plane lies flat on a face; move the cut "
                "points slightly"
            )
        p, q = unique
        length = float(np.linalg.norm(node_pos(p) - node_pos(q)))
        graph[p].append((q, length))
        graph[q].append((p, length))
    return graph


def _shortest_cut(graph, start_node: Node, end_node: Node) -> list[Node]:
    """Dijkstra along the intersection curve from start to end."""
    best = {start_node: 0.0}
    previous: dict[Node, Node] = {}
    queue = [(0.0, start_node)]
    while queue:
        dist, node = heapq.heappop(queue)
        if node == end_node:
            break
        if dist > best.get(node, np.inf):
            continue
        for other, length in graph.get(node, ()):
            candidate = dist + length
            if candidate < best.get(other, np.inf):
                best[other] = candidate
                previous[other] = node
                heapq.heappush(queue, (candidate, other))
    if end_node not in previous and start_node != end_node:
        raise ValueError(
            "the cutting plane does not connect the two points across the "
            "surface; pick closer points or cut in several legs"
        )

    path = [end_node]
    while path[-1] != start_node:
        path.append(previous[path[-1]])
    return path[::-1]


def _apply_cut(vertices, faces, path_nodes, node_pos) -> PlaneCutResult:
    """Insert the path's new vertices and retriangulate around them.

    Every face touching a split edge is rebuilt — not just the faces the
    path runs through — so no face is ever left with a vertex sitting in
    the middle of one of its edges (a T-junction).
    """
    new_index: dict[tuple[int, int], int] = {}
    new_points = []
    for node in path_nodes:
        if node[0] == "e":
            new_index[node[1]] = len(vertices) + len(new_points)
            new_points.append(node_pos(node))

    all_vertices = (
        np.vstack([vertices, new_points]) if new_points else vertices.copy()
    )

    face_list = []
    for face in faces:
        split = _face_split_points(face, new_index)
        if split:
            face_list.extend(_split_face(face, split, all_vertices))
        else:
            face_list.append(tuple(int(v) for v in face))

    path = [
        node[1] if node[0] == "v" else new_index[node[1]] for node in path_nodes
    ]
    return PlaneCutResult(
        mesh=trimesh.Trimesh(
            vertices=all_vertices, faces=np.array(face_list), process=False
        ),
        path=path,
    )


def _face_split_points(face, new_index) -> dict[tuple[int, int], int]:
    """This face's split edges: {edge key: inserted vertex index}."""
    v0, v1, v2 = (int(v) for v in face)
    points = {}
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        key = (min(a, b), max(a, b))
        if key in new_index:
            points[key] = new_index[key]
    return points


def _split_face(face, split: dict, all_vertices) -> list[tuple[int, int, int]]:
    """Retriangulate a triangle whose edges carry 1 or 2 inserted vertices.

    One split edge: fan from the inserted vertex to the opposite corner.
    Two split edges (they always share a corner): three triangles. Winding
    is enforced afterwards against the original face normal, which is
    simpler than case-by-case bookkeeping.
    """
    v0, v1, v2 = (int(v) for v in face)

    def at(a, b):
        return split.get((min(a, b), max(a, b)))

    if len(split) == 1:
        # Rotate so the split edge is (a, b) and c is the opposite corner.
        for a, b, c in ((v0, v1, v2), (v1, v2, v0), (v2, v0, v1)):
            m = at(a, b)
            if m is not None:
                tris = [(a, m, c), (m, b, c)]
                break
    else:
        # The two split edges share a corner; rotate it into `shared`.
        for shared, ea, eb in ((v0, v1, v2), (v1, v2, v0), (v2, v0, v1)):
            pa, pb = at(shared, ea), at(shared, eb)
            if pa is not None and pb is not None:
                tris = [(pa, shared, pb), (ea, pa, pb), (ea, pb, eb)]
                break
        else:
            raise AssertionError("a triangle cannot have opposite split edges")

    normal = np.cross(
        all_vertices[v1] - all_vertices[v0], all_vertices[v2] - all_vertices[v0]
    )
    fixed = []
    for a, b, c in tris:
        n = np.cross(
            all_vertices[b] - all_vertices[a], all_vertices[c] - all_vertices[a]
        )
        fixed.append((a, b, c) if n @ normal >= 0 else (a, c, b))
    return fixed

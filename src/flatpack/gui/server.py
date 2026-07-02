"""Local HTTP server behind the seam-drawing GUI.

`flatpack gui shell.obj` starts this server and opens a browser. The page
(static/index.html + app.js, three.js vendored) renders the mesh and talks
to a small JSON API:

    GET  /api/mesh                vertices + faces of the loaded mesh
    POST /api/path                shortest edge path between two vertices
                                  {"start": int, "end": int} -> {"path": [...]}
    POST /api/cut                 straight cut across faces between two
                                  vertices (inserts vertices, retriangulates);
                                  {"start": int, "end": int} -> {"path": [...],
                                  "mesh": {...}} with the updated mesh
    POST /api/add_vertex          insert a vertex on the edge nearest the
                                  clicked point; {"face": int, "point": [x,y,z]}
                                  -> {"vertex": int, "mesh": {...}}
    POST /api/path_to_boundary    shortest edge path from a vertex to the
                                  nearest mesh boundary; {"start": int} ->
                                  {"path": [...]}
    POST /api/scale               scale the whole mesh (fix unit problems)
                                  {"factor": float} -> {"mesh": {...}}
    POST /api/reset               restore the mesh as originally loaded
    POST /api/split               preview panel components for seam paths
                                  {"seams": [[...], ...]} -> {"labels": [...], ...}
    POST /api/analyze             flag panels that warp beyond their fabric's
                                  stretch tolerance; {spec dict} ->
                                  {"panels": [{name, severity, advice,
                                  worst_point_3d, ...}]}
    POST /api/generate            run the full pipeline, return the report
                                  {spec dict} -> {"report": ..., "files": [...]}
    POST /api/save                write seams.yaml next to the output (plus
                                  the cut mesh as shell_cut.obj if it was cut)
    GET  /files/<name>            download generated pattern files

Everything mesh-related is done server-side with the same code paths as
the CLI, so what you see in the GUI is exactly what `flatpack flatten`
would produce.

The server is plain stdlib http.server: no extra dependencies, and easy
to drive from tests.
"""

from __future__ import annotations

import json
import webbrowser
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import trimesh
import yaml

from flatpack.analysis import analyze
from flatpack.cut import cut_between, insert_vertex_on_edge
from flatpack.meshutil import boundary_loops, unique_edges
from flatpack.pipeline import process
from flatpack.seams import face_labels, spec_from_dict

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class GuiState:
    """The mesh being edited plus everything derived from it."""

    mesh: trimesh.Trimesh
    outdir: Path
    mesh_name: str = "mesh"
    modified: bool = False  # True once a cut changed the geometry
    _original: trimesh.Trimesh | None = field(default=None, repr=False)
    _edge_graph: scipy.sparse.csr_matrix | None = field(default=None, repr=False)

    @classmethod
    def from_file(cls, mesh_path: str | Path, outdir: str | Path) -> "GuiState":
        # process=False: seam vertex indices must match the file (see seams.py).
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        return cls(
            mesh=mesh, outdir=Path(outdir), mesh_name=Path(mesh_path).stem
        )

    @property
    def edge_graph(self) -> scipy.sparse.csr_matrix:
        """Vertex adjacency weighted by edge length, for seam path-finding."""
        if self._edge_graph is None:
            edges = unique_edges(np.asarray(self.mesh.faces, dtype=np.int64))
            v = np.asarray(self.mesh.vertices, dtype=float)
            lengths = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
            n = len(v)
            self._edge_graph = scipy.sparse.coo_matrix(
                (lengths, (edges[:, 0], edges[:, 1])), shape=(n, n)
            ).tocsr()
        return self._edge_graph

    # ------------------------------------------------------------------
    # API operations (all take/return plain JSON-serialisable data)
    # ------------------------------------------------------------------

    def mesh_payload(self) -> dict:
        return {
            "name": self.mesh_name,
            "vertices": np.asarray(self.mesh.vertices, dtype=float).ravel().tolist(),
            "faces": np.asarray(self.mesh.faces, dtype=np.int64).ravel().tolist(),
        }

    def shortest_path(self, start: int, end: int) -> list[int]:
        """Vertex path from start to end along mesh edges (Dijkstra).

        This is what makes seam drawing usable: the user clicks a few
        waypoints and the seam follows the surface between them.
        """
        n = self.edge_graph.shape[0]
        if not (0 <= start < n and 0 <= end < n):
            raise ValueError(f"vertex index out of range (mesh has {n} vertices)")
        _, predecessors = scipy.sparse.csgraph.dijkstra(
            self.edge_graph,
            directed=False,
            indices=start,
            return_predecessors=True,
        )
        if predecessors[end] < 0 and end != start:
            raise ValueError("vertices are not connected on the mesh")
        path = [end]
        while path[-1] != start:
            path.append(int(predecessors[path[-1]]))
        return path[::-1]

    def cut(self, start: int, end: int) -> dict:
        """Cut straight across faces between two vertices (diagonal seam).

        Mutates the mesh: new vertices are appended (existing indices stay
        valid) and crossed faces are retriangulated. Returns the cut path
        plus the updated mesh for the client to reload.
        """
        if self._original is None:
            self._original = self.mesh.copy()
        result = cut_between(self.mesh, start, end)
        self.mesh = result.mesh
        self.modified = True
        self._edge_graph = None
        return {"path": result.path, "mesh": self.mesh_payload()}

    def scale(self, factor: float) -> dict:
        """Uniformly scale the mesh, e.g. x25.4 for a shell modelled in
        inches or x1000 for metres. Everything downstream expects mm.

        Indices are untouched, so existing seams/notches/grainlines stay
        valid; the client just reloads positions.
        """
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("scale factor must be a positive number")
        if self._original is None:
            self._original = self.mesh.copy()
        self.mesh = trimesh.Trimesh(
            np.asarray(self.mesh.vertices, dtype=float) * factor,
            self.mesh.faces,
            process=False,
        )
        self.modified = True
        self._edge_graph = None
        return {"mesh": self.mesh_payload()}

    def reset(self) -> dict:
        """Undo all cuts: restore the mesh as loaded."""
        if self._original is not None:
            self.mesh = self._original.copy()
        self.modified = False
        self._edge_graph = None
        return {"mesh": self.mesh_payload()}

    def add_vertex(self, face: int, point: list[float]) -> dict:
        """Insert a vertex where the user clicked (snapped to the nearest
        edge), for seams that need to end where no vertex exists."""
        if len(point) != 3:
            raise ValueError("point must be [x, y, z]")
        if self._original is None:
            self._original = self.mesh.copy()
        mesh, vertex = insert_vertex_on_edge(self.mesh, face, np.asarray(point))
        if mesh is not self.mesh:
            self.mesh = mesh
            self.modified = True
            self._edge_graph = None
        return {"vertex": int(vertex), "mesh": self.mesh_payload()}

    def path_to_boundary(self, start: int) -> list[int]:
        """Shortest edge path from a vertex to the nearest boundary vertex."""
        n = self.edge_graph.shape[0]
        if not 0 <= start < n:
            raise ValueError(f"vertex index out of range (mesh has {n} vertices)")
        loops = boundary_loops(np.asarray(self.mesh.faces, dtype=np.int64))
        if not loops:
            raise ValueError("mesh has no boundary (it is a closed surface)")
        border = {int(v) for loop in loops for v in loop}
        if start in border:
            return [start]
        distances, predecessors = scipy.sparse.csgraph.dijkstra(
            self.edge_graph,
            directed=False,
            indices=start,
            return_predecessors=True,
        )
        reachable = [v for v in border if np.isfinite(distances[v])]
        if not reachable:
            raise ValueError("no boundary vertex is reachable from here")
        end = min(reachable, key=lambda v: distances[v])
        path = [end]
        while path[-1] != start:
            path.append(int(predecessors[path[-1]]))
        return path[::-1]

    def split_preview(self, seams: list[list[int]]) -> dict:
        n_components, labels = face_labels(self.mesh, seams)
        return {"n_panels": int(n_components), "labels": labels.tolist()}

    def analyze_warp(self, spec_data: dict) -> dict:
        """Grade each panel's flattenability against its fabric."""
        spec = spec_from_dict(spec_data)
        return {"panels": [a.as_dict() for a in analyze(self.mesh, spec)]}

    def generate(self, spec_data: dict) -> dict:
        spec = spec_from_dict(spec_data)
        self.outdir.mkdir(parents=True, exist_ok=True)
        results = process(self.mesh, spec, self.outdir)
        report = json.loads((self.outdir / "report.json").read_text())
        files = sorted(
            p.name for p in self.outdir.iterdir() if p.suffix in (".svg", ".dxf", ".json")
        )
        return {
            "report": report,
            "files": files,
            "panels": [r.panel.name for r in results],
        }

    def save_spec(self, spec_data: dict) -> dict:
        spec_from_dict(spec_data)  # validate before writing
        self.outdir.mkdir(parents=True, exist_ok=True)
        path = self.outdir / "seams.yaml"
        path.write_text(yaml.safe_dump(spec_data, sort_keys=False))
        saved = {"saved": str(path)}
        if self.modified:
            # Cuts changed the geometry, so the seam file only makes sense
            # against the cut mesh: save it alongside for CLI replay.
            mesh_path = self.outdir / "shell_cut.obj"
            self.mesh.export(str(mesh_path))
            saved["mesh"] = str(mesh_path)
        return saved


class GuiRequestHandler(SimpleHTTPRequestHandler):
    """Static files from static/, JSON API under /api, downloads under /files."""

    def __init__(self, *args, state: GuiState, **kwargs):
        self.state = state
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass  # keep the terminal quiet; errors still surface as responses

    def do_GET(self):
        if self.path == "/api/mesh":
            self._send_json(self.state.mesh_payload())
        elif self.path.startswith("/files/"):
            # Strip any cache-busting query string before the file lookup.
            self._send_file(self.path.removeprefix("/files/").split("?")[0])
        else:
            super().do_GET()  # static assets

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or "{}")
            if self.path == "/api/path":
                payload = {
                    "path": self.state.shortest_path(
                        int(body["start"]), int(body["end"])
                    )
                }
            elif self.path == "/api/cut":
                payload = self.state.cut(int(body["start"]), int(body["end"]))
            elif self.path == "/api/scale":
                payload = self.state.scale(float(body["factor"]))
            elif self.path == "/api/add_vertex":
                payload = self.state.add_vertex(
                    int(body["face"]), list(body["point"])
                )
            elif self.path == "/api/path_to_boundary":
                payload = {"path": self.state.path_to_boundary(int(body["start"]))}
            elif self.path == "/api/reset":
                payload = self.state.reset()
            elif self.path == "/api/split":
                payload = self.state.split_preview(body.get("seams", []))
            elif self.path == "/api/analyze":
                payload = self.state.analyze_warp(body)
            elif self.path == "/api/generate":
                payload = self.state.generate(body)
            elif self.path == "/api/save":
                payload = self.state.save_spec(body)
            else:
                self.send_error(404, "unknown API endpoint")
                return
        except (ValueError, KeyError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(payload)

    def _send_json(self, payload: dict, status: int = 200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, name: str):
        path = (self.state.outdir / name).resolve()
        if path.parent != self.state.outdir.resolve() or not path.is_file():
            self.send_error(404, "no such generated file")
            return
        data = path.read_bytes()
        types = {".svg": "image/svg+xml", ".dxf": "application/dxf", ".json": "application/json", ".yaml": "text/yaml"}
        self.send_response(200)
        self.send_header("Content-Type", types.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(state: GuiState, port: int = 0) -> ThreadingHTTPServer:
    """Server bound to localhost; port 0 picks a free port."""
    handler = partial(GuiRequestHandler, state=state)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(
    mesh_path: str | Path,
    outdir: str | Path,
    port: int = 8787,
    open_browser: bool = True,
) -> None:
    """Run the GUI until interrupted."""
    state = GuiState.from_file(mesh_path, outdir)
    server = make_server(state, port=port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"flatpack GUI on {url}  (mesh: {state.mesh_name}, "
          f"{len(state.mesh.vertices)} vertices; Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()

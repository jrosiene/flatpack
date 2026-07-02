"""GUI server API, exercised over real HTTP against a live server thread."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from flatpack.gui.server import GuiState, make_server
from flatpack.synthetic import make_sphere_patch

N = 15  # grid resolution of the test mesh; vertex index = i * N + j
CENTER_SEAM = [i * N + N // 2 for i in range(N)]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("gui_out")
    state = GuiState(
        mesh=make_sphere_patch(radius=150.0, half_width=80.0, n=N),
        outdir=outdir,
        mesh_name="test",
    )
    srv = make_server(state, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(base, path, body=None):
    if body is None:
        req = urllib.request.Request(base + path)
    else:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read())


def test_serves_index_and_vendored_three(server):
    with urllib.request.urlopen(server + "/") as res:
        html = res.read().decode()
    assert "flatpack" in html and "app.js" in html
    with urllib.request.urlopen(server + "/vendor/three.module.min.js") as res:
        assert len(res.read()) > 100_000


def test_mesh_payload(server):
    data = call(server, "/api/mesh")
    assert data["name"] == "test"
    assert len(data["vertices"]) == N * N * 3
    assert len(data["faces"]) % 3 == 0


def test_shortest_path_follows_edges(server):
    data = call(server, "/api/path", {"start": 0, "end": N * N - 1})
    path = data["path"]
    assert path[0] == 0 and path[-1] == N * N - 1
    # Every consecutive pair is a grid edge (neighbouring i or j).
    for a, b in zip(path, path[1:]):
        ai, aj = divmod(a, N)
        bi, bj = divmod(b, N)
        assert abs(ai - bi) + abs(aj - bj) <= 2 and a != b


def test_split_preview(server):
    data = call(server, "/api/split", {"seams": [CENTER_SEAM]})
    assert data["n_panels"] == 2
    assert len(set(data["labels"])) == 2


def test_analyze_warp(server):
    spec = {
        "seams": [{"name": "center", "path": CENTER_SEAM}],
        "panels": [{"name": "west", "anchor_face": 0, "fabric": "rigid"}],
    }
    data = call(server, "/api/analyze", spec)
    panels = {p["name"]: p for p in data["panels"]}
    assert "west" in panels
    west = panels["west"]
    assert west["severity"] in ("ok", "marginal", "high", "error")
    assert "advice" in west and "fabric" in west
    if west["severity"] not in ("ok", "error"):
        assert len(west["worst_point_3d"]) == 3


def test_split_with_bad_seam_is_a_clean_400(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        call(server, "/api/split", {"seams": [[0, N * N - 1]]})
    assert err.value.code == 400
    body = json.loads(err.value.read())
    assert "not connected" in body["error"]


def test_generate_end_to_end(server):
    spec = {
        "units": "mm",
        "seam_allowance": 10,
        "seams": [{"name": "center", "path": CENTER_SEAM}],
        "panels": [
            {"name": "west", "anchor_face": 0, "fabric": "silnylon"},
        ],
    }
    data = call(server, "/api/generate", spec)
    assert "west" in data["panels"]
    assert "pattern.svg" in data["files"]
    assert data["report"]["west"]["distortion"]["triangles"] > 0

    # Generated files are downloadable, with or without cache-busting query...
    with urllib.request.urlopen(server + "/files/pattern.svg") as res:
        assert b"<svg" in res.read()
    with urllib.request.urlopen(server + "/files/pattern.svg?t=123") as res:
        assert b"<svg" in res.read()
    # ...but nothing outside the output directory is.
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(server + "/files/../seams.yaml")
    assert err.value.code == 404


@pytest.fixture
def fresh_server(tmp_path):
    """Function-scoped server for tests that mutate the mesh (cuts)."""
    state = GuiState(
        mesh=make_sphere_patch(radius=150.0, half_width=80.0, n=N),
        outdir=tmp_path,
        mesh_name="test",
    )
    srv = make_server(state, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown()


def test_cut_returns_path_and_updated_mesh(fresh_server):
    base, state = fresh_server
    start, end = N - 1, (N - 1) * N  # anti-diagonal: crosses faces
    data = call(base, "/api/cut", {"start": start, "end": end})
    assert data["path"][0] == start and data["path"][-1] == end
    n_new = len(data["mesh"]["vertices"]) // 3
    assert n_new > N * N, "cut should insert vertices"
    assert state.modified

    # The cut path is now a valid seam: splitting along it gives 2 panels.
    split = call(base, "/api/split", {"seams": [data["path"]]})
    assert split["n_panels"] == 2


def test_cut_then_generate_end_to_end(fresh_server):
    base, _ = fresh_server
    data = call(base, "/api/cut", {"start": N - 1, "end": (N - 1) * N})
    spec = {
        "units": "mm",
        "seam_allowance": 10,
        "seams": [{"name": "diag", "path": data["path"]}],
        "panels": [],
    }
    out = call(base, "/api/generate", spec)
    assert len(out["panels"]) == 2

    # Saving also exports the cut mesh, since seams.yaml refers to it.
    saved = call(base, "/api/save", spec)
    assert saved["mesh"].endswith("shell_cut.obj")


def test_scale_multiplies_coordinates(fresh_server):
    base, state = fresh_server
    before = call(base, "/api/mesh")["vertices"]
    data = call(base, "/api/scale", {"factor": 25.4})
    after = data["mesh"]["vertices"]
    assert after[0] == pytest.approx(before[0] * 25.4)
    assert len(after) == len(before)
    assert state.modified

    # Reset undoes the scaling too.
    restored = call(base, "/api/reset", {})["mesh"]["vertices"]
    assert restored[0] == pytest.approx(before[0])


def test_scale_rejects_nonsense(fresh_server):
    base, _ = fresh_server
    with pytest.raises(urllib.error.HTTPError) as err:
        call(base, "/api/scale", {"factor": -2})
    assert err.value.code == 400


def test_add_vertex_and_use_in_seam(fresh_server):
    base, state = fresh_server
    mesh = state.mesh
    target = ((mesh.vertices[0] + mesh.vertices[1]) / 2).tolist()
    face = next(i for i, f in enumerate(mesh.faces) if 0 in f and 1 in f)

    data = call(base, "/api/add_vertex", {"face": int(face), "point": target})
    vertex = data["vertex"]
    assert vertex == N * N  # appended after the original grid
    assert len(data["mesh"]["vertices"]) // 3 == N * N + 1

    # The new vertex works as a path endpoint immediately.
    path = call(base, "/api/path", {"start": vertex, "end": N * N - 1})["path"]
    assert path[0] == vertex


def test_path_to_boundary(fresh_server):
    base, _ = fresh_server
    center = (N // 2) * N + N // 2  # interior vertex of the patch
    path = call(base, "/api/path_to_boundary", {"start": center})["path"]
    assert path[0] == center
    assert len(path) == N // 2 + 1  # straight run to the nearest edge
    # Endpoint really is on the boundary (row or column 0 / N-1).
    i, j = divmod(path[-1], N)
    assert i in (0, N - 1) or j in (0, N - 1)

    # Starting on the boundary is a no-op path.
    assert call(base, "/api/path_to_boundary", {"start": 0})["path"] == [0]


def test_reset_restores_original_mesh(fresh_server):
    base, state = fresh_server
    call(base, "/api/cut", {"start": N - 1, "end": (N - 1) * N})
    data = call(base, "/api/reset", {})
    assert len(data["mesh"]["vertices"]) // 3 == N * N
    assert not state.modified
    # Edge paths work again on the restored mesh.
    path = call(base, "/api/path", {"start": 0, "end": N - 1})["path"]
    assert path[0] == 0 and path[-1] == N - 1


def test_load_seams_round_trips(fresh_server):
    """Save a session, then load it back and confirm it is restored."""
    base, state = fresh_server
    center_dart = [(N // 2) * N + N // 2, (N // 2) * N + N // 2 - 1]
    spec = {
        "units": "mm",
        "seam_allowance": 14,
        "edge_labels": "cm",
        "seams": [{"name": "center", "path": CENTER_SEAM}],
        "marks": [{"vertex": 5, "type": "bartack", "label": "strap", "toward": 6}],
        "panels": [{"name": "west", "anchor_face": 0, "fabric": "ultrastretch"}],
    }
    import yaml as _yaml

    data = call(base, "/api/load", {"yaml": _yaml.safe_dump(spec)})
    assert data["seam_allowance"] == 14
    assert data["edge_labels"] == "cm"
    assert data["seams"][0]["path"] == CENTER_SEAM
    assert data["marks"][0]["label"] == "strap"
    # The 'west' panel is mapped to whichever component its anchor face is in.
    west = next(p for p in data["panels"] if p["name"] == "west")
    assert west["fabric"] == "ultrastretch"
    assert 0 <= west["label"] < data["n_panels"]


def test_load_rejects_foreign_mesh(fresh_server):
    base, _ = fresh_server
    import yaml as _yaml

    spec = {"seams": [{"name": "x", "path": [0, 999999]}]}
    with pytest.raises(urllib.error.HTTPError) as err:
        call(base, "/api/load", {"yaml": _yaml.safe_dump(spec)})
    assert err.value.code == 400
    assert "same model" in json.loads(err.value.read())["error"]


def test_save_spec_round_trips(server, tmp_path):
    spec = {
        "units": "mm",
        "seam_allowance": 12,
        "seams": [{"name": "center", "path": CENTER_SEAM}],
        "panels": [],
    }
    data = call(server, "/api/save", spec)
    from flatpack.seams import load_seam_spec

    loaded = load_seam_spec(data["saved"])
    assert loaded.seam_allowance == 12
    assert loaded.seams == [CENTER_SEAM]

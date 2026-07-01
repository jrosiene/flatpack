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

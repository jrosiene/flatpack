"""Seam file parsing and mesh splitting."""

import numpy as np
import pytest
import yaml

from flatpack.seams import load_seam_spec, split_mesh
from flatpack.synthetic import make_plane


@pytest.fixture
def grid():
    """5x5-vertex flat grid; vertex index = row * 5 + col."""
    return make_plane(half_width=50.0, n=5)


def seam_file(tmp_path, data):
    path = tmp_path / "seams.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


CENTER_COLUMN = [2, 7, 12, 17, 22]  # a straight vertical cut through the grid


def test_split_into_two_panels(grid, tmp_path):
    spec = load_seam_spec(
        seam_file(
            tmp_path,
            {
                "seams": [{"name": "cut", "path": CENTER_COLUMN}],
                "panels": [
                    {"name": "west", "anchor_face": 0, "fabric": "silnylon"},
                    {"name": "east", "anchor_face": len(grid.faces) - 1},
                ],
                "seam_allowance": 12,
            },
        )
    )
    assert spec.seam_allowance == 12

    panels = split_mesh(grid, spec)
    assert sorted(p.name for p in panels) == ["east", "west"]
    # Faces are partitioned, not duplicated.
    assert sum(len(p.faces) for p in panels) == len(grid.faces)
    # Seam vertices appear in both panels (they will be sewn together).
    west = next(p for p in panels if p.name == "west")
    east = next(p for p in panels if p.name == "east")
    for v in CENTER_COLUMN:
        assert v in west.orig_vertex_index
        assert v in east.orig_vertex_index
    assert west.spec.fabric == "silnylon"


def test_vertex_mapping_round_trips(grid, tmp_path):
    spec = load_seam_spec(
        seam_file(tmp_path, {"seams": [{"name": "cut", "path": CENTER_COLUMN}]})
    )
    for panel in split_mesh(grid, spec):
        for orig in panel.orig_vertex_index:
            local = panel.local_index(int(orig))
            assert np.allclose(panel.vertices[local], grid.vertices[orig])


def test_no_seams_gives_one_panel(grid, tmp_path):
    spec = load_seam_spec(seam_file(tmp_path, {}))
    panels = split_mesh(grid, spec)
    assert len(panels) == 1
    assert len(panels[0].faces) == len(grid.faces)


def test_invalid_seam_path_raises(grid, tmp_path):
    # 0 and 12 are not connected by an edge.
    spec = load_seam_spec(
        seam_file(tmp_path, {"seams": [{"name": "bad", "path": [0, 12]}]})
    )
    with pytest.raises(ValueError, match="not connected by a mesh edge"):
        split_mesh(grid, spec)


def test_two_anchors_in_same_component_raises(grid, tmp_path):
    spec = load_seam_spec(
        seam_file(
            tmp_path,
            {
                "seams": [],
                "panels": [
                    {"name": "a", "anchor_face": 0},
                    {"name": "b", "anchor_face": 1},
                ],
            },
        )
    )
    with pytest.raises(ValueError, match="same component"):
        split_mesh(grid, spec)


def test_panel_flattens_after_split(grid, tmp_path):
    """The panels coming out of a split are valid input for the flattener."""
    from flatpack.flatten import flatten
    import trimesh

    spec = load_seam_spec(
        seam_file(tmp_path, {"seams": [{"name": "cut", "path": CENTER_COLUMN}]})
    )
    for panel in split_mesh(grid, spec):
        result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
        assert np.allclose(result.distortion.sigma1, 1.0, atol=1e-8)

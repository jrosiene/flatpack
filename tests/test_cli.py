"""End-to-end smoke tests through the CLI."""

import json

from flatpack.cli import main


def test_demo_runs_end_to_end(tmp_path, capsys):
    assert main(["demo", "-o", str(tmp_path)]) == 0

    assert (tmp_path / "pattern.svg").exists()
    assert (tmp_path / "pattern.dxf").exists()
    assert (tmp_path / "report.json").exists()
    assert list(tmp_path.glob("page_*.svg")), "expected tiled pages"

    report = json.loads((tmp_path / "report.json").read_text())
    assert set(report) == {"left", "right"}
    for panel in report.values():
        assert "distortion" in panel and "fabric_fit" in panel

    out = capsys.readouterr().out
    assert "panel 'left'" in out
    assert "fabric fit" in out


def test_flatten_command_on_exported_demo(tmp_path):
    """The demo writes its own mesh + seam file; feed them back through `flatten`."""
    demo_dir = tmp_path / "demo"
    assert main(["demo", "-o", str(demo_dir)]) == 0

    out_dir = tmp_path / "pattern"
    assert (
        main(
            [
                "flatten",
                str(demo_dir / "demo_shell.obj"),
                str(demo_dir / "demo_seams.yaml"),
                "-o",
                str(out_dir),
                "--page",
                "a4",
                "--no-relax",
            ]
        )
        == 0
    )
    assert (out_dir / "pattern.svg").exists()
    assert (out_dir / "report.json").exists()

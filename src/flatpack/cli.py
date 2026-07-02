"""Command line interface.

    flatpack flatten shell.stl seams.yaml -o pattern/
    flatpack demo -o demo/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from flatpack.pipeline import PanelResult, process, process_files
from flatpack.seams import load_seam_spec
from flatpack.synthetic import make_sphere_patch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="flatpack",
        description="Flatten 3D backpack shells into 2D fabric panels",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_flat = sub.add_parser("flatten", help="run the full pipeline on a mesh")
    p_flat.add_argument("mesh", help="STL/OBJ/PLY shell mesh (millimetres)")
    p_flat.add_argument("seams", help="seam definition YAML")
    p_flat.add_argument("-o", "--outdir", default="pattern", help="output directory")
    p_flat.add_argument("--page", choices=["letter", "a4"], default="letter")
    p_flat.add_argument(
        "--no-relax",
        action="store_true",
        help="skip the fabric-aware relaxation step (pure conformal map)",
    )

    p_demo = sub.add_parser(
        "demo", help="run the pipeline on a synthetic doubly-curved patch"
    )
    p_demo.add_argument("-o", "--outdir", default="demo", help="output directory")
    p_demo.add_argument("--page", choices=["letter", "a4"], default="letter")

    p_gui = sub.add_parser(
        "gui", help="open the browser GUI to draw seams on a mesh"
    )
    p_gui.add_argument(
        "mesh",
        nargs="?",
        help="OBJ/PLY shell mesh; omit to edit a synthetic demo patch",
    )
    p_gui.add_argument("-o", "--outdir", default="pattern", help="output directory")
    p_gui.add_argument("--port", type=int, default=8787)
    p_gui.add_argument("--no-browser", action="store_true", help="don't open a browser")

    args = parser.parse_args(argv)

    if args.command == "gui":
        from flatpack.gui import serve

        mesh_path = args.mesh
        if mesh_path is None:
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            mesh_path = outdir / "demo_shell.obj"
            make_sphere_patch(radius=200.0, half_width=120.0, n=25).export(str(mesh_path))
            print(f"no mesh given; editing synthetic patch {mesh_path}")
        serve(mesh_path, args.outdir, port=args.port, open_browser=not args.no_browser)
        return 0

    if args.command == "flatten":
        results = process_files(
            args.mesh, args.seams, args.outdir, page=args.page, relax=not args.no_relax
        )
    else:
        results = _demo(Path(args.outdir), page=args.page)

    _print_report(results)
    print(
        f"\nwrote pattern.svg, pattern.dxf, tiled pages, pattern_tiled.pdf "
        f"and report.json to {args.outdir}/"
    )
    return 0


def _demo(outdir: Path, page: str) -> list[PanelResult]:
    """Sphere patch (like a domed pack back panel) split down the middle."""
    mesh = make_sphere_patch(radius=200.0, half_width=120.0, n=25)
    # Grid vertices are indexed i * n + j at position (s[i], s[j]); the seam
    # fixes j at the middle column, cutting the patch into j-low / j-high
    # halves, so grainlines below run along i on each half's outer edge.
    n = 25
    center_col = n // 2
    seam_path = [i * n + center_col for i in range(n)]

    spec_data = {
        "units": "mm",
        "seam_allowance": 10,
        "seams": [{"name": "center", "path": seam_path}],
        "panels": [
            {
                "name": "left",
                "anchor_face": 0,
                "fabric": "silnylon",
                "grain": [0, (n - 1) * n],  # j = 0 edge, along i
                "notches": [seam_path[n // 2]],
            },
            {
                "name": "right",
                "anchor_face": len(mesh.faces) - 1,
                "fabric": "ultrastretch",
                "grain": [n - 1, n * n - 1],  # j = n-1 edge, along i
                "notches": [seam_path[n // 2]],
            },
        ],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    seam_file = outdir / "demo_seams.yaml"
    seam_file.write_text(yaml.safe_dump(spec_data))
    mesh.export(str(outdir / "demo_shell.obj"))
    return process(mesh, load_seam_spec(seam_file), outdir, page=page)


def _print_report(results: list[PanelResult]) -> None:
    for r in results:
        d = r.flat.distortion.summary()
        f = r.fit.summary()
        worst = r.flat.distortion.worst_triangle_uv()
        print(f"\npanel {r.panel.name!r}  ({f['fabric']}, {d['triangles']} triangles)")
        print(
            f"  area ratio      mean {d['area_ratio_mean']:.3f}   "
            f"range [{d['area_ratio_worst_low']:.3f}, {d['area_ratio_worst_high']:.3f}]"
        )
        print(
            f"  strain          stretch max {d['max_stretch_strain'] * 100:.1f}%   "
            f"compress max {d['max_compress_strain'] * 100:.1f}%"
        )
        print(
            f"  angle error     mean {d['angle_error_deg_mean']:.2f} deg   "
            f"max {d['angle_error_deg_max']:.2f} deg"
        )
        print(
            f"  fabric fit      {f['fraction_ok'] * 100:.0f}% ok, "
            f"{f['triangles_needing_dart']} need dart, "
            f"{f['triangles_needing_relief']} need relief"
        )
        if f["triangles_needing_dart"] or f["triangles_needing_relief"]:
            print(
                f"  worst spot at   uv ({worst[0]:.0f}, {worst[1]:.0f}) mm "
                "- consider a dart or relief cut there"
            )
        if d["flipped_triangles"]:
            print(f"  WARNING: {d['flipped_triangles']} flipped triangles")


if __name__ == "__main__":
    sys.exit(main())

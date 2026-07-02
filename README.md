# flatpack

Convert a 3D backpack shell (triangulated mesh from CadQuery, OpenSCAD,
Blender, ...) into flat 2D fabric panels for MYOG pattern drafting.

**Windows, no Python?** A standalone `flatpack.exe` can be built in one
command on any Windows machine with Python — see
[Building the executable yourself](#building-the-executable-yourself).
Double-clicking the exe opens the seam editor in your browser with a
demo shell; dragging a mesh file onto it opens that mesh.

The pipeline:

1. **Input** — a triangulated mesh (OBJ/PLY preferred, STL accepted) in
   millimetres. Parametric CadQuery/OpenSCAD surfaces enter by tessellating
   to a mesh first.
2. **Seams** — you cut the shell into panels by listing vertex paths in a
   small YAML file (see below). No automatic seam-finding. Seams are
   genuinely opened (vertices duplicated along the cut), so open-ended
   shells work: one seam up the side of a tube-shaped pack body unrolls
   it into a single flat panel. Panels that still wrap around are
   rejected with a message saying which seam is missing.
3. **Flattening** — each panel is flattened with a Least Squares Conformal
   Map (LSCM, implemented from scratch in ~100 lines of numpy/scipy and
   validated against libigl in the test suite). Per-triangle area and angle
   distortion is reported so you know where a dart or relief cut is needed.
4. **Fabric awareness** — each panel can name a fabric (silnylon, X-Pac,
   UltraStretch, ...) and a stretch axis. Distortion the fabric can absorb
   is not flagged, and an anisotropic relaxation step biases residual error
   into the stretchy direction.
5. **Output** — seam-allowance-offset outlines with notches, grainline
   arrows and labels as true-scale SVG and DXF, plus letter/A4 page tiles
   (with overlap strips, crop marks and page labels) for home printing.

## Install (from source)

You need Python 3.11+ and either [uv](https://docs.astral.sh/uv/) (recommended)
or plain pip. (Windows users can skip all of this — see the exe above.)

```bash
git clone https://github.com/jrosiene/flatpack.git
cd flatpack
uv sync            # creates .venv and installs everything (incl. test deps)
```

Without uv:

```bash
git clone https://github.com/jrosiene/flatpack.git
cd flatpack
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
```

## Quick GUI demo (no mesh needed)

```bash
uv run flatpack gui             # or just `flatpack gui` in an activated venv
```

This generates a synthetic doubly-curved patch (a dome, like a pack back
panel), starts a local server, and opens the seam editor in your browser
at http://127.0.0.1:8787. Try this flow:

1. Click **Draw seam**, then click a vertex near the top edge of the dome
   and another near the bottom edge — the seam snaps to the surface
   between your clicks. Click **Finish seam**.
2. Click **Preview split** — the two panels light up in different colours.
3. Click a panel in the list, give it a name and a fabric
   (e.g. `ultrastretch`).
4. Click **Generate pattern** — you get the flattened panels with seam
   allowance as SVG/DXF, print-ready tiled pages, and a per-panel
   distortion report, all written to `pattern/` with download links in
   the browser.

To work on your own shell, pass a mesh (OBJ or PLY preferred — STL loses
the vertex indexing that seam files rely on):

```bash
uv run flatpack gui shell.obj -o pattern/
```

## Other commands

```bash
uv run flatpack demo -o demo/               # headless end-to-end demo, no browser
uv run flatpack flatten shell.obj seams.yaml -o pattern/   # CLI pipeline
uv run pytest                               # the whole test suite
```

## GUI

`flatpack gui [mesh.obj]` starts a local server (stdlib only, no extra
dependencies; three.js is vendored so it works offline) and opens a
browser with the seam editor. Omit the mesh argument to play with a
synthetic demo patch.

- **Navigation** (Fusion 360 bindings, shown in the corner of the
  viewport): middle-drag pans, Shift+middle-drag orbits, wheel zooms —
  in every mode, so you can orbit while drawing a seam. In Orbit mode,
  plain left-drag also orbits (for trackpads without a middle button).
  The orbit turntable spins around the mesh's vertical (Z) axis.
- **Draw seam** mode: click vertices on the mesh; between clicks the seam
  snaps to the shortest path along the surface (server-side Dijkstra), so
  a seam across the whole shell takes a handful of clicks. *Finish seam*
  commits it; *Undo leg* steps back one click.
- **To edge** (in the Seams section): extends the seam you're drawing
  along the surface to the nearest mesh boundary — no hunting for the
  right boundary vertex.
- **Add vertex** mode: when a seam needs to start or end where the mesh
  simply has no vertex, click there — a new vertex is inserted on the
  nearest edge (both adjacent faces are retriangulated) and can be used
  in seams, darts, notches and marks immediately.
- **Straight cut** (checkbox in the Seams section): instead of following
  existing mesh edges, the seam cuts straight across triangles — for
  diagonal seams the triangulation doesn't line up with. The mesh is
  re-triangulated on the fly (new vertices on the crossed edges; crossings
  near an existing vertex snap to it to avoid slivers), and *Reset cuts*
  restores the original geometry. When you save, the cut mesh is exported
  as `shell_cut.obj` next to `seams.yaml` so the session replays from the
  CLI.
- **Preview split** colours the mesh by the panels your seams produce.
  Click a panel to name it and pick its fabric and stretch axis.
- **Notch** mode toggles match notches on seam vertices (they end up on
  both panels that share the seam, which is what you want for alignment).
- **Dart** mode: click the dart mouth (a boundary or seam vertex), then
  the apex. The dart is treated as a slit; when the panel flattens, the
  slit spreads into a V and *that opening is the dart intake* — computed
  from the actual curvature rather than guessed. The pattern shows both
  legs as fold/stitch lines meeting at an apex circle. Darts measurably
  reduce distortion (there's a test asserting it), so use the fabric-fit
  report's "worst spot" to decide where one is needed.
- **Mark** mode: place sewing marks — **bar tacks** (click anchor, then a
  second vertex for orientation; drawn as a thick 12 mm bar) and
  **attachment points** (single click; drawn as a circle-cross target)
  with labels like "shoulder strap". They print on the panels and land on
  their own MARK/DART layers in the DXF.
- **Grainline** mode: select a panel, click two vertices. *✕ Remove
  grainline* in the panel properties (visible once a panel is selected)
  takes it off again; without a grainline, panels are auto-aligned to
  their principal axis for layout.
- **Measure** mode: click two vertices to read the straight-line
  distance in mm — a quick sanity check that the mesh imported at the
  size you expected. After measuring, type the length that span *should*
  be and hit *Rescale*: the whole mesh is scaled so the measurement
  matches (the precise way to fix units when you know one real
  dimension, e.g. "this back panel is 500 mm tall").
- **Units**: the header shows the mesh's bounding-box size and warns when
  it looks too small to be a pack. The *Scale ×* control (with in→mm,
  cm→mm, m→mm presets) rescales the mesh in place — everything downstream
  assumes millimetres. Seams, notches and grainlines survive scaling;
  *Reset mesh edits* restores the original.
- **Generate pattern** runs the exact same pipeline as the CLI into the
  output directory, pops up the SVG preview, and lists per-panel
  distortion/fabric-fit numbers with download links. **Save seams.yaml**
  writes the spec so the whole session is reproducible from the CLI.

Everything the mouse can do is also scriptable from the browser console
via `window.flatpack` (used by the automated browser test).

`flatpack demo` builds a sphere patch (think: domed back panel), splits it
down the middle, flattens both halves — one as silnylon, one as
UltraStretch — and writes `pattern.svg`, `pattern.dxf`, tiled
`page_A1.svg`-style sheets and a `report.json` with distortion and
fabric-fit numbers. It also prints a per-panel report:

```
panel 'left'  (silnylon, 288 triangles)
  area ratio      mean 1.000   range [0.960, 1.043]
  strain          stretch max 2.2%   compress max 2.1%
  angle error     mean 0.05 deg   max 0.15 deg
  fabric fit      93% ok, 9 need dart, 12 need relief
  worst spot at   uv (12, -48) mm - consider a dart or relief cut there
```

## Seam file format

Vertex and face indices refer to the mesh as loaded (`process=False`, so
the file's vertex order is preserved — use OBJ or PLY; STL stores no
shared vertices).

```yaml
units: mm
seam_allowance: 10          # mm
seams:
  - name: side              # names are documentation only
    path: [3, 18, 33, 48]   # consecutive entries must share a mesh edge
panels:
  - name: front
    anchor_face: 12         # any face index inside this panel
    fabric: ultrastretch    # see flatpack.fabric.FABRICS; default "rigid"
    stretch_axis_deg: 0     # stretch axis, degrees from the grainline
    grain: [3, 48]          # vertex pair defining the grainline
    notches: [18]           # boundary vertices to mark with notches
darts:
  - name: back_dart
    path: [55, 40, 25]      # mouth (boundary/seam vertex) ... apex
marks:
  - vertex: 87
    type: bartack           # bartack | attach
    label: shoulder strap
    toward: 88              # optional: bar tack orientation
  - vertex: 120
    type: attach
    label: daisy chain
```

Finding vertex indices: any mesh viewer that shows indices works
(MeshLab: Render → Show Label; Blender: indices in edit mode via
developer extras). A helper for interactive picking is a natural next
step.

## How to read the distortion report

For each triangle we compute the singular values σ1 ≥ σ2 of the 3D→2D
map. σ is (flat length) / (surface length) along a principal direction:

- **σ > 1**: the cut piece is bigger than the surface there — the excess
  must be darted or gathered out (no fabric stretches negatively).
- **σ < 1**: the piece is smaller — the fabric must stretch by 1−σ to
  reach. Flagged only if that exceeds the fabric's capability in that
  direction (plus 2% sewing ease).
- σ1/σ2 is angle (shear) distortion; LSCM keeps it near 1 by design, so
  almost all error shows up as area, which is the honest quantity for
  deciding on darts.

`report.json` and the CLI output include the uv location of the worst
triangle so you know where to put the dart.

## Library choice: why trimesh + a from-scratch LSCM

Evaluated for the mesh/flattening core:

| option | verdict |
|---|---|
| **trimesh** (chosen, mesh backbone) | Light, ubiquitous, great STL/OBJ/PLY I/O, face adjacency and graph utilities. No flattening of its own, which is fine. |
| **libigl python bindings** (chosen, tests only) | Has `igl.lscm` and installs cleanly from wheels, but the Python API has churned between releases, and we'd still write all the distortion/fabric code ourselves. Used as a *reference oracle*: the test suite checks our LSCM against `igl.lscm` (dev dependency only). |
| **from-scratch LSCM** (chosen, runtime) | The whole solver is ~100 readable lines on `scipy.sparse` (`flatten.py`), it's validated against libigl and against developable surfaces, and it's easy to extend — the fabric-aware relaxation hooks straight into it. |
| **Blender `bpy`** (rejected) | ~300 MB, pinned to specific Python versions, UV tools are interactive-editor-shaped. Overkill for a library. |

ABF/ABF++ would give marginally better angle preservation but is much
more code for little gain at panel scale; LSCM distortion numbers tell
you where darts go either way.

## Package layout

```
src/flatpack/
  synthetic.py   test surfaces: plane, cylinder (developable), sphere, saddle
  meshutil.py    triangle frames, boundary loops, edge utilities
  flatten.py     LSCM solver (the core)
  distortion.py  per-triangle singular-value metrics
  fabric.py      fabric model, fit check, anisotropic relaxation
  seams.py       seam YAML + mesh splitting into panels
  cut.py         plane cutting across faces (diagonal seams)
  export.py      panel layout, seam allowance, notches, grainline; SVG/DXF
  tiling.py      letter/A4 page tiling with overlap + registration marks
  pipeline.py    end-to-end orchestration
  cli.py         `flatpack flatten` / `flatpack demo` / `flatpack gui`
  gui/
    server.py    stdlib HTTP server + JSON API behind the GUI
    static/      index.html, app.js (three.js seam editor), vendored three.js
packaging/       PyInstaller entry + spec for the standalone executable
.github/workflows/build.yml   CI tests
```

## Building the executable yourself

```bash
uv run --with 'pyinstaller>=6.0' pyinstaller packaging/flatpack.spec
```

produces `dist/flatpack` for the platform you run it on (`flatpack.exe`
on Windows; on Windows without uv: `pip install . "pyinstaller>=6.0"`
then `pyinstaller packaging/flatpack.spec`. PyInstaller does not
cross-compile, so the exe must be built on Windows). The exe with no
arguments opens the GUI; with a mesh path it opens the GUI on that mesh;
`flatten` / `demo` / `gui` subcommands work as in the CLI. Windows
SmartScreen warns about unsigned executables the first time: choose
"More info" → "Run anyway".

## Limitations / next steps

- Dart *placement* is manual (the report tells you where distortion is
  worst; you choose where the dart goes). The intake geometry is computed.
- Fisheye (fully interior) darts are rejected: a closed interior slit
  makes the panel an annulus, which a conformal map cannot flatten.
- Panel packing is a simple left-to-right shelf; no nesting.
- The relaxation is a damped spring iteration, not a full ARAP solve —
  good enough to bias error into the stretch axis, not a simulation.

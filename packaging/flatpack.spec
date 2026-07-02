# PyInstaller build definition for the standalone flatpack executable.
#
#   pyinstaller packaging/flatpack.spec
#
# Produces dist/flatpack (dist/flatpack.exe on Windows): a single-file
# executable bundling Python, all dependencies, and the GUI's static
# assets (vendored three.js included), so it runs on machines without
# Python installed. Built for Windows by the GitHub Actions workflow in
# .github/workflows/build.yml.

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH).parent  # packaging/ -> repository root
src = root / "src"

a = Analysis(
    [str(Path(SPECPATH) / "entry.py")],
    pathex=[str(src)],
    datas=[
        # server.py resolves static/ relative to its own __file__, which
        # lands in the bundle at this same relative location.
        (str(src / "flatpack" / "gui" / "static"), "flatpack/gui/static"),
    ],
    # These libraries import parts of themselves dynamically (mesh format
    # loaders, DXF machinery), which static analysis can miss.
    hiddenimports=(
        collect_submodules("trimesh")
        + collect_submodules("ezdxf")
        + collect_submodules("shapely")
    ),
    excludes=["tkinter", "matplotlib", "IPython"],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="flatpack",
    console=True,  # keep the console: it shows the GUI's URL and CLI output
    upx=False,
)

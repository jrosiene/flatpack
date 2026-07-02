"""Entry point for the standalone executable.

Behaviour is tuned for double-click use on Windows:

- no arguments            -> open the GUI (with the built-in demo patch)
- first arg is a command  -> normal CLI (flatten / demo / gui ...)
- first arg is a file     -> open the GUI on that mesh, so dragging a
                             mesh file onto flatpack.exe just works
"""

import sys

from flatpack.cli import main

COMMANDS = {"flatten", "demo", "gui"}

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = ["gui"]
    elif args[0] not in COMMANDS and not args[0].startswith("-"):
        args = ["gui", *args]
    sys.exit(main(args))

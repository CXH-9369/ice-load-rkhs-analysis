"""Add project-local optional binary dependencies to ``sys.path``."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# The pipeline generates image files headlessly; a GUI backend can fail on
# machines without a complete Tcl/Tk installation.
os.environ.setdefault("MPLBACKEND", "Agg")
for relative in (Path(".deps/science"), Path(".deps/opencv")):
    path = ROOT / relative
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))

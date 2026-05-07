"""Centralised paths and physics constants.

The MadGraph install directory is overridable via the MG5_INSTALL_DIR environment
variable. All other paths are derived from it. Set the override before running any
script that touches MG5 data:

    export MG5_INSTALL_DIR=/your/path/to/MG5_aMC_v3_5_6
"""
import os
import sys
from pathlib import Path

# Until pyproject.toml lands (P3), expose `src/` so legacy scripts can `import diboson.*`.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# MadGraph install root. Override per-machine via the MG5_INSTALL_DIR env var.
MG5_INSTALL_DIR = Path(
    os.environ.get("MG5_INSTALL_DIR", "/home/felip/MG5_aMC")
)

# MadGraph process directories.
ZZ_PROCESS_DIR = MG5_INSTALL_DIR / "pp_ZZ_SM"
WW_PROCESS_DIR = MG5_INSTALL_DIR / "pp_WW_SM"

# 'Plots and data' subdirectories where analysis code reads/writes.
ZZ_DATA_DIR = ZZ_PROCESS_DIR / "Plots and data"
WW_DATA_DIR = WW_PROCESS_DIR / "Plots and data"


# Beam / event-generation parameters used by histogram normalisations.
COM_ENERGY = 13_000  # GeV
NEVENTS = 1_000_000

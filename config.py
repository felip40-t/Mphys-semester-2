"""Centralised paths and physics constants.

MadGraph install directory is overridable via the MG5_INSTALL_DIR environment variable:
    export MG5_INSTALL_DIR=/your/path/to/MG5_aMC_v3_5_6

Analysis output directories are project-local under outputs/:
    outputs/data/raw/ZZ|WW       — per-bin angular data written by lorentz_boost scripts
    outputs/data/processed/ZZ|WW — coefficient CSVs and Bell/concurrence grids
    outputs/plots/ZZ|WW          — generated PDF/PNG figures

All six output directories are created automatically on import.
"""
import os
import sys
from pathlib import Path

# Until pyproject.toml lands (P3), expose `src/` so legacy scripts can `import diboson.*`.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

PROJECT_DIR = Path(__file__).resolve().parent

# MadGraph install root. Override per-machine via the MG5_INSTALL_DIR env var.
MG5_INSTALL_DIR = Path(
    os.environ.get("MG5_INSTALL_DIR", "/home/felip/MG5_aMC")
)

# MadGraph process directories (created by automate_event_gen; may not exist yet).
ZZ_PROCESS_DIR = MG5_INSTALL_DIR / "pp_ZZ_SM"
WW_PROCESS_DIR = MG5_INSTALL_DIR / "pp_WW_SM"

# Analysis output directories (project-local).
ZZ_RAW_DIR       = PROJECT_DIR / "outputs" / "data" / "raw"       / "ZZ"
WW_RAW_DIR       = PROJECT_DIR / "outputs" / "data" / "raw"       / "WW"
ZZ_PROCESSED_DIR = PROJECT_DIR / "outputs" / "data" / "processed" / "ZZ"
WW_PROCESSED_DIR = PROJECT_DIR / "outputs" / "data" / "processed" / "WW"
ZZ_PLOTS_DIR     = PROJECT_DIR / "outputs" / "plots" / "ZZ"
WW_PLOTS_DIR     = PROJECT_DIR / "outputs" / "plots" / "WW"

for _d in [ZZ_RAW_DIR, WW_RAW_DIR, ZZ_PROCESSED_DIR, WW_PROCESSED_DIR, ZZ_PLOTS_DIR, WW_PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Beam / event-generation parameters.
COM_ENERGY = 13_000  # GeV
NEVENTS = 1_000_000

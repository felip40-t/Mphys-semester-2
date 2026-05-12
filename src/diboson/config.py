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
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# MadGraph install root. Override per-machine via the MG5_INSTALL_DIR env var.
MG5_INSTALL_DIR = Path(
    os.environ.get("MG5_INSTALL_DIR", "/home/felip/MG5_aMC")
)

# MadGraph process directories (created by automate_event_gen; may not exist yet).
ZZ_PROCESS_DIR = MG5_INSTALL_DIR / "pp_ZZ"
WW_PROCESS_DIR = MG5_INSTALL_DIR / "pp_WW"

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
NEVENTS = 50_000

# ZZ electroweak coupling constants.
ZZ_G_L = -0.26953
ZZ_G_R = 0.2317

ZZ_ETA = 0.213
WW_ETA = 1.0

# Phase-space binning — identical bin widths for both processes, different mass ranges.
N_COS_BINS = 10          # cosΘ: 0.0–1.0 in steps of 0.1
ZZ_N_MASS_BINS = 16     # M_ZZ: 200–1000 GeV in steps of 50 GeV
WW_N_MASS_BINS = 20     # M_WW: 200–1200 GeV in steps of 50 GeV
COS_BIN_MIN = 0.0
COS_BIN_WIDTH = 0.1
MASS_BIN_MIN = 200.0    # GeV
MASS_BIN_WIDTH = 50.0   # GeV

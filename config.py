"""Centralised paths and physics constants.

The MadGraph install directory is overridable via the MG5_INSTALL_DIR environment
variable. All other paths are derived from it. Set the override before running any
script that touches MG5 data:

    export MG5_INSTALL_DIR=/your/path/to/MG5_aMC_v3_5_6
"""
import os
from pathlib import Path

# MadGraph install root. Override per-machine via the MG5_INSTALL_DIR env var.
MG5_INSTALL_DIR = Path(
    os.environ.get("MG5_INSTALL_DIR", "/home/felipetcach/project/MG5_aMC_v3_5_6")
)

# MadGraph process directories (one per parallel analysis track).
ZZ_PROCESS_DIR = MG5_INSTALL_DIR / "pp_ZZ_SM"
WW_PROCESS_DIR = MG5_INSTALL_DIR / "pp_WW_SM"
WW_4L_PROCESS_DIR = MG5_INSTALL_DIR / "pp_WW_4l_final_process"
WZ_PROCESS_DIR = MG5_INSTALL_DIR / "pp_WZ_SM"

# Alternate ZZ directory referenced only by ZZ/zz_params_calc.py.
ZZ_PROCESS_DIR_LEGACY = MG5_INSTALL_DIR / "ZZ_process"

# Generic 4-lepton process dir referenced only by utils/histo_plotter.py.
FOUR_LEPTON_PROCESS_DIR = MG5_INSTALL_DIR / "4_lepton_process"

# Fortran (LO-only) reference samples used by the plot_histo_fortran scripts.
ZZ_FORTRAN_REF_DIR = MG5_INSTALL_DIR / "pp_4l_zz_LOonly"
WZ_FORTRAN_REF_DIR = MG5_INSTALL_DIR / "pp_4l_wz_LOonly"

# 'Plots and data' subdirectories where analysis code reads/writes.
ZZ_DATA_DIR = ZZ_PROCESS_DIR / "Plots and data"
WW_DATA_DIR = WW_PROCESS_DIR / "Plots and data"
WW_4L_DATA_DIR = WW_4L_PROCESS_DIR / "Plots and data"
WZ_DATA_DIR = WZ_PROCESS_DIR / "Plots and data"
ZZ_DATA_DIR_LEGACY = ZZ_PROCESS_DIR_LEGACY / "Plots and data"

# Reorganised data and entanglement-plot output dirs used by main_bell scripts.
ZZ_REORGANISED_DATA = ZZ_DATA_DIR / "reorganised_data"
ZZ_ENTANGLEMENT_PLOTS = ZZ_REORGANISED_DATA / "Entanglement plots"
WW_ORGANISED_DATA = WW_4L_DATA_DIR / "organised_data"
WW_PLOTS_DIR = WW_4L_DATA_DIR / "Plots"

# Beam / event-generation parameters used by histogram normalisations.
COM_ENERGY = 13_000  # GeV
NEVENTS = 1_000_000

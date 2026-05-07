# Re-exports from the root-level config.py so that package-style imports work:
#   from diboson.config import MG5_INSTALL_DIR
# In P1, when all scripts migrate into src/diboson/, the canonical content will
# move here and the root config.py will be removed.
from config import (  # noqa: F401
    MG5_INSTALL_DIR,
    ZZ_PROCESS_DIR,
    WW_PROCESS_DIR,
    WW_4L_PROCESS_DIR,
    ZZ_PROCESS_DIR_LEGACY,
    FOUR_LEPTON_PROCESS_DIR,
    ZZ_FORTRAN_REF_DIR,
    ZZ_DATA_DIR,
    WW_DATA_DIR,
    WW_4L_DATA_DIR,
    ZZ_DATA_DIR_LEGACY,
    ZZ_REORGANISED_DATA,
    ZZ_ENTANGLEMENT_PLOTS,
    WW_ORGANISED_DATA,
    WW_PLOTS_DIR,
    COM_ENERGY,
    NEVENTS,
)

# Re-exports from the root-level config.py so that package-style imports work:
#   from diboson.config import ZZ_RAW_DIR
# In P3, when all scripts migrate into src/diboson/, the canonical content will
# move here and the root config.py will be removed.
from config import (  # noqa: F401
    PROJECT_DIR,
    MG5_INSTALL_DIR,
    ZZ_PROCESS_DIR,
    WW_PROCESS_DIR,
    ZZ_RAW_DIR,
    WW_RAW_DIR,
    ZZ_PROCESSED_DIR,
    WW_PROCESSED_DIR,
    ZZ_PLOTS_DIR,
    WW_PLOTS_DIR,
    COM_ENERGY,
    NEVENTS,
)

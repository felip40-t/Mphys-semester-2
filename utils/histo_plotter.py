# Compatibility shim — kept until all callers migrate to diboson.plotting.
# Only read_data is used by the legacy ZZ/WW/WZ scripts.
import numpy as np


def read_data(file_path):
    """Read comma-delimited 4-momentum data from a .txt file."""
    return np.loadtxt(file_path, delimiter=',')

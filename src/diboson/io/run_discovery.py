import os
from glob import glob


def find_latest_run_dir(base_dir):
    """Find the highest-numbered run_* directory under base_dir."""
    run_dirs = glob(os.path.join(base_dir, "run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    run_numbers = [int(d.split("_")[-1]) for d in run_dirs]
    latest = max(run_numbers)
    return os.path.join(base_dir, f"run_{latest:02d}"), latest

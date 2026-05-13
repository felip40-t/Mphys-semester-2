"""Diboson entanglement analysis entry point.

Usage:
    python src/diboson/main.py --process ZZ
    python src/diboson/main.py --process WW
    python src/diboson/main.py --process ZZ --projection raw     # skip PSD projection
    python src/diboson/main.py --process ZZ --projection hard    # hard cutoff (default)
    python src/diboson/main.py --process ZZ --projection smooth  # gradual shift
    python src/diboson/main.py --process ZZ --plot-only          # replot from saved grids
"""

import argparse
import time

import numpy as np

from diboson.config import (
    ZZ_RAW_DIR, ZZ_PROCESSED_DIR, ZZ_PLOTS_DIR,
    WW_RAW_DIR, WW_PROCESSED_DIR, WW_PLOTS_DIR,
    ZZ_ETA, WW_ETA,
    N_COS_BINS,
    ZZ_N_MASS_BINS, WW_N_MASS_BINS,
    MASS_BIN_MIN, MASS_BIN_WIDTH,
    COS_BIN_MIN, COS_BIN_WIDTH
)
from diboson.physics.coefficients import (
    calculate_coefficients_AC, calculate_variance_AC,
    calculate_coefficients_fgh, calculate_variance_fgh,
)
from diboson.physics.density_matrix import (
    calculate_density_matrix_AC,
    calculate_density_matrix_fgh,
)
from diboson.plotting.contour import plot_contour_heatmap, generate_unphysicality_heatmap
from diboson.analysis.region_analysis import ProcessSpec, process_region as _process_region


def _zz_get_density_matrix(theta_paths, phi_paths):
    A, C = calculate_coefficients_AC(theta_paths, phi_paths)
    return calculate_density_matrix_AC(A, C)


def _ww_get_density_matrix(theta_paths, phi_paths):
    f, g, h = calculate_coefficients_fgh(theta_paths, phi_paths)
    return calculate_density_matrix_fgh(f, g, h)


_ANGLE_FILENAMES = dict(
    psi_filename="cos_psi.npy",
    inv_mass_filename="inv_mass.npy",
    theta_filenames={1: "theta1.npy", 3: "theta3.npy"},
    phi_filenames={1: "phi1.npy", 3: "phi3.npy"},
)

_PROCESS_CONFIGS = {
    "ZZ": dict(
        spec=ProcessSpec(
            name="ZZ",
            n_mass_bins=ZZ_N_MASS_BINS,
            n_cos_bins=N_COS_BINS,
            mass_width=MASS_BIN_WIDTH,
            cos_width=COS_BIN_WIDTH,
            mass_max=MASS_BIN_MIN + ZZ_N_MASS_BINS * MASS_BIN_WIDTH,
            cos_max=COS_BIN_MIN + N_COS_BINS * COS_BIN_WIDTH,
            mass_min=MASS_BIN_MIN,
            cos_min=COS_BIN_MIN,
            eta=ZZ_ETA,
            get_density_matrix=_zz_get_density_matrix,
            get_variance=calculate_variance_AC,
            **_ANGLE_FILENAMES,
        ),
        raw_dir=ZZ_RAW_DIR,
        processed_dir=ZZ_PROCESSED_DIR,
        plots_dir=ZZ_PLOTS_DIR,
        coeff_suffix="AC",
    ),
    "WW": dict(
        spec=ProcessSpec(
            name="WW",
            n_mass_bins=WW_N_MASS_BINS,
            n_cos_bins=N_COS_BINS,
            mass_width=MASS_BIN_WIDTH,
            cos_width=COS_BIN_WIDTH,
            mass_max=MASS_BIN_MIN + WW_N_MASS_BINS * MASS_BIN_WIDTH,
            cos_max=COS_BIN_MIN + N_COS_BINS * COS_BIN_WIDTH,
            mass_min=MASS_BIN_MIN,
            cos_min=COS_BIN_MIN,
            eta=WW_ETA,
            get_density_matrix=_ww_get_density_matrix,
            get_variance=calculate_variance_fgh,
            **_ANGLE_FILENAMES,
        ),
        raw_dir=WW_RAW_DIR,
        processed_dir=WW_PROCESSED_DIR,
        plots_dir=WW_PLOTS_DIR,
        coeff_suffix="fgh",
    ),
}


def run(process: str, projection: str = "hard", plot_only: bool = False, start_region=None) -> None:
    cfg = _PROCESS_CONFIGS[process]
    spec = cfg["spec"]
    raw_dir = cfg["raw_dir"]
    processed_dir = cfg["processed_dir"]
    plots_dir = cfg["plots_dir"]
    label = projection
    grid_name = f"{process}_{cfg['coeff_suffix']}_{label}"

    regions = spec.build_regions()
    shape = (spec.n_cos_bins, spec.n_mass_bins)

    def _load_grid(fname):
        path = processed_dir / fname
        return np.load(path) if path.exists() else np.zeros(shape)

    bell_grid = _load_grid(f"bell_operator_grid_{grid_name}.npy")
    uncertainty_grid = _load_grid(f"uncertainty_grid_{grid_name}.npy")
    concurrence_grid = _load_grid(f"concurrence_grid_{grid_name}.npy")
    unphysicality_grid = _load_grid(f"unphysicality_grid_{grid_name}.npy")

    params_path = processed_dir / f"optimal_params_grid_{grid_name}.npy"
    optimal_params_grid = np.load(params_path) if params_path.exists() else np.zeros((12, *shape))

    if not plot_only:
        start = tuple(start_region) if start_region is not None else (0, 0)
        for key in regions:
            if key < start:
                print(f"Skipping region {key} (before start {start})")
                continue
            time_start = time.time()
            result = _process_region(
                key, spec, raw_dir, regions,
                calc_bell=True, calc_concurrence=True, projection=projection,
            )
            time_end = time.time()
            print(f"Processed region {key} in {time_end - time_start:.2f} seconds.")
            if result is None:
                continue
            i, j = key
            bell_grid[i, j] = result['bell_value']
            uncertainty_grid[i, j] = result['uncertainty_bell']
            concurrence_grid[i, j] = result['concurrence_val']
            unphysicality_grid[i, j] = result['unphysicality']
            optimal_params_grid[:, i, j] = result['optimal_params']

            np.save(processed_dir / f"bell_operator_grid_{grid_name}.npy", bell_grid)
            np.save(processed_dir / f"concurrence_grid_{grid_name}.npy", concurrence_grid)
            np.save(processed_dir / f"uncertainty_grid_{grid_name}.npy", uncertainty_grid)
            np.save(processed_dir / f"unphysicality_grid_{grid_name}.npy", unphysicality_grid)
            np.save(processed_dir / f"optimal_params_grid_{grid_name}.npy", optimal_params_grid)

    bell_grid = np.load(processed_dir / f"bell_operator_grid_{grid_name}.npy")
    concurrence_grid = np.load(processed_dir / f"concurrence_grid_{grid_name}.npy")
    unphysicality_grid = np.load(processed_dir / f"unphysicality_grid_{grid_name}.npy")

    plot_contour_heatmap(plots_dir, spec, bell_grid, label)
    plot_contour_heatmap(plots_dir, spec, concurrence_grid, label, concurrence=True)
    generate_unphysicality_heatmap(spec, plots_dir, data=unphysicality_grid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diboson entanglement analysis.")
    parser.add_argument("--process", choices=["ZZ", "WW"], required=True,
                        help="Which diboson process to analyse.")
    parser.add_argument("--projection", choices=["raw", "hard", "smooth"], default="hard",
                        help="PSD projection mode: raw (none), hard (clip negatives to 0), "
                             "smooth (gradual shift). Default: hard.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip analysis and replot from already-saved grids.")
    parser.add_argument("--start-region", nargs=2, type=int, metavar=("COS_IDX", "MASS_IDX"),
                        default=None,
                        help="Bin indices (cos_idx, mass_idx) at which to start analysis, "
                             "skipping all earlier bins. "
                             "Example: --start-region 4 2 resumes from [(0.4,0.5),(300,350)].")
    args = parser.parse_args()
    run(args.process, args.projection, args.plot_only, args.start_region)


if __name__ == "__main__":
    main()

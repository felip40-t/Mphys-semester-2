# Changelog

## [Unreleased]

### Added — P1 deduplication (a–d)

- `src/diboson/physics/kinematics.py` (P1-a) — single source of truth for
  `boostinvp`, `rotinvp`, `phistar`, `calc_scattering_angle`, `calc_inv_mass`.
  Previously duplicated byte-for-byte in `ZZ/lorentz_boost_zz.py` and
  `WW/lorentz_boost_ww.py`. Consumers in `ZZ/lhe_reading_ZZ.py`,
  `WW/lhe_reading_WW.py`, and both `lorentz_boost_*.py` now import from this
  module. Cross-track import `from ZZ.lorentz_boost_zz import ...` in
  `WW/lhe_reading_WW.py` removed.
- `src/diboson/physics/projectors.py` (P1-c) — single source of truth for
  `projector_1..8`, `projector_vector`, `plus_minus`, and `read_masked_data`.
  Both `coefficient_calculator_*.py` files now import these. Process-specific
  scaling (`a_matrix` and the `0.5` factor in ZZ; `wp/wm/double_indices` in WW)
  remains in the per-process files.
- `src/diboson/plotting/contour.py` (P1-d) — single `plot_contour_heatmap`
  parameterised by `process_label` ("ZZ" or "WW"). Both `main_bell_*.py` retain
  thin per-process wrappers for callers; the previous ~80-line bodies are gone.

### Removed — P1 deduplication (b)

- Dead matrix-form helpers in `ZZ/lorentz_boost_zz.py`: `lorentz_boost`,
  `find_beta`, `execute_boost`, `azimuthal_angle`, `rotation_matrix`,
  `azimuthal_angle2`, `calc_polar_angle`. None had any callers across the
  codebase. The chosen boost path is `boostinvp` (now in
  `diboson.physics.kinematics`).

### Changed — config bootstrap (P1)

- `config.py` now also prepends `src/` onto `sys.path` at import time, so legacy
  scripts can resolve `from diboson.x import ...` until P3 introduces a
  `pyproject.toml`. Each consumer that imports both modules has been reordered
  so `from config import ...` precedes `from diboson.* import ...`.

### Removed — WZ track

- Deleted `WZ/` directory entirely (`lhe_reading_WZ.py`, `lorentz_boost_wz.py`,
  `wz_fracs_calc.py`, `wz_theta_hist.py`, `plot_histo_fortran_wz.py`). The WZ
  process was not included in the final analysis and the track was incomplete
  (no `coefficient_calculator_WZ.py`, no `main_bell_WZ.py`).
- Removed `WZ_PROCESS_DIR`, `WZ_FORTRAN_REF_DIR`, `WZ_DATA_DIR` from `config.py`.

### Changed — deliberate numerical corrections (re-baseline golden references before P1)

#### `ZZ/zz_params_calc.py` — operator-precedence fix in `gamma2020_uncertainty`

- **Before:** `(5 / 16*np.pi) * sigma_cos_sqr_mix`
  Python evaluates this as `(5/16) * pi ≈ 0.9817`, which is wrong.
- **After:** `(5 / (16*np.pi)) * sigma_cos_sqr_mix`
  Correct value: `5/(16π) ≈ 0.0995`.
- **Impact:** `gamma2020_uncertainty` was overestimated by a factor of `(16π)²/25 ≈ 101`.
  Any previously saved `gamma2020_uncertainty` values must be re-generated.

#### `core/Bell_inequality_optimizer.py` — `seed=0` added to `differential_evolution`

- **Before:** `differential_evolution(..., workers=-1)` — non-reproducible; each run
  produced a different Bell-violation value within optimiser tolerance.
- **After:** `bell_inequality_optimization(..., seed=0)` — deterministic by default.
  Pass `seed=None` to restore the previous non-reproducible behaviour.
- **Impact:** All Bell heatmap grids must be re-generated with the new seed to serve
  as golden references for regression tests. Results differ from old runs by at most
  the optimiser tolerance (≲1e-4 relative).

### Added

- `config.py` — centralised path and physics constants; replaces 26+ hardcoded
  `/home/felipetcach/project/MG5_aMC_v3_5_6/...` occurrences across 19 files.
  Override the MadGraph install directory via `export MG5_INSTALL_DIR=/your/path`.
- `src/diboson/io/run_discovery.py` — extracted `find_latest_run_dir` function,
  previously duplicated across `WW/lhe_reading_WW.py` and imported illegally by
  `ZZ/lorentz_boost_zz.py` and `utils/histo_plotter.py`.
- `utils/` shim — compatibility bridge; re-exposes `read_data` at the old import
  path until all callers migrate to `src/diboson/` in P1.

### Removed

- Hardcoded `/home/felipetcach/...` path literals across ZZ, WW, WZ, utils,
  event_gen.
- Dead cross-track imports: `from WW.lhe_reading_WW import find_latest_run_dir`
  in `ZZ/lorentz_boost_zz.py`, `WW/lorentz_boost_ww.py`, `WZ/lorentz_boost_wz.py`.
- Debugger leftover `from pdb import run` in `ZZ/lorentz_boost_zz.py`.

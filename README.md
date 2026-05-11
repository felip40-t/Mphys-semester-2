# Testing Quantum Entanglement at the LHC with Electroweak Boson Pairs

Code accompanying the Master's project:

**"Testing Quantum Entanglement at the LHC with Electroweak Boson Pairs"**  
Felipe Tcach  
University of Manchester  

---

## Overview

This repository contains the analysis framework used to reconstruct spin density matrices of electroweak boson pairs (ZZ and W⁺W⁻) produced in proton–proton collisions at √s = 13 TeV, and to evaluate quantum entanglement observables. Full project report available in this repo (University of Manchester MPhys, 2025).

The analysis pipeline:

1. Generate Monte Carlo events (MadGraph5, LO)
2. Extract lepton angular distributions in the modified helicity basis
3. Reconstruct the bipartite qutrit density matrix
4. Enforce physicality (PSD projection)
5. Compute:
   - Lower bound on concurrence
   - Bell operator expectation value (CGLMP inequality)
6. Evaluate observables across phase space (M_VV, cosΘ)

This work probes entanglement in high-energy physics systems and tests Bell-type inequalities in spin-1 (qutrit) systems.

---

## Setup

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib pylhe
```

Always activate the virtual environment before running any scripts:

```bash
source .venv/bin/activate
```

### MadGraph5 path

The event-generation scripts call MadGraph5_aMC@NLO. Set the install location via an environment variable before running any `make events` target:

```bash
export MG5_INSTALL_DIR=/path/to/your/MG5_aMC_v3_5_6
```

If the variable is not set, the code falls back to the default path in `src/diboson/config.py`.

---

## Running the Analysis

### Full pipeline (recommended via Makefile)

```bash
make events        # Generate LHE events for ZZ and WW via MadGraph5
make parse         # Parse LHE files; write per-event kinematics to outputs/data/raw/
make analyse       # Bin events, extract coefficients, compute Bell/concurrence, save plots
```

Process-specific targets:

```bash
make events-zz     make events-ww
make parse-zz      make parse-ww
make analyse-zz    make analyse-ww
```

Optional overrides:

```bash
make events NEVENTS=125000
make events NEVENTS=1000000 WHOLE_PHASE_SPACE=1   # Single uncut run instead of per-bin runs
```

### Parsing options

The parser (`src/diboson/io/parse_lhe.py`) processes MadGraph `run_01`, `run_02`, ... sub-directories sequentially and concatenates all events into a single set of `.npy` files. Several flags control which runs are included and whether existing output is overwritten or extended.

Pass flags to `make parse` via `PARSE_FLAGS`:

```bash
# Parse all runs (default — overwrites existing output)
make parse-zz

# Parse from run 5 onwards
make parse-zz PARSE_FLAGS="--run-start 5"

# Parse a specific range of runs
make parse-zz PARSE_FLAGS="--run-start 3 --run-end 8"

# Append new runs onto existing output (does not overwrite existing .npy files)
make parse-zz PARSE_FLAGS="--run-start 11 --append"

# Parse from a non-default events directory (e.g. a test run)
make parse-zz PARSE_FLAGS="--events-dir /path/to/Events --output-dir outputs/data/raw/ZZ/tests"
```

Full list of `parse_lhe.py` CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--process ZZ\|WW` | required | Diboson process |
| `--run-start N` | 1 | First run index to include (inclusive) |
| `--run-end N` | all | Last run index to include (inclusive) |
| `--append` | off | Concatenate onto existing `.npy` files instead of overwriting |
| `--events-dir PATH` | from config | Directory containing `run_NN` sub-directories |
| `--output-dir PATH` | from config | Directory where `.npy` files are written |
| `--batch-size N` | 100,000 | Events held in memory before each disk flush |

**Typical incremental workflow** — if new runs were generated after a previous parse:

```bash
# First parse: runs 1–10, results written to outputs/data/raw/ZZ/
make parse-zz PARSE_FLAGS="--run-end 10"

# Later: runs 11–20 available; append without re-processing runs 1–10
make parse-zz PARSE_FLAGS="--run-start 11 --append"
```

### Running scripts directly

```bash
# Activate environment first
source .venv/bin/activate
export PYTHONPATH=src

python src/diboson/event_gen/automate.py --process ZZ   # event generation
python src/diboson/io/parse_lhe.py --process ZZ         # LHE parsing
python src/diboson/main.py --process ZZ                 # full analysis + plots
python src/diboson/main.py --process ZZ --raw           # skip PSD projection
```

### Output layout

```
outputs/
├── data/raw/ZZ|WW/        # per-bin .npy files from parse_lhe
├── data/processed/ZZ|WW/  # coefficient CSVs, Bell/concurrence grids
└── plots/ZZ|WW/           # PDF/PNG figures
```

All directories are created automatically when `src/diboson/config.py` is imported.

---

## Validating the Installation

After generating events with `WHOLE_PHASE_SPACE=1` and parsing them to `outputs/data/raw/ZZ/tests/`, run the validation script to check that the event generation settings and kinematics calculations are correct:

```bash
make events-zz WHOLE_PHASE_SPACE=1
python src/diboson/io/parse_lhe.py --process ZZ --output-dir outputs/data/raw/ZZ/tests
python src/diboson/analysis/validate.py
```

The script (`src/diboson/analysis/validate.py`) performs two checks for the ZZ system:

1. **Angular distributions** — plots normalised histograms of cos θ₁, cos θ₃, φ₁, φ₃ to `outputs/plots/tests/`. The cos θ distributions should be roughly flat (small anisotropy); the φ distributions should be uniform.

2. **Coefficient comparison** — computes the whole-phase-space angular coefficients (A₁₁₀, A₁₂₀, A₃₁₀, A₃₂₀, A₁₂₋₂, A₃₂₋₂, g₁₀₁₀, g₂₀₂₀) from spherical harmonic projections and prints them alongside the literature values from the ATLAS analysis. Typical agreement is at the per-cent level.

If the coefficients deviate significantly from the literature values, check the MadGraph run card settings (centre-of-mass energy, phase space cuts) and the kinematic conventions in `src/diboson/physics/kinematics.py`.

---

## Code Structure

All shared logic lives in `src/diboson/`. The `ZZ/` and `WW/` top-level directories are legacy and are not part of the current pipeline.

```
src/diboson/
├── config.py                    # paths, physics constants, bin definitions
├── main.py                      # entry point: loops over bins, saves grids and plots
│
├── physics/
│   ├── density_matrix.py        # ITO/Gell-Mann basis, PSD projection, Bell operator
│   ├── coefficients.py          # spherical harmonic projections (ZZ: AC; WW: fgh)
│   ├── bell_optimiser.py        # differential evolution over U(3)×U(3)
│   ├── concurrence.py           # concurrence lower bound from partial traces
│   ├── kinematics.py            # Lorentz boost, decay angles, invariant mass
│   ├── projectors.py            # Gell-Mann projectors for WW extraction
│   └── unitary_matrix.py        # general U(3) from Euler angles + phases
│
├── analysis/
│   ├── region_analysis.py       # ProcessSpec dataclass; per-bin analysis driver
│   └── validate.py              # whole-phase-space validation against literature
│
├── io/
│   └── parse_lhe.py             # reads LHE files, boosts to CM frame, calculates decay angles, writes .npy
│
├── event_gen/
│   └── automate.py              # unified MadGraph5 driver (--process ZZ|WW)
│
└── plotting/
    ├── contour.py               # Gaussian-smoothed 2D heatmaps
    └── style.py                 # figure size and font constants
```

### `physics/`

#### `density_matrix.py`
Defines all operator bases and constructs the 9×9 bipartite qutrit density matrix.

- `T1_operators`, `T2_operators` — rank-1 and rank-2 ITO basis
- `lambda_operators` — Gell-Mann basis matrices (λ₁–λ₈)
- `O_bell_prime1` — CGLMP Bell operator for qutrit systems
- `calculate_density_matrix_AC(A, C)` — builds ρ from spherical harmonic projections (ITO parameterisation, ZZ)
- `calculate_density_matrix_fgh(f, g, h)` — builds ρ from Gell-Mann coefficients (WW)
- `project_to_psd(rho, const)` — clips negative eigenvalues via Higham-style projection, then renormalises
- `unphysicality_score(rho)` — sum of absolute negative eigenvalues normalised by largest eigenvalue

#### `coefficients.py`
Extracts density matrix coefficients from lepton angular distributions.

- `calculate_coefficients_AC` / `calculate_variance_AC` — spherical harmonic projections for ZZ (ITO)
- `calculate_coefficients_fgh` / `calculate_variance_fgh` — Gell-Mann projector functions for WW
- Both include variance propagation via the full per-event covariance matrix

#### `bell_optimiser.py`
Maximises the CGLMP Bell inequality expectation value over all local unitary rotations.

- `bell_inequality_optimization(rho, O_bell_prime)` — global optimisation over 12 Euler angle parameters using `scipy.differential_evolution` with parallel workers

#### `kinematics.py`
All functions operate on batched `(N, 4)` arrays ordered `(E, px, py, pz)`.

- `lorentz_boost` — boosts a 4-momentum into a target frame
- `calc_decay_angles` — computes θ, φ of a daughter lepton in the boson rest frame (modified helicity convention)
- `calc_scattering_angle` — cosΘ of the boson in the pp CM frame
- `calc_inv_mass` — invariant mass M_VV of the boson pair

#### `unitary_matrix.py`
- `euler_unitary_matrix(θ₁₂, θ₁₃, θ₂₃, δ, α₁, α₂)` — general U(3) matrix from three mixing angles and three phases, following the PMNS/CKM parameterisation

### `analysis/`

#### `region_analysis.py`
- `ProcessSpec` dataclass — captures all ZZ/WW differences (coefficient functions, bin counts, ETA correction, output directories)
- `process_region` — full per-bin pipeline: reads `.npy` files, extracts coefficients, builds ρ, applies PSD projection, computes Bell value and concurrence
- Heatmap generators for event counts, uniformity, and unphysicality

#### `validate.py`
Standalone validation script for checking MadGraph installation and kinematics. Reads whole-phase-space data from `outputs/data/raw/ZZ/tests/` and:
- Plots normalised angular distributions (cos θ, φ) for both Z bosons
- Computes whole-phase-space angular coefficients and prints them alongside literature values

Run directly after a whole-phase-space parse:

```bash
python src/diboson/analysis/validate.py
```

Plots are saved to `outputs/plots/tests/`.

### `io/parse_lhe.py`
Reads MadGraph5 LHE event files, applies Lorentz boosts to the diboson CM frame, computes decay angles and kinematic variables, and writes per-event arrays to `.npy` files in `outputs/data/raw/`.

- `ZZ_CONFIG` / `WW_CONFIG` — specify PDG daughter particle IDs for each process
- Processes events in batches to bound peak memory usage
- CLI: `python src/diboson/io/parse_lhe.py --process ZZ|WW`

### `event_gen/automate.py`
Unified MadGraph5 driver. Generates per-bin LHE event files for ZZ or WW over the full (M_VV, cosΘ) phase-space grid, or a single uncut run when `--whole-phase-space` is passed.

```bash
python src/diboson/event_gen/automate.py --process ZZ [--nevents N] [--whole-phase-space]
```

---

## MadGraph5 Installation

### 1. Download and unpack

```bash
wget https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5_aMC_v3.7.0.tar.gz
tar -xzf MG5_aMC_v3.7.0.tar.gz
```

Place the unpacked directory wherever you prefer (e.g. `~/MG5_aMC/`).

### 2. Dependencies

```bash
# Debian/Ubuntu
sudo apt install gfortran g++ python3 python3-six
```

### 3. Verify the installation

```bash
cd MG5_aMC
python3 bin/mg5_aMC
```

You should see the `MG5_aMC>` prompt. Exit with `exit`.

### 4. Set the install path

```bash
export MG5_INSTALL_DIR=/path/to/MG5_aMC
```

Or edit the default directly in `src/diboson/config.py`:

```python
MG5_INSTALL_DIR = Path("/path/to/MG5_aMC")
```

The event-generation script creates the process directories (`pp_ZZ`, `pp_WW`) inside `MG5_INSTALL_DIR` on first run. No manual MadGraph setup is required.

---

## Physics Scope

This project investigates quantum entanglement in bipartite systems of electroweak bosons — specifically ZZ and W⁺W⁻ pairs produced at the LHC via pp collisions at √s = 13 TeV. Monte Carlo events are generated with MadGraph5 and the spin density matrices are reconstructed from the angular distributions of fully leptonic decay products.

### Density Matrix Reconstruction

The spin density matrix ρ encodes the full quantum state of the diboson system. Two parametrisations are used:

- **ITO (Irreducible Tensor Operator) parametrisation** for ZZ, expanding ρ in rank-l tensor operators Tˡₘ:

$$\rho = \frac{1}{9}\left[\mathbf{1}_3 \otimes \mathbf{1}_3 + \sum A^{(1)}_{l,m} T^l_m \otimes \mathbf{1}_3 + \sum A^{(3)}_{l,m} \mathbf{1}_3 \otimes T^l_m + \sum C_{l_1,m_1,l_3,m_3} T^{l_1}_{m_1} \otimes T^{l_3}_{m_3}\right]$$

- **Gell-Mann basis** for W⁺W⁻, expanding ρ in the SU(3) generators λᵢ:

$$\rho = \frac{1}{9}\mathbf{1}_3\otimes\mathbf{1}_3 + \frac{1}{3}\sum_i f_i\,\lambda_i\otimes\mathbf{1}_3 + \frac{1}{3}\sum_j g_j\,\mathbf{1}_3\otimes\lambda_j + \sum_{i,j} h_{ij}\,\lambda_i\otimes\lambda_j$$

Coefficients are extracted as expectation values of spherical harmonics (ITO) or projector functions πᵢ± (Gell-Mann) over the lepton decay angles, defined in the modified helicity frame of the ATLAS collaboration.

### Quantum Observables

**Concurrence lower bound** — quantifies entanglement in mixed bipartite states:

$$\mathcal{C}^2_{LB} = 2\max\!\left(0,\,\mathrm{Tr}\{\rho^2\} - \mathrm{Tr}\{\rho_A^2\},\,\mathrm{Tr}\{\rho^2\} - \mathrm{Tr}\{\rho_B^2\}\right)$$

A non-zero value certifies entanglement.

**CGLMP Bell inequality** — generalises the CHSH inequality to qutrit (spin-1) systems:

$$\mathcal{I}_3 = \langle\mathcal{O}_B\rangle = \mathrm{Tr}\{\rho\,\mathcal{O}_B\} \leq 2$$

The Bell operator is optimised over unitary rotations U, V ∈ U(3) to maximise violation. Its initial form is:

$$\mathcal{O}'_B = -\frac{2}{\sqrt{3}}\left(S_x\otimes S_x + S_y\otimes S_y\right) + \lambda_4\otimes\lambda_4 + \lambda_5\otimes\lambda_5$$

### Phase Space

Observables are mapped across the diboson kinematic plane (M_VV, cosΘ):

- cosΘ: 0.0–1.0 in 10 bins of width 0.1
- M_ZZ: 200–1000 GeV in 16 bins of width 50 GeV
- M_WW: 200–1200 GeV in 20 bins of width 50 GeV

Reconstructed density matrices are projected to the nearest positive semi-definite state via Higham projection when negative eigenvalues arise. Uncertainties on Bell operator values are propagated via the full covariance matrix of density matrix coefficients.

# Testing Quantum Entanglement at the LHC with Electroweak Boson Pairs

Code accompanying the Master's project:

**"Testing Quantum Entanglement at the LHC with Electroweak Boson Pairs"**  
Felipe Tcach  
University of Manchester  

---

## Overview

This repository contains the analysis framework used to reconstruct spin density matrices of electroweak boson pairs (ZZ and W⁺W⁻) produced in proton–proton collisions at √s = 13 TeV, and to evaluate quantum entanglement observables. Full project report available in this repo (University of Manchester MPhys, 2025). **Note:** The results shown in the outputs/ directory were generated from a different dataset than the one that was used to make the results section of the report attached.

The analysis pipeline:

1. Generate Monte Carlo events (MadGraph5, LO)
2. Extract lepton angular distributions in the modified helicity basis
3. Reconstruct the bipartite qutrit density matrix
4. Enforce physicality (PSD projection)
5. Compute:
   - Lower bound on concurrence
   - Bell operator expectation value (CGLMP inequality)
6. Evaluate observables across phase space ($M_{VV}$, cos$\Theta$)

This work probes entanglement in high-energy physics systems and tests Bell-type inequalities in spin-1 (qutrit) systems.

---

## Setup

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

Process-specific targets run only one process:

```bash
make events-zz     make events-ww
make parse-zz      make parse-ww
make analyse-zz    make analyse-ww
```

### Makefile modifier targets and variable overrides

The Makefile accepts **modifier words** appended to the target name and **variable overrides** set as `KEY=VALUE`. These can be combined freely.

#### Modifier words

| Modifier | Applies to | Effect |
|---|---|---|
| `whole-phase-space` | `events-*`, `parse-*` | Generates or parses a single uncut whole-phase-space run instead of per-bin runs |
| `append` | `parse-*` | Concatenates onto existing `.npy` files instead of overwriting; only valid with `whole-phase-space` |
| `raw` | `analyse-*` | No PSD projection; density matrix used as-is |
| `hard` | `analyse-*` | Hard cutoff: negative eigenvalues clipped to 0 (default if no projection modifier is given) |
| `smooth` | `analyse-*` | Smooth projection: negative eigenvalues shifted gradually towards 0 via a gradual shift function |
| `plot-only` | `analyse-*` | Skips the analysis computation and replots from already-saved grids |

#### Variable overrides

| Variable | Applies to | Effect |
|---|---|---|
| `NEVENTS=N` | `events-*` | Number of events to generate per run |
| `OUTPUT_DIR=path` | `parse-*` | Directory where `.npy` files are written (overrides the default from `config.py`) |
| `START_REGION="COS_IDX MASS_IDX"` | `events-*` | Resume binned generation from a specific bin, skipping all earlier ones (see below) |

#### Examples

```bash
# Generate 50,000 events per bin for ZZ
make events-zz NEVENTS=50000

# Generate a single whole-phase-space run with 500,000 events
make events-zz whole-phase-space NEVENTS=500000

# Parse a whole-phase-space run into the validation directory
make parse-zz whole-phase-space OUTPUT_DIR=outputs/data/raw/ZZ/tests

# Append a new whole-phase-space parse onto existing output
make parse-zz whole-phase-space append OUTPUT_DIR=outputs/data/raw/ZZ/tests

# Analyse without PSD projection
make analyse-zz raw

# Analyse with hard cutoff projection (default)
make analyse-zz hard

# Analyse with smooth (gradual shift) projection
make analyse-zz smooth

# Replot from saved grids without rerunning the analysis
make analyse-zz plot-only

# Combine raw and plot-only
make analyse-zz raw plot-only

# Resume ZZ event generation from bin (cos_idx=4, mass_idx=2), i.e. cos∈[0.4,0.5], M∈[300,350] GeV
make events-zz START_REGION="4 2"
```

#### Resuming interrupted event generation with `START_REGION`

Binned event generation runs one MadGraph job per phase-space bin (up to 160 bins for ZZ). If a run is interrupted, use `START_REGION` to pick up where it left off rather than regenerating all earlier bins.

The two indices are:
- `COS_IDX` — integer index of the cosΘ bin: `COS_IDX = int((cos_lo - 0.0) / 0.1)`
- `MASS_IDX` — integer index of the invariant mass bin: `MASS_IDX = int((mass_lo - 200) / 50)`
However, this specific example is for widths of 0.1 and 50 GeV and minimum values of 0.0 and 200 for the scattering angle and invariant mass of the bosons (respectively). 
If the values defining your phase space are different then you would need to adjust this indexing convention.

For example, bin `[cos∈(0.4, 0.5), M∈(300, 350) GeV]` has `COS_IDX=4, MASS_IDX=2`.

```bash
# Resume ZZ generation from cos∈[0.4,0.5], M∈[300,350] GeV onward
make events-zz START_REGION="4 2"

# Same but with a custom event count
make events-zz START_REGION="4 2" NEVENTS=50000
```

All bins that come before `(COS_IDX, MASS_IDX)` in the natural `(i, j)` ordering are skipped with a printed message. The Fortran phase-space cut file is updated correctly for each resumed bin.

### Parsing options (direct script invocation)

When running `parse_lhe.py` directly (outside of `make`), additional flags are available that the Makefile does not expose, notably `--run-start` and `--run-end` for incremental parsing of specific run ranges.

Full list of `parse_lhe.py` CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--process ZZ\|WW` | required | Diboson process |
| `--run-start N` | 2 | First run index to include (inclusive); default is 2 since the first run in the directory is usually the initial startup run, which is not a part of the analysis, since it has no phase-space cuts applied. |
| `--run-end N` | all | Last run index to include (inclusive) |
| `--append` | off | Concatenate onto existing `.npy` files instead of overwriting; only valid with `--whole-phase-space` |
| `--whole-phase-space` | off | Write all events to a single flat directory instead of per-region subdirectories; use for validation runs |
| `--events-dir PATH` | from config | Directory containing `run_NN` sub-directories |
| `--output-dir PATH` | from config | Directory where `.npy` files are written |
| `--batch-size N` | 100,000 | Events held in memory before each disk flush |

**Typical incremental workflow** — if new runs were generated after a previous parse:

```bash
source .venv/bin/activate
export PYTHONPATH=src

# First parse: runs 1–10
python src/diboson/io/parse_lhe.py --process ZZ --run-end 10

# Later: runs 11–20 available; append without re-processing runs 1–10
python src/diboson/io/parse_lhe.py --process ZZ --run-start 11 --append
```

### Running scripts directly

```bash
# Activate environment first
source .venv/bin/activate
export PYTHONPATH=src

python src/diboson/event_gen/automate.py --process ZZ                          # event generation
python src/diboson/event_gen/automate.py --process ZZ --nevents 50000          # with event count
python src/diboson/event_gen/automate.py --process ZZ --whole-phase-space      # single uncut run
python src/diboson/event_gen/automate.py --process ZZ --start-region 4 2       # resume from bin (4,2)

python src/diboson/io/parse_lhe.py --process ZZ           # LHE parsing (binned)
python src/diboson/io/parse_lhe.py --process ZZ --whole-phase-space --output-dir outputs/data/raw/ZZ/tests

python src/diboson/main.py --process ZZ                             # full analysis + plots (hard projection)
python src/diboson/main.py --process ZZ --projection raw            # no PSD projection
python src/diboson/main.py --process ZZ --projection hard           # hard cutoff (default)
python src/diboson/main.py --process ZZ --projection smooth         # gradual shift projection
python src/diboson/main.py --process ZZ --plot-only                 # replot from saved grids
```

### Output layout

```
outputs/
├── data/raw/ZZ|WW/        # per-bin .npy files from parse_lhe
├── data/processed/ZZ|WW/  # coefficient CSVs; Bell, uncertainty, concurrence, unphysicality, and optimal-params grids
└── plots/ZZ|WW/           # PDF/PNG figures
```

All directories are created automatically when `src/diboson/config.py` is imported.

---

## Validating the Installation

After generating events with `WHOLE_PHASE_SPACE=1` and parsing them to `outputs/data/raw/ZZ/tests/`, run the validation script to check that the event generation settings and kinematics calculations are correct:

```bash
# Step 1: generate a single uncut whole-phase-space run
make events-zz whole-phase-space

# Step 2: parse into the validation directory
make parse-zz whole-phase-space OUTPUT_DIR=outputs/data/raw/ZZ/tests

# Step 3: run validation
source .venv/bin/activate
export PYTHONPATH=src
python src/diboson/analysis/validate.py
```

The script (`src/diboson/analysis/validate.py`) performs two checks for the ZZ system:

1. **Angular distributions** — plots normalised histograms of cos θ₁, cos θ₃, φ₁, φ₃ to `outputs/plots/tests/`.

2. **Coefficient comparison** — computes the whole-phase-space angular coefficients (A₁₁₀, A₁₂₀, A₃₁₀, A₃₂₀, A₁₂₋₂, A₃₂₋₂, g₁₀₁₀, g₂₀₂₀) from spherical harmonic projections and prints them alongside the expected values obtained during the work conducted for this Master's project, as shown in the report pdf file. Typical agreement is within one standard deviation.

If the coefficients deviate significantly from the literature values, check the MadGraph settings in the run card and param card.

---

## Code Structure

All shared logic lives in `src/diboson/`.

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
- `unphysicality_score(rho)` — sum of absolute negative eigenvalues

#### `coefficients.py`
Extracts density matrix coefficients from lepton angular distributions.

- `calculate_coefficients_AC` / `calculate_variance_AC` — spherical harmonic projections for ZZ (ITO)
- `calculate_coefficients_fgh` / `calculate_variance_fgh` — Gell-Mann projector functions for WW
- Both include variance propagation for the bell operator via the full per-event covariance matrix

#### `bell_optimiser.py`
Maximises the CGLMP Bell inequality expectation value over all local unitary rotations U, V ∈ U(3).

**Tensor factorisation.** The 9×9 density matrix ρ and Bell operator O'_B are each reshaped to rank-4 tensors of shape `(3, 3, 3, 3)` before any computation. A composite 9-dimensional index I encodes two qutrit states as `I = 3a + b`, so the reshape decomposes each row/column index into individual single-particle indices `(a, b)`. The expectation value then reduces to a pure index contraction over eight 3-valued indices:

$$
\mathcal{I}_3 = \sum_{a,b,c,d,\,i,j,k,l}\, \rho_{ab,cd}\; U^*_{ic}\; V^*_{jd}\; [\mathcal{O}'_B]_{ij,kl}\; U_{ka}\; V_{lb}
$$

This avoids constructing the full 9×9 Kronecker product U⊗V at any point. The optimal contraction order is derived once per call via `numpy.einsum_path` and reused on every function evaluation.

**Optimisation.** The 12 free parameters — three Euler angles and three phases per unitary (see `unitary_matrix.py`) — are optimised with multistart L-BFGS-B. Forty random starting points are drawn uniformly from [0, 2π]¹² and each run to convergence independently; the global maximum across all starts is returned. Starts are dispatched in parallel via `ThreadPoolExecutor`.

- `bell_inequality_optimization(rho, O_bell_prime)` — returns `(best_value, best_params)`
- `optimal_bell_operator(O_bell_prime, params)` — reconstructs the rotated Bell operator from stored parameters

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
python src/diboson/event_gen/automate.py --process ZZ [--nevents N] [--whole-phase-space] [--start-region COS_IDX MASS_IDX]
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

### 5. Process and card setup

Before the automated pipeline can run, the process must be defined inside the MadGraph5 interactive shell and the physics and run cards must be configured. Launch the shell and enter the commands below for ZZ:

```bash
define p = g u c d s u~ c~ d~ s~ b b~
define j = g u c d s u~ c~ d~ s~ b b~
generate p p > e+ e- mu+ mu-
output pp_ZZ
launch
```
It is important to define p and j this way so that we use the 5-flavour scheme (treat all quarks except for the top quark as massless).
The equivalent set of commands for WW would be the same, just switching out two of the charged leptons for neutrinos of the corresponding flavour.
After the process directory has been generated, copy the example cards from this repository into the MadGraph5 process directory to match the settings used in this analysis:

```
setup/
├── ZZ/
│   ├── run_card.dat    # run settings (collider energy, cuts, PDF, scale choices)
│   └── param_card.dat  # SM parameter values
└── WW/
    ├── run_card.dat
    └── param_card.dat
```

Copy the relevant card(s) into the `Cards/` subdirectory of the generated process directory (e.g. `$MG5_INSTALL_DIR/pp_ZZ/Cards/`), overwriting the defaults. The automated pipeline then reads these cards on every run.

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

A non-zero value certifies entanglement. In this study we have plotted the square root of the quantity above, hence the axis label $\mathcal{C}_{LB}$.

**CGLMP Bell inequality** — generalises the CHSH inequality to qutrit (spin-1) systems:

$$\mathcal{I}_3 = \langle\mathcal{O}_B\rangle = \mathrm{Tr}\{\rho\,\mathcal{O}_B\} \leq 2$$

The observable is maximised over local unitary rotations U, V ∈ U(3), giving the optimised value Tr{ρ (U⊗V)† O'_B (U⊗V)}. The base operator is:

$$\mathcal{O}'_B = -\frac{2}{\sqrt{3}}\left(S_x\otimes S_x + S_y\otimes S_y\right) + \lambda_4\otimes\lambda_4 + \lambda_5\otimes\lambda_5$$

Each unitary is parameterised by 12 real angles (PMNS/CKM-style: three mixing angles and three phases per factor). The bipartite Hilbert space structure — two qutrits sharing a 9-dimensional composite space — means ρ and O'_B can be treated as rank-4 tensors of shape (3,3,3,3), with each 9-dimensional index factoring as I = 3a + b. This tensor form allows the expectation value to be evaluated without constructing any 9×9 Kronecker products (see `bell_optimiser.py`).

### Phase Space

Observables are mapped across the diboson kinematic plane (M_VV, cosΘ):

- cosΘ: 0.0–1.0 in 10 bins of width 0.1
- M_ZZ: 200–1000 GeV in 16 bins of width 50 GeV
- M_WW: 200–1000 GeV in 16 bins of width 50 GeV

Reconstructed density matrices are projected to the nearest positive semi-definite state via Higham projection when negative eigenvalues arise. Uncertainties on Bell operator values are propagated via the full covariance matrix of density matrix coefficients.

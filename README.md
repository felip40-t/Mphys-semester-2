# Testing Quantum Entanglement at the LHC with Electroweak Boson Pairs

Code accompanying the Master's project:

**“Testing Quantum Entanglement at the LHC with Electroweak Boson Pairs”**  
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

## Code Structure

The codebase is organised into three parallel analysis tracks — **ZZ**, **WW**, and **WZ** — each following the same pipeline from LHE parsing through to entanglement observables. A set of process-independent core modules handles the quantum state calculations that are shared across all tracks.

### Core Modules

#### `density_matrix_calculator.py`
Defines all operator bases and constructs the 9×9 bipartite qutrit density matrix.

- `T1_operators`, `T2_operators` — rank-1 and rank-2 irreducible tensor operators (ITO basis)
- `lambda_operators` — Gell-Mann basis matrices (λ₁–λ₈)
- `O_bell_prime1` — CGLMP Bell operator for qutrit systems
- `calculate_density_matrix_AC(A_coefficients, C_coefficients)` — builds ρ from spherical harmonic projections (A, C coefficients) via the ITO parameterisation
- `calculate_density_matrix_fgh(f, g, h)` — builds ρ from Gell-Mann basis coefficients (f, g, h)
- `project_to_psd(rho, const)` — enforces physicality by clipping negative eigenvalues via a smooth exponential shift (Higham-style projection), then renormalises
- `unphysicality_score(rho)` — returns the sum of absolute negative eigenvalues normalised by the largest eigenvalue; used to quantify how unphysical a raw density matrix is

#### `concurrence_bound.py`
Quantifies bipartite entanglement from the reconstructed density matrix.

- `partial_trace(rho, subsystem)` — computes ρ_A or ρ_B by tracing out subsystem 2 or 1 respectively
- `concurrence_lower(rho)` — returns C²_LB = 2·max(0, Tr{ρ²}−Tr{ρ_A²}, Tr{ρ²}−Tr{ρ_B²})
- `concurrence_MB(f, g, h)` — alternative concurrence estimator directly from Gell-Mann coefficients (WW analysis)
- `check_density_matrix(rho)` — validates Hermiticity, unit trace, and positive semi-definiteness; prints diagnostics

#### `Bell_inequality_optimizer.py`
Maximises the CGLMP Bell inequality expectation value over all local unitary rotations.

- `optimal_bell_operator(O_bell_prime, parameters)` — applies local U⊗V rotation to the Bell operator: O_bell = (U⊗V)† O' (U⊗V)
- `bell_inequality_optimization(rho, O_bell_prime)` — global optimisation over 12 Euler angle parameters (6 per subsystem) using `scipy.differential_evolution` with parallel workers; returns the maximum Bell value and optimal parameters

#### `Unitary_Matrix.py`
- `euler_unitary_matrix(θ₁₂, θ₁₃, θ₂₃, δ, α₁, α₂)` — constructs a general 3×3 U(3) matrix from three mixing angles and three phases, following the PMNS/CKM parameterisation with an additional diagonal phase matrix

---

### Per-Process Modules

Each track (ZZ/WW/WZ) has equivalent modules at each pipeline stage:

#### LHE Parsing — `lhe_reading_ZZ.py`, `lhe_reading_WW.py`, `lhe_reading_WZ.py`
Read MadGraph5 LHE event files using `pylhe`, extract lepton 4-momenta, and organise events by particle type.

#### Lorentz Boosts — `lorentz_boost_zz.py`, `lorentz_boost_ww.py`, `lorentz_boost_wz.py`
Transform particle momenta to the boson pair centre-of-mass frame and compute the kinematic variables used for binning and angular analysis.

- `boostinvp(p, boost)` — performs a Lorentz boost on a 4-momentum vector
- `calc_inv_mass(p1, p2)` — computes invariant mass M_VV of the boson pair
- `calc_scattering_angle(p)` — extracts cosΘ of the boson in the pp CM frame
- `phistar(p1, p2)` — computes the modified azimuthal angle φ* in the helicity basis

#### Coefficient Calculators — `coefficient_calculator_ZZ.py`, `coefficient_calculator_WW.py`
Extract density matrix coefficients from lepton angular distributions by projecting onto the appropriate basis functions.

- `calculate_coefficients_AC(theta_paths, phi_paths)` — projects onto spherical harmonics Y^m_l to extract A (single-boson) and C (bipartite) coefficients; used for the ITO parameterisation (ZZ)
- `calculate_coefficients_fgh(theta_paths, phi_paths)` — projects onto the 8 Gell-Mann projector functions to extract f, g (single-boson) and h (bipartite correlation) coefficients; used for the Gell-Mann parameterisation (WW)
- `calculate_variance_AC` / `calculate_variance_fgh` — propagate statistical uncertainties on the Bell operator expectation value via a covariance matrix built from the per-event projector values
- `read_masked_data(cos_psi, inv_mass, psi_range, mass_range)` — returns a boolean mask selecting events in a given (cosΘ, M_VV) phase-space bin

#### Main Analysis Scripts — `main_bell_ZZ.py`, `main_bell_WW.py`
Orchestrate the full pipeline over a 2D phase-space grid (M_VV × cosΘ). For each bin they reconstruct ρ, apply PSD projection, compute the concurrence lower bound and Bell inequality value with uncertainties, and produce Gaussian-smoothed 2D heatmap plots.

---

## MadGraph5 Installation

MadGraph5_aMC@NLO (v3.5.6) is required for Monte Carlo event generation. The scripts in this repository call MadGraph5 via hardcoded absolute paths, so the install location must be noted and updated accordingly.

### 1. Download and unpack

```bash
wget https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5_aMC_v3.5.6.tar.gz
tar -xzf MG5_aMC_v3.5.6.tar.gz
```

Place the unpacked directory wherever you prefer (e.g. `~/MG5_aMC_v3_5_6/`).

### 2. Dependencies

MadGraph5 requires Python 3 and the following system packages:

```bash
# Debian/Ubuntu
sudo apt install gfortran g++ python3 python3-six
```

Optional but recommended for LO+PS runs: install `pythia8` and `lhapdf6` from within the MadGraph5 shell (see step 4).

### 3. Verify the installation

```bash
cd MG5_aMC_v3_5_6
python3 bin/mg5_aMC
```

You should see the `MG5_aMC>` prompt.

### 4. Generate the processes used in this project

From the MadGraph5 interactive shell:

```
# ZZ → 4ℓ
generate p p > e+ e- mu+ mu-
output ZZ_4l
launch ZZ_4l

# W⁺W⁻ → ℓνℓν
generate p p > e+ ve mu- vm~
output WW_lvlv
launch WW_lvlv
```

Select `LO` (leading order) and use the default `RunCard` settings, or copy the run cards used in previous generation runs (stored under `Events/` in the process directory).

### 5. Update hardcoded paths

After installation, update the MadGraph5 path in the event-generation scripts to match your local install:

```python
# In automate_event_gen.py and automate_event_gen_ww.py, replace:
MG5_PATH = "/home/felipetcach/project/MG5_aMC_v3_5_6/"
# with your actual path, e.g.:
MG5_PATH = "/home/<your_username>/MG5_aMC_v3_5_6/"
```

The generated LHE files (`.lhe` or `.lhe.gz`) are then passed to the `lhe_reading_*.py` scripts to begin the analysis pipeline.

---

## Physics Scope

## Physics Scope

This project investigates quantum entanglement in bipartite systems of electroweak (EW) bosons — specifically ZZ and W⁺W⁻ pairs produced at the LHC via pp collisions at √s = 13 TeV. Monte Carlo events are generated with MadGraph5 and the spin density matrices are reconstructed from the angular distributions of fully leptonic decay products.

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

### Phase Space & Validity

Observables are mapped across the diboson kinematic plane ($$M_VV$$, cos Θ). Reconstructed density matrices are projected to the nearest positive semi-definite state via Higham projection when negative eigenvalues arise. Uncertainties on Bell operator values are propagated via the full covariance matrix of density matrix coefficients.

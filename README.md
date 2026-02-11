# Testing Quantum Entanglement at the LHC with Electroweak Boson Pairs

Code accompanying the Master's project:

**“Testing Quantum Entanglement at the LHC with Electroweak Boson Pairs”**  
Felipe Tcach  
University of Manchester  

---

## Overview

This repository contains the analysis framework used to reconstruct spin density matrices of electroweak boson pairs (ZZ and W⁺W⁻) produced in proton–proton collisions at √s = 13 TeV, and to evaluate quantum entanglement observables.

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

## Physics Scope

Processes studied:

- `pp → e⁺e⁻ μ⁺μ⁻`  (ZZ)
- `pp → e⁺νₑ μ⁻ν̄_μ` (W⁺W⁻)

Key theoretical components implemented:

- Density matrix reconstruction via:
  - Irreducible Tensor Operator (ITO) parametrisation
  - Gell-Mann basis expansion
- Lower bound on concurrence:

  $\mathcal{C}_{LB}^2 = 2 \, \mathrm{max} \, (0, \,\mathrm{Tr}\{\rho^2\} - \mathrm{Tr}\{\rho_A^2\}, \, \mathrm{Tr}\{\rho^2\} - \mathrm{Tr}\{\rho_B^2\})$

- Bell operator for qutrit systems (CGLMP inequality)
- Phase-space dependent entanglement analysis
- Eigenvalue clipping via Higham projection
- Covariance-based uncertainty propagation

---

"""Tensor-operator projector basis used to extract spin density matrix coefficients
from angular distributions. Process-independent: the same projectors are applied to
ZZ and WW (the per-process scaling sits in `calculate_coefficients_fgh`)."""

import numpy as np


def plus_minus(dataset):
    if dataset == 1:
        return +1
    elif dataset == 3:
        return -1


def projector_1(theta, phi, dataset):
    value = np.sqrt(2) * np.sin(theta) * (5 * np.cos(theta) + plus_minus(dataset) * 1) * np.cos(phi)
    return value


def projector_2(theta, phi, dataset):
    value = np.sqrt(2) * np.sin(theta) * (5 * np.cos(theta) + plus_minus(dataset) * 1) * np.sin(phi)
    return value


def projector_3(theta, phi, dataset):
    value = (1/4) * (5 + plus_minus(dataset) * 4 * np.cos(theta) + 15 * np.cos(2*theta))
    return value


def projector_4(theta, phi, dataset):
    return 5 * np.sin(theta)**2 * np.cos(2 * phi)


def projector_5(theta, phi, dataset):
    return 5 * np.sin(theta)**2 * np.sin(2 * phi)


def projector_6(theta, phi, dataset):
    value = np.sqrt(2) * np.sin(theta) * (-5 * np.cos(theta) + plus_minus(dataset) * 1) * np.cos(phi)
    return value


def projector_7(theta, phi, dataset):
    value = np.sqrt(2) * np.sin(theta) * (-5 * np.cos(theta) + plus_minus(dataset) * 1) * np.sin(phi)
    return value


def projector_8(theta, phi, dataset):
    value = (1 / (4 * np.sqrt(3))) * (-5 + plus_minus(dataset) * 12 * np.cos(theta) - 15 * np.cos(2*theta))
    return value


def projector_vector(theta, phi, dataset):
    vector = np.array([
        projector_1(theta, phi, dataset),
        projector_2(theta, phi, dataset),
        projector_3(theta, phi, dataset),
        projector_4(theta, phi, dataset),
        projector_5(theta, phi, dataset),
        projector_6(theta, phi, dataset),
        projector_7(theta, phi, dataset),
        projector_8(theta, phi, dataset)
    ])
    return vector


def read_masked_data(cos_psi_data, inv_mass, psi_range, mass_range):
    """Apply a mask based on psi and diboson invariant mass."""
    return (cos_psi_data > psi_range[0]) & (cos_psi_data < psi_range[1]) & (inv_mass > mass_range[0]) & (inv_mass < mass_range[1])

"""Process-independent relativistic kinematics: Lorentz boosts, rotations, and decay-angle reconstruction.

All functions operate on batched inputs:
  - 4-momenta: (N, 4) arrays ordered (E, px, py, pz)
  - 3-momenta: (N, 3) arrays ordered (px, py, pz)
"""

import numpy as np


def lorentz_boost(p_in, p_frame):
    """
    Lorentz boost a batch of 4-momenta into a new frame.

    Parameters
    ----------
    p_in : (N, 4) array
        4-momenta to boost (E, px, py, pz).
    p_frame : (N, 4) array
        4-momenta defining the target frame.

    Returns
    -------
    p_out : (N, 4) array
    """
    m2 = p_frame[:, 0]**2 - np.sum(p_frame[:, 1:]**2, axis=1)
    m = np.sqrt(np.maximum(m2, 0.0))

    e_boosted = (p_in[:, 0]*p_frame[:, 0] - np.einsum('ij,ij->i', p_in[:, 1:], p_frame[:, 1:])) / m
    proj = (e_boosted + p_in[:, 0]) / (p_frame[:, 0] + m)

    p_out = np.empty_like(p_in)
    p_out[:, 0] = e_boosted
    p_out[:, 1:] = p_in[:, 1:] - proj[:, np.newaxis] * p_frame[:, 1:]
    return p_out


def calc_decay_angles(p1, p2, p3, p4):
    """
    Calculate azimuthal and polar decay angles for a batch of events.

    All angles are in the helicity frame: each boson is boosted to its rest frame
    with the quantisation axis along its CM-frame flight direction n.

    theta -- polar angle between the decay product and n -- is read directly from
    the dot product with n.

    phi -- azimuthal angle around n -- uses a right-handed helicity basis (x, y, n)
    built via two cross products:
        y = (z x n) / |z x n|   (z = beam axis; out-of-scattering-plane)
        x = y x n                (in-scattering-plane, perpendicular to n) - > No need
        to explicitly normalise since it is already unit length by construction. 
    phi = arctan2(p.y, p.x). Projecting directly onto x and y is equivalent to
    first computing p_perp = p - (p.n)n and then measuring its azimuth, because
    x and y are both perpendicular to n so the parallel component projects to zero.

    Parameters
    ----------
    p1, p2 : (N, 4) arrays
        4-momenta of the decay products of boson 1.
    p3, p4 : (N, 4) arrays
        4-momenta of the decay products of boson 2.

    Returns
    -------
    phi1, phi3, theta1, theta3 : (N,) arrays
    """
    p_b1 = p1 + p2
    p_b2 = p3 + p4
    p_tot = p_b1 + p_b2

    p_b1_cm = lorentz_boost(p_b1, p_tot)
    p_b2_cm = lorentz_boost(p_b2, p_tot)
    p1_cm   = lorentz_boost(p1,   p_tot)
    p3_cm   = lorentz_boost(p3,   p_tot)

    p1_rest = lorentz_boost(p1_cm, p_b1_cm)
    p3_rest = lorentz_boost(p3_cm, p_b2_cm)

    # Unit vectors along each boson's flight direction in the CM frame
    n1 = p_b1_cm[:, 1:] / np.linalg.norm(p_b1_cm[:, 1:], axis=1, keepdims=True)
    n3 = p_b2_cm[:, 1:] / np.linalg.norm(p_b2_cm[:, 1:], axis=1, keepdims=True)

    p1_3vec = p1_rest[:, 1:]
    p3_3vec = p3_rest[:, 1:]

    # theta: angle between decay product and boson direction
    cos_t1 = np.einsum('ij,ij->i', p1_3vec, n1) / np.linalg.norm(p1_3vec, axis=1) # use scalar product for angle calculation
    cos_t3 = np.einsum('ij,ij->i', p3_3vec, n3) / np.linalg.norm(p3_3vec, axis=1)
    theta1 = np.arccos(np.clip(cos_t1, -1.0, 1.0))
    theta3 = np.arccos(np.clip(cos_t3, -1.0, 1.0))

    # phi: build right-handed helicity basis (x, y, n) via successive cross products
    n1p = np.sqrt(n1[:, 0]**2 + n1[:, 1]**2)  # transverse magnitude of flight direction
    n3p = np.sqrt(n3[:, 0]**2 + n3[:, 1]**2)
    s1 = np.where(n1p > 0, n1p, 1.0)  # guard: n aligned with beam has no unique perp plane
    s3 = np.where(n3p > 0, n3p, 1.0)

    # y = (z x n) / n1p
    y1 = np.column_stack([-n1[:, 1]/s1,            n1[:, 0]/s1,          np.zeros(len(n1))])
    y3 = np.column_stack([-n3[:, 1]/s3,            n3[:, 0]/s3,          np.zeros(len(n3))])
    # x = y x n
    x1 = np.column_stack([ n1[:, 2]*n1[:, 0]/s1,  n1[:, 2]*n1[:, 1]/s1, -s1])
    x3 = np.column_stack([ n3[:, 2]*n3[:, 0]/s3,  n3[:, 2]*n3[:, 1]/s3, -s3])

    phi1 = np.arctan2(np.einsum('ij,ij->i', p1_3vec, y1), np.einsum('ij,ij->i', p1_3vec, x1))
    phi3 = np.arctan2(np.einsum('ij,ij->i', p3_3vec, y3), np.einsum('ij,ij->i', p3_3vec, x3))

    return phi1, phi3, theta1, theta3


def calc_scattering_angle(parent_4_mom):
    """
    Cosine of the scattering angle w.r.t. the beam axis for a batch.

    Parameters
    ----------
    parent_4_mom : (N, 4) array

    Returns
    -------
    cos_psi : (N,) array
    """
    p3 = parent_4_mom[:, 1:]
    return p3[:, 2] / np.linalg.norm(p3, axis=1)


def calc_inv_mass(four_vec):
    """
    Invariant mass for a batch of 4-momenta.

    Parameters
    ----------
    four_vec : (N, 4) array

    Returns
    -------
    mass : (N,) array
    """
    m2 = four_vec[:, 0]**2 - np.sum(four_vec[:, 1:]**2, axis=1)
    return np.sqrt(np.maximum(m2, 0.0))

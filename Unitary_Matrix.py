import numpy as np

def euler_unitary_matrix(theta12, theta13, theta23, delta, alpha1, alpha2):
    """
    Construct a general U(3) matrix from 3 mixing angles and 3 phases.
    
    Args:
        theta12, theta13, theta23: Euler angles (in radians).
        delta: CP-violating phase (for SU(3) structure).
        alpha1, alpha2: Majorana-like phases (for U(3) completion).
    
    Returns:
        A 3x3 unitary U(3) matrix.
    """

    # Define sines and cosines
    c12 = np.cos(theta12)
    s12 = np.sin(theta12)
    c13 = np.cos(theta13)
    s13 = np.sin(theta13)
    c23 = np.cos(theta23)
    s23 = np.sin(theta23)

    # Build the SU(3) core (like PMNS/CKM)
    U = np.array([
        [c12 * c13, s12 * c13, s13 * np.exp(-1j * delta)],
        [-s12 * c23 - c12 * s23 * s13 * np.exp(1j * delta),
         c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta),
         s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * np.exp(1j * delta),
         -c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta),
         c23 * c13]
    ])

    # Apply extra diagonal phase matrix for U(3)
    D = np.diag([1, np.exp(1j * alpha1), np.exp(1j * alpha2)])

    return U @ D

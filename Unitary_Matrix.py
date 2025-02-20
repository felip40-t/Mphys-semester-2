import numpy as np

def euler_unitary_matrix(theta1, theta2, theta3, phi1, phi2, phi3, delta1, delta2):
    
    """
    Generate a U(3) matrix given 8 input parameters: 
    3 Euler angles (theta1, theta2, theta3) and 5 phases (phi1, phi2, phi3, delta1, delta2).
    
    Args:
        theta1, theta2, theta3: Rotation angles in radians (real parameters).
        phi1, phi2, phi3: Phases in radians for SU(3) rotations (real parameters).
        delta1, delta2: Phases in radians for diagonal matrix D(delta1, delta2) (real parameters).
    
    Returns:
        A 3x3 U(3) matrix.
    """
    # Rotation matrix V (SU(3) part)
    V = np.array([
        [np.cos(theta1) * np.exp(1j * phi1), np.sin(theta1) * np.exp(1j * phi2), 0],
        [-np.sin(theta1) * np.cos(theta2) * np.exp(-1j * phi2), 
         np.cos(theta1) * np.cos(theta2) * np.exp(-1j * phi1), np.sin(theta2)],
        [np.sin(theta1) * np.sin(theta2) * np.exp(-1j * phi2), 
         -np.cos(theta1) * np.sin(theta2) * np.exp(-1j * phi1), np.cos(theta2)]
    ])
    
    # Diagonal phase matrix D(delta1, delta2)
    D = np.diag([1, np.exp(1j * delta1), np.exp(1j * delta2)])
    
    # Rotation matrix W (SU(3) part, similar to V but for the second set of angles)
    W = np.array([
        [np.cos(theta3), np.sin(theta3), 0],
        [-np.sin(theta3), np.cos(theta3), 0],
        [0, 0, 1]
    ])
    
    # Combine V, D, and W to get the U(3) matrix
    U = V @ D @ W
    return U


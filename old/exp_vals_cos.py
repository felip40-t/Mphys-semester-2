from math import gamma
import os
import numpy as np
from histo_plotter import read_data

# Define the base paths for e- and e+ directories in WW folder
WW_path_e_minus = "/home/felipetcach/project/MG5_aMC_v3_5_6/WW_process/Plots and data/e-"
WW_path_e_plus = "/home/felipetcach/project/MG5_aMC_v3_5_6/WW_process/Plots and data/e+"

# Define the data directories for e- and e+
data_dir_e_minus = os.path.join(WW_path_e_minus, "angle_data_3.txt")
data_dir_e_plus = os.path.join(WW_path_e_plus, "angle_data_3.txt")

# Define the base paths for mu- and e+ directories in WZ folder
WZ_path_mu_minus = "/home/felipetcach/project/MG5_aMC_v3_5_6/WZ_process/Plots and data/mu-"
WZ_path_e_plus = "/home/felipetcach/project/MG5_aMC_v3_5_6/WZ_process/Plots and data/e+"

# Define the data directories for mu- and e+
data_dir_mu_minus = os.path.join(WZ_path_mu_minus, "angle_data_7.txt")
data_dir_e_plus_WZ = os.path.join(WZ_path_e_plus, "angle_data_7.txt")

# Read data for both e- and e+ in WW and mu- and e+ in WZ
try:
    cos_theta_values_e_minus = read_data(data_dir_e_minus)
    cos_theta_values_e_plus = read_data(data_dir_e_plus)
    cos_theta_values_mu_minus = read_data(data_dir_mu_minus)
    cos_theta_values_e_plus_WZ = read_data(data_dir_e_plus_WZ)
except FileNotFoundError as e:
    print(f"Error reading file: {e}")
    raise

# Functions to calculate parameters
def calc_exp_cos(data):
    """Calculates expectation value for cos(theta) using Monte Carlo integration."""
    return np.mean(data)

def calc_exp_cos_sqrd(data):
    """Calculates expectation value for cos^2(theta) using Monte Carlo integration."""
    return np.mean(np.square(data))

def calc_alpha_1(data):
    """Calculates alpha_1 coefficient."""
    return 0.5 * (np.sqrt(3/np.pi)) * calc_exp_cos(data)

def calc_alpha_2(data):
    """Calculates alpha_2 coefficient."""
    return 0.25 * np.sqrt(5/np.pi) * (3 * calc_exp_cos_sqrd(data) - 1)

def calc_gamma1010(data1, data2):
    """Calculates gamma_1010 coefficient."""
    return 0.25 * (3/np.pi) * np.mean(data1 * data2)

def calc_gamma2020(data1, data2):
    """Calculates gamma_2020 coefficient."""
    q1 = 3 * np.square(data1) - 1
    q2 = 3 * np.square(data2) - 1
    return (5 / (16 * np.pi)) * np.mean(q1 * q2)

# # Calculate alphas and gammas for e- and e+ from WW
# alpha_1_e_minus_WW = calc_alpha_1(cos_theta_values_e_minus)
# alpha_2_e_minus_WW = calc_alpha_2(cos_theta_values_e_minus)
# alpha_1_e_plus_WW = calc_alpha_1(cos_theta_values_e_plus)
# alpha_2_e_plus_WW = calc_alpha_2(cos_theta_values_e_plus)

# gamma1010_WW = calc_gamma1010(cos_theta_values_e_minus, cos_theta_values_e_plus)
# gamma2020_WW = calc_gamma2020(cos_theta_values_e_minus, cos_theta_values_e_plus)

# Calculate alphas and gammas for mu- and e+ from WZ
alpha_1_mu_minus_WZ = calc_alpha_1(cos_theta_values_mu_minus)
alpha_2_mu_minus_WZ = calc_alpha_2(cos_theta_values_mu_minus)
alpha_1_e_plus_WZ = calc_alpha_1(cos_theta_values_e_plus_WZ)
alpha_2_e_plus_WZ = calc_alpha_2(cos_theta_values_e_plus_WZ)

gamma1010_WZ = calc_gamma1010(cos_theta_values_mu_minus, cos_theta_values_e_plus_WZ)
gamma2020_WZ = calc_gamma2020(cos_theta_values_mu_minus, cos_theta_values_e_plus_WZ)

# # Save all coefficients for e+ in WW
# with open(os.path.join(WW_path_e_plus, "exp_vals_3.txt"), 'w') as f:
#     f.write(f"Alpha_1 (e+): {alpha_1_e_plus_WW}\n")
#     f.write(f"Alpha_2 (e+): {alpha_2_e_plus_WW}\n")
#     f.write(f"Gamma_1010 (WW): {gamma1010_WW}\n")
#     f.write(f"Gamma_2020 (WW): {gamma2020_WW}\n")

# # Save all coefficients for e- in WW
# with open(os.path.join(WW_path_e_minus, "exp_vals_3.txt"), 'w') as f:
#     f.write(f"Alpha_1 (e-): {alpha_1_e_minus_WW}\n")
#     f.write(f"Alpha_2 (e-): {alpha_2_e_minus_WW}\n")

# Save all coefficients for mu- in WZ
with open(os.path.join(WZ_path_mu_minus, "exp_vals_7.txt"), 'w') as f:
    f.write(f"Alpha_1 (mu-): {0.219*alpha_1_mu_minus_WZ}\n")
    f.write(f"Alpha_2 (mu-): {alpha_2_mu_minus_WZ}\n")
    f.write(f"Gamma_1010 (WZ): {0.219*gamma1010_WZ}\n")
    f.write(f"Gamma_2020 (WZ): {gamma2020_WZ}\n")

# Save all coefficients for e+ in WZ
with open(os.path.join(WZ_path_e_plus, "exp_vals_7.txt"), 'w') as f:
    f.write(f"Alpha_1 (e+): {alpha_1_e_plus_WZ}\n")
    f.write(f"Alpha_2 (e+): {alpha_2_e_plus_WZ}\n")


def calc_R_c(a21, a23, gamma2020):
    top = 1 - 4*np.sqrt(5*np.pi)*a21 - 4*np.sqrt(5*np.pi)*a23 + 80*np.pi*gamma2020
    bottom = 1 - 4*np.sqrt(5*np.pi)*a21 - 4*np.sqrt(5*np.pi)*a23 + 80*np.pi*a21*a23
    return top/bottom

# print("R_c for WW: ", calc_R_c(alpha_2_e_minus_WW, alpha_2_e_plus_WW, gamma2020_WW))
print("R_c for WZ: ", calc_R_c(alpha_2_mu_minus_WZ, alpha_2_e_plus_WZ, gamma2020_WZ))

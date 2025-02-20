import os
import numpy as np
from histo_plotter import read_data


# Define the base paths for e- and e+ directories
base_path_e_minus = "/home/felipetcach/project/MG5_aMC_v3_5_6/WW_process/Plots and data/e-"
base_path_e_plus = "/home/felipetcach/project/MG5_aMC_v3_5_6/WW_process/Plots and data/e+"

# Define the data directories for e- and e+
data_dir_e_minus = os.path.join(base_path_e_minus, "cos_exp_vals_3.txt")
data_dir_e_plus = os.path.join(base_path_e_plus, "cos_exp_vals_3.txt")


# Define the base paths for mu- and e+ directories in WZ folder
WZ_path_mu_minus = "/home/felipetcach/project/MG5_aMC_v3_5_6/WZ_process/Plots and data/mu-"
WZ_path_e_plus = "/home/felipetcach/project/MG5_aMC_v3_5_6/WZ_process/Plots and data/e+"

# Define the data directories for mu- and e+
data_dir_mu_minus = os.path.join(WZ_path_mu_minus, "angle_data_4.txt")
data_dir_e_plus_WZ = os.path.join(WZ_path_e_plus, "angle_data_4.txt")


cos_theta_values_mu_minus = read_data(data_dir_mu_minus)
cos_theta_values_e_plus_WZ = read_data(data_dir_e_plus_WZ)


def calc_exp_cos(data):
    """Calculates expectation value for cos(theta) using Monte Carlo integration."""
    return np.mean(data)

def calc_exp_cos_sqrd(data):
    """Calculates expectation value for cos^2(theta) using Monte Carlo integration."""
    return np.mean(np.square(data))


def calc_exp_cos_4(data):
    """Calculates expectation value for cos^4(theta) using Monte Carlo integration."""
    return np.mean(np.power(data, 4))


def calc_f0(exp_sqr):
    return (2 - 5*exp_sqr)

def calc_fR(exp, exp_sqr):
    return (-0.5 + exp + 2.5*exp_sqr)

def calc_fL(exp, exp_sqr):
    return (-0.5 - exp + 2.5*exp_sqr)

def calc_A0(exp_sqr):
    return 4 - 10*exp_sqr

def calc_A4(exp):
    return 4*exp




# Calculate the expectation values for mu- and e+ (WZ process)
exp_val_mu_minus = calc_exp_cos(cos_theta_values_mu_minus)
exp_sqr_val_mu_minus = calc_exp_cos_sqrd(cos_theta_values_mu_minus)

exp_val_e_plus_WZ = calc_exp_cos(cos_theta_values_e_plus_WZ)
exp_sqr_val_e_plus_WZ = calc_exp_cos_sqrd(cos_theta_values_e_plus_WZ)

# Calculate f0, fR, fL, A0, A4 for mu- (WZ process)
f0_mu_minus = calc_f0(exp_sqr_val_mu_minus)
fR_mu_minus = calc_fR(exp_val_mu_minus, exp_sqr_val_mu_minus)
fL_mu_minus = calc_fL(exp_val_mu_minus, exp_sqr_val_mu_minus)
A_0_mu_minus = calc_A0(exp_sqr_val_mu_minus)
A_4_mu_minus = calc_A4(exp_val_mu_minus)

# Calculate uncertainty on A_0 for mu-
A_0_mu_minus_uncertainty = 10*(np.sqrt(calc_exp_cos_4(cos_theta_values_mu_minus) - (calc_exp_cos_sqrd(cos_theta_values_mu_minus)**2))/np.sqrt(500000))

# Calculate uncertainty on A_4 for mu-
A_4_mu_minus_uncertainty = 4*0.219*(np.sqrt(calc_exp_cos_sqrd(cos_theta_values_mu_minus) - calc_exp_cos(cos_theta_values_mu_minus)**2))/np.sqrt(500000)


# Calculate f0, fR, fL, A0, A4 for e+ (WZ process)
f0_e_plus_WZ = calc_f0(exp_sqr_val_e_plus_WZ)
fR_e_plus_WZ = calc_fR(exp_val_e_plus_WZ, exp_sqr_val_e_plus_WZ)
fL_e_plus_WZ = calc_fL(exp_val_e_plus_WZ, exp_sqr_val_e_plus_WZ)
A_0_e_plus_WZ = calc_A0(exp_sqr_val_e_plus_WZ)
A_4_e_plus_WZ = calc_A4(exp_val_e_plus_WZ)

# Calculate uncertainty on A_0 for e+
A_0_mu_minus_uncertainty = 10*(np.sqrt(calc_exp_cos_4(cos_theta_values_mu_minus) - (calc_exp_cos_sqrd(cos_theta_values_mu_minus)**2))/np.sqrt(500000))


# Output the results for mu- and e+ in WZ process
print(f"mu- Results (WZ process):")
print(f"f0: {f0_mu_minus}")
print(f"fR: {fR_mu_minus}")
print(f"fL: {fL_mu_minus}")
print(f"A_0: {A_0_mu_minus} +- {A_0_mu_minus_uncertainty}")
print(f"A_4: {0.219*A_4_mu_minus}\n")

print(f"e+ Results (WZ process):")
print(f"f0: {f0_e_plus_WZ}")
print(f"fR: {fR_e_plus_WZ}")
print(f"fL: {fL_e_plus_WZ}")
print(f"A_0: {A_0_e_plus_WZ}")
print(f"A_4: {A_4_e_plus_WZ}")

# # Read the expectation values for e- and e+
# exp_val_e_minus, exp_sqr_val_e_minus = read_data(data_dir_e_minus)  # for electron
# exp_val_e_plus, exp_sqr_val_e_plus = read_data(data_dir_e_plus)      # for positron

# # Calculate f0, fR, fL for e-
# f0_e_minus = calc_f0(exp_sqr_val_e_minus)
# fR_e_minus = calc_fL(exp_val_e_minus, exp_sqr_val_e_minus)
# fL_e_minus = calc_fR(exp_val_e_minus, exp_sqr_val_e_minus)

# # Calculate A_0 and A_4 for e-
# A_0_e_minus = calc_A0(exp_sqr_val_e_minus)
# A_4_e_minus = calc_A4(exp_val_e_minus)

# # Calculate f0, fR, fL for e+
# f0_e_plus = calc_f0(exp_sqr_val_e_plus)
# fR_e_plus = calc_fR(exp_val_e_plus, exp_sqr_val_e_plus)
# fL_e_plus = calc_fL(exp_val_e_plus, exp_sqr_val_e_plus)

# # Calculate A_0 and A_4 for e+
# A_0_e_plus = calc_A0(exp_sqr_val_e_plus)
# A_4_e_plus = calc_A4(exp_val_e_plus)

# # Print results for e-
# print(f"e- Results:")
# print(f"f0: {f0_e_minus}")
# print(f"fR: {fR_e_minus}")
# print(f"fL: {fL_e_minus}")
# print(f"A_0: {A_0_e_minus}")
# print(f"A_4: {A_4_e_minus}\n")

# # Print results for e+
# print(f"e+ Results:")
# print(f"f0: {f0_e_plus}")
# print(f"fR: {fR_e_plus}")
# print(f"fL: {fL_e_plus}")
# print(f"A_0: {A_0_e_plus}")
# print(f"A_4: {A_4_e_plus}")

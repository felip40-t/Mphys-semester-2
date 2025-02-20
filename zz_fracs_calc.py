import os
import numpy as np
from histo_plotter import read_data

# Base path for ZZ data
ZZ_PATH = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_ZZ_SM/Plots and data"

# Function to calculate all parameters for ZZ datasets
def calculate_parameters(data1, data3, phi_data1, phi_data3):
    """Calculates all parameters for the ZZ case using two datasets (mu+ and e+)."""
    num_events = len(data1)
    eta = 0.214

    def spherical_harmonic_1(data):
        return 0.5 * np.sqrt(3 / np.pi) * data

    def spherical_harmonic_2(data):
        return 0.25 * np.sqrt(5 / np.pi) * (-1 + 3 * (data ** 2))

    def spherical_harmonic_2_m2(cos_theta, phi):
        return 0.25 * np.cos(2 * phi) * np.sqrt(15 / (2 * np.pi)) * np.sin(np.arccos(cos_theta)) ** 2

    def gamma222m2(cos_theta1, phi1, cos_theta3, phi3):
        return (15 / (32 * np.pi)) * np.cos(2 * (phi3 - phi1)) * np.sin(np.arccos(cos_theta1)) ** 2 * np.sin(np.arccos(cos_theta3)) ** 2

    def gamma212m1(cos_theta1, phi1, cos_theta3, phi3):
        return -(15 / (8 * np.pi)) * np.cos(phi3 - phi1) * cos_theta1 * cos_theta3 * np.sin(np.arccos(cos_theta1)) * np.sin(np.arccos(cos_theta3))

    def gamma111m1(cos_theta1, phi1, cos_theta3, phi3):
        return -(3 / (8 * np.pi)) * np.cos(phi3 - phi1) * np.sin(np.arccos(cos_theta1)) * np.sin(np.arccos(cos_theta3))

    Y11 = np.mean(spherical_harmonic_1(data1))
    Y13 = np.mean(spherical_harmonic_1(data3))
    Y21 = np.mean(spherical_harmonic_2(data1))
    Y23 = np.mean(spherical_harmonic_2(data3))
    Y11Y13 = np.mean(spherical_harmonic_1(data1) * spherical_harmonic_1(data3))
    Y21Y23 = np.mean(spherical_harmonic_2(data1) * spherical_harmonic_2(data3))
    Y11Y23 = np.mean(spherical_harmonic_1(data1) * spherical_harmonic_2(data3))
    Y21Y13 = np.mean(spherical_harmonic_2(data1) * spherical_harmonic_1(data3))

    Y2m2_1 = np.mean(spherical_harmonic_2_m2(data1, phi_data1))
    Y2m2_3 = np.mean(spherical_harmonic_2_m2(data3, phi_data3))

    gamma222m2_val = np.mean(gamma222m2(data1, phi_data1, data3, phi_data3))
    gamma212m1_val = np.mean(gamma212m1(data1, phi_data1, data3, phi_data3))
    gamma111m1_val = np.mean(gamma111m1(data1, phi_data1, data3, phi_data3))

    sigma_g222m2 = np.sqrt((np.mean(gamma222m2(data1, phi_data1, data3, phi_data3) ** 2) - gamma222m2_val ** 2) / num_events)
    sigma_g212m1 = np.sqrt((np.mean(gamma212m1(data1, phi_data1, data3, phi_data3) ** 2) - gamma212m1_val ** 2) / num_events)
    sigma_g111m1 = np.sqrt((np.mean(gamma111m1(data1, phi_data1, data3, phi_data3) ** 2) - gamma111m1_val ** 2) / num_events)

    sigma_Y11 = np.sqrt((np.mean(spherical_harmonic_1(data1) ** 2) - Y11 ** 2) / num_events)
    sigma_Y13 = np.sqrt((np.mean(spherical_harmonic_1(data3) ** 2) - Y13 ** 2) / num_events)
    sigma_Y21 = np.sqrt((np.mean(spherical_harmonic_2(data1) ** 2) - Y21 ** 2) / num_events)
    sigma_Y23 = np.sqrt((np.mean(spherical_harmonic_2(data3) ** 2) - Y23 ** 2) / num_events)

    sigma_Y11Y13 = np.sqrt((np.mean((spherical_harmonic_1(data1) * spherical_harmonic_1(data3)) ** 2) - Y11Y13 ** 2) / num_events)
    sigma_Y21Y23 = np.sqrt((np.mean((spherical_harmonic_2(data1) * spherical_harmonic_2(data3)) ** 2) - Y21Y23 ** 2) / num_events)

    sigma_Y2m2_1 = np.sqrt((np.mean(spherical_harmonic_2_m2(data1, phi_data1) ** 2) - Y2m2_1 ** 2) / num_events)
    sigma_Y2m2_3 = np.sqrt((np.mean(spherical_harmonic_2_m2(data3, phi_data3) ** 2) - Y2m2_3 ** 2) / num_events)

    def f00():
        return (1 / 9) * (1 - 4 * np.sqrt(5 * np.pi) * Y21 + 80 * np.pi * Y21Y23 - 4 * np.sqrt(5 * np.pi) * Y23)

    def f0R():
        return -((-2 * np.sqrt(3 * np.pi) * Y13 + 8 * np.sqrt(15) * np.pi * Y21Y13 - eta + 
                4 * np.sqrt(5 * np.pi) * Y21 * eta + 40 * np.pi * Y21Y23 * eta - 
                2 * np.sqrt(5 * np.pi) * Y23 * eta) / (9 * eta))

    def f0L():
        return -((2 * np.sqrt(3 * np.pi) * Y13 - 8 * np.sqrt(15) * np.pi * Y21Y13 - eta + 
                4 * np.sqrt(5 * np.pi) * Y21 * eta + 40 * np.pi * Y21Y23 * eta - 
                2 * np.sqrt(5 * np.pi) * Y23 * eta) / (9 * eta))

    def fR0():
        return -((-2 * np.sqrt(3 * np.pi) * Y11 + 8 * np.sqrt(15) * np.pi * Y11Y23 - eta - 
                2 * np.sqrt(5 * np.pi) * Y21 * eta + 40 * np.pi * Y21Y23 * eta + 
                4 * np.sqrt(5 * np.pi) * Y23 * eta) / (9 * eta))

    def fL0():
        return -((2 * np.sqrt(3 * np.pi) * Y11 - 8 * np.sqrt(15) * np.pi * Y11Y23 - eta - 
                2 * np.sqrt(5 * np.pi) * Y21 * eta + 40 * np.pi * Y21Y23 * eta + 
                4 * np.sqrt(5 * np.pi) * Y23 * eta) / (9 * eta))

    def fLL():
        return -(1 / (9 * eta ** 2)) * (-12 * np.pi * Y11Y13 + 2 * np.sqrt(3 * np.pi) * Y11 * eta + 
                                    4 * np.sqrt(15) * np.pi * Y11Y23 * eta + 2 * np.sqrt(3 * np.pi) * Y13 * eta + 
                                    4 * np.sqrt(15) * np.pi * Y21Y13 * eta - eta ** 2 - 
                                    2 * np.sqrt(5 * np.pi) * Y21 * eta ** 2 - 20 * np.pi * Y21Y23 * eta ** 2 - 
                                    2 * np.sqrt(5 * np.pi) * Y23 * eta ** 2)

    def fLR():
        return -(1 / (9 * eta ** 2)) * (12 * np.pi * Y11Y13 + 2 * np.sqrt(3 * np.pi) * Y11 * eta + 
                                    4 * np.sqrt(15) * np.pi * Y11Y23 * eta - 2 * np.sqrt(3 * np.pi) * Y13 * eta - 
                                    4 * np.sqrt(15) * np.pi * Y21Y13 * eta - eta ** 2 - 
                                    2 * np.sqrt(5 * np.pi) * Y21 * eta ** 2 - 20 * np.pi * Y21Y23 * eta ** 2 - 
                                    2 * np.sqrt(5 * np.pi) * Y23 * eta ** 2)

    def fRL():
        return -(1 / (9 * eta ** 2)) * (12 * np.pi * Y11Y13 - 2 * np.sqrt(3 * np.pi) * Y11 * eta - 
                                    4 * np.sqrt(15) * np.pi * Y11Y23 * eta + 2 * np.sqrt(3 * np.pi) * Y13 * eta + 
                                    4 * np.sqrt(15) * np.pi * Y21Y13 * eta - eta ** 2 - 
                                    2 * np.sqrt(5 * np.pi) * Y21 * eta ** 2 - 20 * np.pi * Y21Y23 * eta ** 2 - 
                                    2 * np.sqrt(5 * np.pi) * Y23 * eta ** 2)

    def fRR():
        return -(1 / (9 * eta ** 2)) * (-12 * np.pi * Y11Y13 - 2 * np.sqrt(3 * np.pi) * Y11 * eta - 
                                    4 * np.sqrt(15) * np.pi * Y11Y23 * eta - 2 * np.sqrt(3 * np.pi) * Y13 * eta - 
                                    4 * np.sqrt(15) * np.pi * Y21Y13 * eta - eta ** 2 - 
                                    2 * np.sqrt(5 * np.pi) * Y21 * eta ** 2 - 20 * np.pi * Y21Y23 * eta ** 2 - 
                                    2 * np.sqrt(5 * np.pi) * Y23 * eta ** 2)

    def alpha11():
        return - (1 / 4) * (fL0() + fLL() + fLR() - fR0() - fRL() - fRR()) * np.sqrt(3 / np.pi) * eta

    def alpha21():
        return (-2 * f00() - 2 * f0L() - 2 * f0R() + fL0() + fLL() + fLR() + fR0() + fRL() + fRR()) / (4 * np.sqrt(5 * np.pi))

    def alpha13():
        return - (1 / 4) * (f0L() - f0R() + fLL() - fLR() + fRL() - fRR()) * np.sqrt(3 / np.pi) * eta

    def alpha23():
        return (-2 * f00() + f0L() + f0R() - 2 * fL0() + fLL() + fLR() - 2 * fR0() + fRL() + fRR()) / (4 * np.sqrt(5 * np.pi))

    def gamma1010():
        return (3 / (16 * np.pi)) * (fLL() - fLR() - fRL() + fRR()) * eta * eta

    def gamma2020():
        return (4 * f00() - 2 * f0L() - 2 * f0R() - 2 * fL0() + fLL() + fLR() - 2 * fR0() + fRL() + fRR()) / (80 * np.pi)

    parameters = {
        'f00': f00(),
        'f0R': f0R(),
        'f0L': f0L(),
        'fR0': fR0(),
        'fL0': fL0(),
        'fLL': fLL(),
        'fLR': fLR(),
        'fRL': fRL(),
        'fRR': fRR(),
        'alpha11': alpha11(),
        'alpha13': alpha13(),
        'alpha21': alpha21(),
        'alpha23': alpha23(),
        'gamma1010': gamma1010(),
        'gamma2020': gamma2020(),
        'alpha2m2_1': Y2m2_1,
        'alpha2m2_3': Y2m2_3,
        'gamma222m2': gamma222m2_val,
        'gamma212m1': gamma212m1_val,
        'gamma111m1': gamma111m1_val
    }

    uncertainties = {
        'alpha11': sigma_Y11,
        'alpha13': sigma_Y13,
        'alpha21': sigma_Y21,
        'alpha23': sigma_Y23,
        'gamma1010': sigma_Y11Y13,
        'gamma2020': sigma_Y21Y23,
        'alpha2m2_1': sigma_Y2m2_1,
        'alpha2m2_3': sigma_Y2m2_3,
        'gamma222m2': sigma_g222m2,
        'gamma212m1': sigma_g212m1,
        'gamma111m1': sigma_g111m1
    }

    return parameters, uncertainties

# Helper function to write parameters and uncertainties to file
def write_results_to_file(output_file, params, uncs):
    with open(output_file, 'w') as f:
        for param, value in params.items():
            if param in uncs:
                f.write(f"{param}: {value:.6f} ± {uncs[param]:.6f}\n")
            else:
                f.write(f"{param}: {value:.6f}\n")

# Read cos(theta) values for mu+ and e+ datasets
theta_path_1 = os.path.join(ZZ_PATH, "mu+/theta_data_4.txt")
theta_path_2 = os.path.join(ZZ_PATH, "e+/theta_data_4.txt")

print(f"Reading file... {theta_path_1}")
cos_theta_values_1 = read_data(theta_path_1)
print(f"Reading file... {theta_path_2}")
cos_theta_values_2 = read_data(theta_path_2)

# Read phi values
phi_path_1 = os.path.join(ZZ_PATH, "mu+/phi_data_4.txt")
phi_path_2 = os.path.join(ZZ_PATH, "e+/phi_data_4.txt")

print(f"Reading file... {phi_path_1}")
phi_values_1 = read_data(phi_path_1)
print(f"Reading file... {phi_path_2}")
phi_values_2 = read_data(phi_path_2)

# Calculate parameters and uncertainties
parameters, uncertainties = calculate_parameters(cos_theta_values_1, cos_theta_values_2, phi_values_1, phi_values_2)

# Write results to file
output_file = os.path.join(ZZ_PATH, "fracs_4.txt")
write_results_to_file(output_file, parameters, uncertainties)
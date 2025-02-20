import os
import numpy as np
from histo_plotter import read_data

# Base path for WZ data
WW_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WW_SM/Plots and data"


# Function to calculate all parameters for WZ datasets
def calculate_parameters(data1, data3, phi_data1, phi_data3):
    """Calculates all parameters for the WW case using two datasets (mu- and e+)."""

    num_events = len(data1)

    def spherical_harmonic_1(data):
        return 0.5 * np.sqrt(3 / np.pi) * data

    def spherical_harmonic_2(data):
        return 0.25 * np.sqrt(5 / np.pi) * ( - 1 + 3 * (data ** 2))

    def spherical_harmonic_2_m2(cos_theta, phi):
        return 0.25 * np.cos(2*phi) * np.sqrt( 15 / (2*np.pi) ) * np.sin(np.arccos(cos_theta))**2

    
    def gamma222m2(cos_theta1, phi1, cos_theta3, phi3):
        return (15/(32*np.pi)) * np.cos(2 * (phi3 - phi1)) * np.sin(np.arccos(cos_theta1))**2 * np.sin(np.arccos(cos_theta3))**2

    def gamma212m1(cos_theta1, phi1, cos_theta3, phi3):
        return - (15/(8*np.pi)) * np.cos(phi3 - phi1) * cos_theta1 * cos_theta3 * np.sin(np.arccos(cos_theta1)) * np.sin(np.arccos(cos_theta3))

    def gamma111m1(cos_theta1, phi1, cos_theta3, phi3):
        return - (3/(8*np.pi)) * np.cos(phi3 - phi1) * np.sin(np.arccos(cos_theta1)) * np.sin(np.arccos(cos_theta3))


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

    sigma_Y2m2_1 = np.sqrt( (np.mean( spherical_harmonic_2_m2(data1, phi_data1) ** 2  ) - Y2m2_1 ** 2) / num_events)
    sigma_Y2m2_3 = np.sqrt( (np.mean( spherical_harmonic_2_m2(data3, phi_data3) ** 2  ) - Y2m2_3 ** 2) / num_events)

    gamma222m2_val = np.mean(gamma222m2(data1, phi_data1, data3, phi_data3))
    gamma212m1_val = np.mean(gamma212m1(data1, phi_data1, data3, phi_data3))
    gamma111m1_val = np.mean(gamma111m1(data1, phi_data1, data3, phi_data3))

    sigma_g222m2 = np.sqrt( (np.mean(gamma222m2(data1, phi_data1, data3, phi_data3) ** 2) - gamma222m2_val ** 2) / num_events)
    sigma_g212m1 = np.sqrt( (np.mean(gamma212m1(data1, phi_data1, data3, phi_data3) ** 2) - gamma212m1_val ** 2) / num_events)
    sigma_g111m1 = np.sqrt( (np.mean(gamma111m1(data1, phi_data1, data3, phi_data3) ** 2) - gamma111m1_val ** 2) / num_events)


    sigma_Y11 = np.sqrt( (np.mean(spherical_harmonic_1(data1) ** 2) - Y11 ** 2) / num_events)
    sigma_Y13 = np.sqrt( (np.mean(spherical_harmonic_1(data3) ** 2) - Y13 ** 2) / num_events)
    sigma_Y21 = np.sqrt( (np.mean(spherical_harmonic_2(data1) ** 2) - Y21 ** 2) / num_events)
    sigma_Y23 = np.sqrt( (np.mean(spherical_harmonic_2(data3) ** 2) - Y23 ** 2) / num_events)

    sigma_Y11Y13 = np.sqrt( (np.mean((spherical_harmonic_1(data1) * spherical_harmonic_1(data3)) ** 2) - Y11Y13 ** 2) / num_events)
    sigma_Y21Y23 = np.sqrt( (np.mean((spherical_harmonic_2(data1) * spherical_harmonic_2(data3)) ** 2) - Y21Y23 ** 2) / num_events)

    def alpha11():
        return Y11
    
    def alpha13():
        return Y13

    def alpha21():
        return Y21
    
    def alpha23():
        return Y23
    
    def gamma1010():
        return Y11Y13
    
    def gamma2020():
        return Y21Y23

    # exp_cos_1 = np.mean(data1)
    # exp_sqr_cos_1 = np.mean(data1**2)

    # unc_cos_1 = np.sqrt((exp_sqr_cos_1 - exp_cos_1**2)/num_events) 
    # unc_cos_sqr_1 = np.sqrt((np.mean(data1**4) - exp_sqr_cos_1**2)/num_events)
    
    # exp_cos_3 = np.mean(data3)
    # exp_sqr_cos_3 = np.mean(data3**2)

    # unc_cos_3 = np.sqrt((exp_sqr_cos_3 - exp_cos_3**2)/num_events) 
    # unc_cos_sqr_3 = np.sqrt((np.mean(data3**4) - exp_sqr_cos_3**2)/num_events)

    # f0_1 = 2 - 5 * exp_sqr_cos_1
    # fR_1 = -0.5 + exp_cos_1 + 2.5 * exp_sqr_cos_1
    # fL_1 = -0.5 - exp_cos_1 + 2.5 * exp_sqr_cos_1

    # sigma_f0_1 = 5 * unc_cos_sqr_1
    # sigma_fR_1 = np.sqrt( (unc_cos_1**2) + 2.5**2 * unc_cos_sqr_1**2 )
    
    # print(f"f0_1: {f0_1} +- {sigma_f0_1}")
    # print(f"fL_1: {fL_1} +- {sigma_fR_1}")
    # print(f"fR_1: {fR_1} +- {sigma_fR_1}")


    # f0_3 = 2 - 5 * exp_sqr_cos_3
    # fR_3 = -0.5 - exp_cos_3 + 2.5 * exp_sqr_cos_3
    # fL_3 = -0.5 + exp_cos_3 + 2.5 * exp_sqr_cos_3

    # sigma_f0_3 = 5 * unc_cos_sqr_3
    # sigma_fR_3 = np.sqrt( (unc_cos_3**2) + 2.5**2 * unc_cos_sqr_3**2 )
    
    # print(f"f0_3: {f0_3} +- {sigma_f0_3}")
    # print(f"fL_3: {fL_3} +- {sigma_fR_3}")
    # print(f"fR_3: {fR_3} +- {sigma_fR_3}")



    def R_c():
        top = 1 - 4*np.sqrt(5*np.pi)*alpha21() - 4*np.sqrt(5*np.pi)*alpha23() + 80*np.pi*gamma2020()
        bottom = 1 - 4*np.sqrt(5*np.pi)*alpha21() - 4*np.sqrt(5*np.pi)*alpha23() + 80*np.pi*alpha21()*alpha23()
        return top/bottom

    #print(f"R_c quantity: {R_c()}")

    parameters = {
        'alpha11': alpha11(),
        'alpha21': alpha21(),
        'alpha13': alpha13(),
        'alpha23': alpha23(),
        'gamma1010': gamma1010(),
        'gamma2020': gamma2020(),
        'alpha2m2_1' : Y2m2_1,
        'alpha2m2_3' : Y2m2_3,
        'gamma222m2' : gamma222m2_val,
        'gamma212m1' : gamma212m1_val,
        'gamma111m1' : gamma111m1_val
    }

    uncertainties = {
        'alpha11':sigma_Y11,
        'alpha21': sigma_Y21,
        'alpha13': sigma_Y13,
        'alpha23': sigma_Y23,
        'gamma1010': sigma_Y11Y13,
        'gamma2020': sigma_Y21Y23,
        'alpha2m2_1' : sigma_Y2m2_1,
        'alpha2m2_3' : sigma_Y2m2_3,
        'gamma222m2' : sigma_g222m2,
        'gamma212m1' : sigma_g212m1,
        'gamma111m1' : sigma_g111m1
    }

    return parameters, uncertainties

# Helper function to write parameters and uncertainties to file
def write_results_to_file(output_file, params, uncs):
    with open(output_file, 'w') as f:
        for param, value in params.items():
            if param in uncs:
                # Write parameter value with its uncertainty
                f.write(f"{param}: {value:.6f} ± {uncs[param]:.6f}\n")
            else:
                # Write parameter value without uncertainty
                f.write(f"{param}: {value:.6f}\n")


# Read cos(theta) values for mu+, e+, and mu- datasets
theta_path_1 = os.path.join(WW_path, "e+/theta_data_1.txt")
theta_path_3 = os.path.join(WW_path, "mu-/theta_data_1.txt")
data1 = read_data(theta_path_1)
data3 = read_data(theta_path_3)

# Read phi values for mu+, e+, and mu- datasets
phi_path_1 = os.path.join(WW_path, "e+/phi_data_1.txt")
phi_path_3 = os.path.join(WW_path, "mu-/phi_data_1.txt")
phi_data1 = read_data(phi_path_1)
phi_data3 = read_data(phi_path_3)

# Calculate parameters and uncertainties
parameters, uncertainties = calculate_parameters(data1, data3, phi_data1, phi_data3)

# Write results to fracs_1.txt
output_file = os.path.join(WW_path, "fracs_1.txt")
write_results_to_file(output_file, parameters, uncertainties)
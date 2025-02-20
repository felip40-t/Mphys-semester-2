import os
import numpy as np
from histo_plotter import read_data

# Base path for WZ data
WZ_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/pp_WZ_SM/Plots and data"

# Function to calculate all parameters for WZ datasets
def calculate_parameters(data1, data3, data_mu_m, phi_data1, phi_data3, phi_data_mu_m):
    """Calculates all parameters for the WZ case using two datasets (mu+ and e+)."""

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

    
    # Calculate polarisation fractions
    eta = 0.214

    def f00():
        return (1/9) * (1 - 4 * np.sqrt(5 * np.pi) * Y21 + 80 * np.pi * Y21Y23 - 4 * np.sqrt(5 * np.pi) * Y23)

    def f0R():
        return -((-2 * np.sqrt(3 * np.pi) * Y13 + 8 * np.sqrt(15) * np.pi * Y21Y13 - eta + 
                4 * np.sqrt(5 * np.pi) * Y21 * eta + 40 * np.pi * Y21Y23 * eta - 
                2 * np.sqrt(5 * np.pi) * Y23 * eta) / (9 * eta))

    def f0L():
        return -((2 * np.sqrt(3 * np.pi) * Y13 - 8 * np.sqrt(15) * np.pi * Y21Y13 - eta + 
                4 * np.sqrt(5 * np.pi) * Y21 * eta + 40 * np.pi * Y21Y23 * eta - 
                2 * np.sqrt(5 * np.pi) * Y23 * eta) / (9 * eta))

    def fR0():
        return (1/9) * (1 + 2 * np.sqrt(3 * np.pi) * Y11 - 8 * np.sqrt(15) * np.pi * Y11Y23 + 
                        2 * np.sqrt(5 * np.pi) * Y21 - 40 * np.pi * Y21Y23 - 
                        4 * np.sqrt(5 * np.pi) * Y23)

    def fL0():
        return (1/9) * (1 - 2 * np.sqrt(3 * np.pi) * Y11 + 8 * np.sqrt(15) * np.pi * Y11Y23 + 
                        2 * np.sqrt(5 * np.pi) * Y21 - 40 * np.pi * Y21Y23 - 
                        4 * np.sqrt(5 * np.pi) * Y23)

    def fLL():
        return -(1 / (9 * eta)) * (-12 * np.pi * Y11Y13 + 2 * np.sqrt(3 * np.pi) * Y13 + 
                                4 * np.sqrt(15) * np.pi * Y21Y13 - eta + 
                                2 * np.sqrt(3 * np.pi) * Y11 * eta + 
                                4 * np.sqrt(15) * np.pi * Y11Y23 * eta - 
                                2 * np.sqrt(5 * np.pi) * Y21 * eta - 
                                20 * np.pi * Y21Y23 * eta - 
                                2 * np.sqrt(5 * np.pi) * Y23 * eta)

    def fLR():
        return -(1 / (9 * eta)) * (12 * np.pi * Y11Y13 - 2 * np.sqrt(3 * np.pi) * Y13 - 
                                4 * np.sqrt(15) * np.pi * Y21Y13 - eta + 
                                2 * np.sqrt(3 * np.pi) * Y11 * eta + 
                                4 * np.sqrt(15) * np.pi * Y11Y23 * eta - 
                                2 * np.sqrt(5 * np.pi) * Y21 * eta - 
                                20 * np.pi * Y21Y23 * eta - 
                                2 * np.sqrt(5 * np.pi) * Y23 * eta)

    def fRL():
        return -(1 / (9 * eta)) * (12 * np.pi * Y11Y13 + 2 * np.sqrt(3 * np.pi) * Y13 + 
                                4 * np.sqrt(15) * np.pi * Y21Y13 - eta - 
                                2 * np.sqrt(3 * np.pi) * Y11 * eta - 
                                4 * np.sqrt(15) * np.pi * Y11Y23 * eta - 
                                2 * np.sqrt(5 * np.pi) * Y21 * eta - 
                                20 * np.pi * Y21Y23 * eta - 
                                2 * np.sqrt(5 * np.pi) * Y23 * eta)

    def fRR():
        return -(1 / (9 * eta)) * (-12 * np.pi * Y11Y13 - 2 * np.sqrt(3 * np.pi) * Y13 - 
                                4 * np.sqrt(15) * np.pi * Y21Y13 - eta - 
                                2 * np.sqrt(3 * np.pi) * Y11 * eta - 
                                4 * np.sqrt(15) * np.pi * Y11Y23 * eta - 
                                2 * np.sqrt(5 * np.pi) * Y21 * eta - 
                                20 * np.pi * Y21Y23 * eta - 
                                2 * np.sqrt(5 * np.pi) * Y23 * eta)
    
    def alpha11():
        return (1 / 4) * (-fL0() - fLL() - fLR() + fR0() + fRL() + fRR()) * np.sqrt(3 / np.pi)

    def alpha21():
        return (-2 * f00() - 2 * f0L() - 2 * f0R() + fL0() + fLL() + fLR() + fR0() + fRL() + fRR()) / (4 * np.sqrt(5 * np.pi))

    def alpha13():
        return -(1 / 4) * (f0L() - f0R() + fLL() - fLR() + fRL() - fRR()) * np.sqrt(3 / np.pi) * eta

    def alpha23():
        return (-2 * f00() + f0L() + f0R() - 2 * fL0() + fLL() + fLR() - 2 * fR0() + fRL() + fRR()) / (4 * np.sqrt(5 * np.pi))

    def gamma1010():
        return (3 * (fLL() - fLR() - fRL() + fRR()) * eta) / (16 * np.pi)

    def gamma2020():
        return (4 * f00() - 2 * f0L() - 2 * f0R() - 2 * fL0() + fLL() + fLR() - 2 * fR0() + fRL() + fRR()) / (80 * np.pi)

    def R_c():
        top = 1 - 4*np.sqrt(5*np.pi)*alpha21() - 4*np.sqrt(5*np.pi)*alpha23() + 80*np.pi*gamma2020()
        bottom = 1 - 4*np.sqrt(5*np.pi)*alpha21() - 4*np.sqrt(5*np.pi)*alpha23() + 80*np.pi*alpha21()*alpha23()
        return top/bottom

    print(f"R_c quantity: {R_c()}")

    def A_0(cos_theta):
        return 4 - 10 * np.mean(cos_theta**2)
    
    def A_1(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 5 * np.mean(np.cos(phi) * np.sin(2 * theta))
    
    def A_2(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 10 * np.mean(np.cos(2 * phi) * np.sin(theta)**2)
    
    def A_3(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 4 * np.mean(np.cos(phi) * np.sin(theta))
    
    def A_4(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 4 * np.mean(cos_theta)
    
    def A_5(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 4 * np.mean(np.sin(phi) * np.sin(theta))
    
    def A_6(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 5 * np.mean(np.sin(phi) * np.sin(2 * theta))
    
    def A_7(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 5 * np.mean(np.sin(2 * phi) * np.sin(theta)**2)
    
    def A_0_unc(cos_theta):
        return 10 * np.sqrt( np.mean( cos_theta**4 ) - A_0(cos_theta)**2 )/np.sqrt(num_events)
    
    def A_1_unc(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 5 * np.sqrt( np.mean(np.cos(phi)**2 * np.sin(2 * theta)**2) - A_1(cos_theta, phi)**2)/np.sqrt(num_events)

    def A_2_unc(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 10 * np.sqrt(np.mean(np.cos(2 * phi)**2 * np.sin(theta)**4) - A_2(cos_theta, phi)**2) / np.sqrt(num_events)

    def A_3_unc(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 4 * np.sqrt(np.mean(np.cos(phi)**2 * np.sin(theta)**2) - A_3(cos_theta, phi)**2) / np.sqrt(num_events)

    def A_4_unc(cos_theta, phi):
        return 4 * np.sqrt(np.mean(cos_theta**2) - A_4(cos_theta, phi)**2) / np.sqrt(num_events)

    def A_5_unc(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 4 * np.sqrt(np.mean(np.sin(phi)**2 * np.sin(theta)**2) - A_5(cos_theta, phi)**2) / np.sqrt(num_events)

    def A_6_unc(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 5 * np.sqrt(np.mean(np.sin(phi)**2 * np.sin(2 * theta)**2) - A_6(cos_theta, phi)**2) / np.sqrt(num_events)

    def A_7_unc(cos_theta, phi):
        theta = np.arccos(cos_theta)
        return 5 * np.sqrt(np.mean(np.sin(2 * phi)**2 * np.sin(theta)**4) - A_7(cos_theta, phi)**2) / np.sqrt(num_events)
    

    # exp_cos_1 = np.mean(data1)
    # exp_sqr_cos_1 = np.mean(data1**2)
    # print(exp_cos_1)
    # print(exp_sqr_cos_1)

    # unc_cos_1 = np.sqrt((exp_sqr_cos_1 - exp_cos_1**2)/num_events) 
    # unc_cos_sqr_1 = np.sqrt((np.mean(data1**4) - exp_sqr_cos_1**2)/num_events)
    
    # exp_cos_3 = np.mean(data3)
    # exp_sqr_cos_3 = np.mean(data3**2)
    # print(exp_cos_3)
    # print(exp_sqr_cos_3)

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
    # fR_3 = -0.5 - (1/eta) * exp_cos_3 + 2.5 * exp_sqr_cos_3
    # fL_3 = -0.5 + (1/eta) * exp_cos_3 + 2.5 * exp_sqr_cos_3

    # sigma_f0_3 = 5 * unc_cos_sqr_3
    # sigma_fR_3 = np.sqrt( (unc_cos_3**2)/eta**2 + 2.5**2 * unc_cos_sqr_3**2 )
    
    # print(f"f0_3: {f0_3} +- {sigma_f0_3}")
    # print(f"fL_3: {fL_3} +- {sigma_fR_3}")
    # print(f"fR_3: {fR_3} +- {sigma_fR_3}")



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
        'alpha21': alpha21(),
        'alpha13': alpha13(),
        'alpha23': alpha23(),
        'gamma1010': gamma1010(),
        'gamma2020': gamma2020(),
        'alpha2m2_1' : Y2m2_1,
        'alpha2m2_3' : Y2m2_3,
        'gamma222m2' : gamma222m2_val,
        'gamma212m1' : gamma212m1_val,
        'gamma111m1' : gamma111m1_val,
        #--- New coefficients for e+ ---
        'A_0_e+': A_0(data1),
        'A_1_e+': A_1(data1, phi_data1),
        'A_2_e+': A_2(data1, phi_data1),
        'A_3_e+': A_3(data1, phi_data1),
        'A_4_e+': A_4(data1, phi_data1),
        'A_5_e+': A_5(data1, phi_data1),
        'A_6_e+': A_6(data1, phi_data1),
        'A_7_e+': A_7(data1, phi_data1),

        # --- New coefficients for mu- ---
        'A_0_mu-': A_0(data_mu_m),
        'A_1_mu-': A_1(data_mu_m, phi_data_mu_m),
        'A_2_mu-': A_2(data_mu_m, phi_data_mu_m),
        'A_3_mu-': A_3(data_mu_m, phi_data_mu_m),
        'A_4_mu-': A_4(data_mu_m, phi_data_mu_m),
        'A_5_mu-': A_5(data_mu_m, phi_data_mu_m),
        'A_6_mu-': A_6(data_mu_m, phi_data_mu_m),
        'A_7_mu-': A_7(data_mu_m, phi_data_mu_m),
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
        'gamma111m1' : sigma_g111m1,
        'A_0_e+': A_0_unc(data1),
        'A_1_e+': A_1_unc(data1, phi_data1),
        'A_2_e+': A_2_unc(data1, phi_data1),
        'A_3_e+': A_3_unc(data1, phi_data1),
        'A_4_e+': A_4_unc(data1, phi_data1),
        'A_5_e+': A_5_unc(data1, phi_data1),
        'A_6_e+': A_6_unc(data1, phi_data1),
        'A_7_e+': A_7_unc(data1, phi_data1),

        # --- New coefficients for mu- ---
        'A_0_mu-': A_0_unc(data_mu_m),
        'A_1_mu-': A_1_unc(data_mu_m, phi_data_mu_m),
        'A_2_mu-': A_2_unc(data_mu_m, phi_data_mu_m),
        'A_3_mu-': A_3_unc(data_mu_m, phi_data_mu_m),
        'A_4_mu-': A_4_unc(data_mu_m, phi_data_mu_m),
        'A_5_mu-': A_5_unc(data_mu_m, phi_data_mu_m),
        'A_6_mu-': A_6_unc(data_mu_m, phi_data_mu_m),
        'A_7_mu-': A_7_unc(data_mu_m, phi_data_mu_m),
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
theta_path_1 = os.path.join(WZ_path, "e+/theta_data_3.txt")
theta_path_2 = os.path.join(WZ_path, "mu+/theta_data_3.txt")
theta_path_3 = os.path.join(WZ_path, "mu-/theta_data_3.txt")
data1 = read_data(theta_path_1)
data2 = read_data(theta_path_2)
data3 = read_data(theta_path_3)

# Read phi values for mu+, e+, and mu- datasets
phi_path_1 = os.path.join(WZ_path, "e+/phi_data_3.txt")
phi_path_2 = os.path.join(WZ_path, "mu+/phi_data_3.txt")
phi_path_3 = os.path.join(WZ_path, "mu-/phi_data_3.txt")
phi_data1 = read_data(phi_path_1)
phi_data2 = read_data(phi_path_2)
phi_data3 = read_data(phi_path_3)

# Calculate parameters and uncertainties
parameters, uncertainties = calculate_parameters(data1, data2, data3, phi_data1, phi_data2, phi_data3)

# Write results to fracs_1.txt
output_file = os.path.join(WZ_path, "fracs_3.txt")
write_results_to_file(output_file, parameters, uncertainties)
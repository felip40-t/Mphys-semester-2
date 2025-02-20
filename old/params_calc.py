import os
import numpy as np
from histo_plotter import read_data

# Base path for WZ data
WZ_path = "/home/felipetcach/project/MG5_aMC_v3_5_6/WZ_new/Plots and data"

# Data file names for WZ process
WZ_files = ["theta_data_3.txt"]

# Function to calculate all parameters for WZ datasets
def calculate_parameters(data1, data2, num_events_1, num_events_2):
    """Calculates all parameters for the WZ case using two datasets (mu+ and e+)."""

    def calc_exp_cos(data):
        return np.mean(data)

    def calc_exp_cos_sqrd(data):
        return np.mean(np.square(data))

    def calc_exp_cos_4(data):
        return np.mean(np.power(data, 4))

    def calc_exp_cos_mix(data1, data2):
        return np.mean(data1 * data2)
    
    def calc_exp_cos_sqr_mix(data1, data2):
        return np.mean((data1**2) * (data2**2))

    def calc_f0(exp_sqr):
        return 2 - 5 * exp_sqr

    def calc_fR(exp, exp_sqr):
        return -0.5 + exp + 2.5 * exp_sqr

    def calc_fL(exp, exp_sqr):
        return -0.5 - exp + 2.5 * exp_sqr

    def calc_A0(exp_sqr):
        return 4 - 10 * exp_sqr

    def calc_A4(exp):
        return 4 * exp

    def calc_alpha_1(data):
        return 0.5 * (np.sqrt(3 / np.pi)) * calc_exp_cos(data)

    def calc_alpha_2(data):
        return 0.25 * np.sqrt(5 / np.pi) * (3 * calc_exp_cos_sqrd(data) - 1)

    def calc_gamma1010(data1, data2):
        return 0.25 * (3 / np.pi) * np.mean(data1 * data2)

    def calc_gamma2020(data1, data2):
        q1 = 3 * np.square(data1) - 1
        q2 = 3 * np.square(data2) - 1
        return (5 / (16 * np.pi)) * np.mean(q1 * q2)

    # Helper function to calculate uncertainties
    def calc_uncertainty(exp, exp_sqr, num_events):
        return np.sqrt(exp_sqr - exp ** 2) / np.sqrt(num_events)

    def calc_sqr_uncertainty(data, exp_sqr, num_events):
        return np.sqrt(calc_exp_cos_4(data) - exp_sqr ** 2) / np.sqrt(num_events)
    
    def calc_mix_uncertainty(data1, data2, exp_cos_1, exp_cos_2, sigma_cos_1, sigma_cos_2, num_events):
        cov = calc_exp_cos_mix(data1, data2) - exp_cos_1 * exp_cos_2
        first_term = (sigma_cos_1**2) * (sigma_cos_2**2)
        full_expr = first_term + 2 * cov
        print(f"Quantity squared: {full_expr}")
        print(f"Covariance: {cov}")
        
        return 0

    def calc_mix_sqr_uncertainty(data1, data2, sigma_cos_sqr_1, sigma_cos_sqr_2, num_events):
        first_term = 9 * sigma_cos_sqr_1 * sigma_cos_sqr_2
        cov_term1 = (16 * np.pi / 5) * calc_gamma2020(data1, data2)
        cov_term2 = np.mean( 3 * (np.square(data1)) - 1) * np.mean( 3 * (np.square(data2)) - 1)
        cov = cov_term1 - cov_term2
        sigma_sqr = first_term + 2 * cov
        return np.sqrt(sigma_sqr/num_events)
    
    # Calculate expectation values for both datasets
    exp_val_1, exp_sqr_val_1 = calc_exp_cos(data1), calc_exp_cos_sqrd(data1)
    exp_val_2, exp_sqr_val_2 = calc_exp_cos(data2), calc_exp_cos_sqrd(data2)

    # Calculate uncertainties for cos and cos^2 etc.
    sigma_cos_1, sigma_cos_sqr_1 = calc_uncertainty(exp_val_1, exp_sqr_val_1, num_events_1), calc_sqr_uncertainty(data1, exp_sqr_val_1, num_events_1)
    sigma_cos_2, sigma_cos_sqr_2 = calc_uncertainty(exp_val_2, exp_sqr_val_2, num_events_2), calc_sqr_uncertainty(data2, exp_sqr_val_2, num_events_2)
    #sigma_cos_mix = calc_mix_uncertainty(data1, data2, exp_val_1, exp_val_2, sigma_cos_1, sigma_cos_2, num_events_1)
    sigma_cos_sqr_mix = calc_mix_sqr_uncertainty(data1, data2, sigma_cos_sqr_1, sigma_cos_sqr_2, num_events_1)

    # Calculate f0, fR, fL, A0, A4 for both datasets
    parameters = {
        'f0_1': calc_f0(exp_sqr_val_1), 'fR_1': calc_fR(exp_val_1, exp_sqr_val_1), 'fL_1': calc_fL(exp_val_1, exp_sqr_val_1),
        'A0_1': calc_A0(exp_sqr_val_1), 'A4_1': 0.219 * calc_A4(exp_val_1),
        'f0_2': calc_f0(exp_sqr_val_2), 'fR_2': calc_fR(exp_val_2, exp_sqr_val_2), 'fL_2': calc_fL(exp_val_2, exp_sqr_val_2),
        'A0_2': calc_A0(exp_sqr_val_2), 'A4_2': calc_A4(exp_val_2)
    }

    # Calculate uncertainties for A0, A4, alpha1, alpha2
    uncertainties = {
        'A0_1_uncertainty': 10 * sigma_cos_sqr_1,
        'A0_2_uncertainty': 10 * sigma_cos_sqr_2,
        'A4_1_uncertainty': 0.219 * 4 * sigma_cos_1,
        'A4_2_uncertainty': 4 * sigma_cos_2,
        'alpha1_1_uncertainty': 0.219 * 0.5 * np.sqrt(3 / np.pi) * sigma_cos_1,
        'alpha1_2_uncertainty': 0.5 * np.sqrt(3 / np.pi) * sigma_cos_2,
        'alpha2_1_uncertainty': 0.25 * np.sqrt(5 / np.pi) * 3 * sigma_cos_sqr_1,
        'alpha2_2_uncertainty': 0.25 * np.sqrt(5 / np.pi) * 3 * sigma_cos_sqr_2,
        #'gamma1010_uncertainty': 0.219 * 0.25 * (3 / np.pi) * sigma_cos_mix, 
        'gamma2020_uncertainty': (5 / 16*np.pi) * sigma_cos_sqr_mix 
    }

    # Calculate alphas and gammas
    parameters.update({
        'alpha1_1': 0.219 * calc_alpha_1(data1), 'alpha2_1': calc_alpha_2(data1),
        'alpha1_2': calc_alpha_1(data2), 'alpha2_2': calc_alpha_2(data2),
        'gamma1010': 0.219 * calc_gamma1010(data1, data2), 'gamma2020': calc_gamma2020(data1, data2)
    })

    return parameters, uncertainties


# Helper function to write parameters and uncertainties to file
def write_results_to_file(output_file, params, uncertainties):
    with open(output_file, 'w') as f:
        f.write(f"f0 (mu+): {params['f0_1']}\n")
        f.write(f"fR (mu+): {params['fR_1']}\n")
        f.write(f"fL (mu+): {params['fL_1']}\n")
        f.write(f"A_0 (mu+): {params['A0_1']} +- {uncertainties['A0_1_uncertainty']}\n")
        f.write(f"A_4 (mu+): {params['A4_1']} +- {uncertainties['A4_1_uncertainty']}\n\n")
        
        f.write(f"f0 (e+): {params['f0_2']}\n")
        f.write(f"fR (e+): {params['fR_2']}\n")
        f.write(f"fL (e+): {params['fL_2']}\n")
        f.write(f"A_0 (e+): {params['A0_2']} +- {uncertainties['A0_2_uncertainty']}\n")
        f.write(f"A_4 (e+): {params['A4_2']} +- {uncertainties['A4_2_uncertainty']}\n\n")
        
        f.write(f"Alpha_1 (mu+): {params['alpha1_1']} +- {uncertainties['alpha1_1_uncertainty']}\n")
        f.write(f"Alpha_2 (mu+): {params['alpha2_1']} +- {uncertainties['alpha2_1_uncertainty']}\n")
        f.write(f"Alpha_1 (e+): {params['alpha1_2']} +- {uncertainties['alpha1_2_uncertainty']}\n")
        f.write(f"Alpha_2 (e+): {params['alpha2_2']} +- {uncertainties['alpha2_2_uncertainty']}\n")
        f.write(f"Gamma1010: {params['gamma1010']} \n") # +- {uncertainties['gamma1010_uncertainty']}\n")
        f.write(f"Gamma2020: {params['gamma2020']} +- {uncertainties['gamma2020_uncertainty']}\n")


# Main loop for processing datasets
for dataset in WZ_files:
    # Read cos(theta) values for mu+ and e+ datasets
    data_path_1 = os.path.join(WZ_path, "mu+", dataset)
    data_path_2 = os.path.join(WZ_path, "e+", dataset)

    cos_theta_values_1 = read_data(data_path_1)
    cos_theta_values_2 = read_data(data_path_2)

    # Number of events
    num_events_1, num_events_2 = len(cos_theta_values_1), len(cos_theta_values_2)

    # Calculate parameters and uncertainties
    parameters, uncertainties = calculate_parameters(cos_theta_values_1, cos_theta_values_2, num_events_1, num_events_2)

    # Write results to file
    output_file = os.path.join(WZ_path, f"parameters_{dataset.split('_')[-1].split('.')[0]}.txt")
    write_results_to_file(output_file, parameters, uncertainties)


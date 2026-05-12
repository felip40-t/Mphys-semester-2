""" Script to validate the generated events by checking that the angular coefficients for the whole phase space
match the literature values. This is a check to make sure that the event generation and kinematics calculations
are correct.

It also plots the histograms of the angular distributions to visually check that they match the expected shapes.

This validation is only for the ZZ system.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y

from diboson.config import ZZ_RAW_DIR, PROJECT_DIR

TESTS_PLOTS_DIR = PROJECT_DIR / "outputs" / "plots" / "tests"
TESTS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

validate_dict = {
    'a_110': 0.000,
    'a_120': 0.0299,
    'a_310': 0.000,
    'a_320': 0.0305,
    'a_12m2': -0.0167,
    'a_32m2': -0.0171,
    'g_1010': -0.00173,
    'g_2020': 0.00189,
}

NUM_BINS = 50


_LABEL_CONFIG = {
    'cos_theta': {
        'xlabel':   r'$\cos\theta$',
        'ylabel':   r'$1/\sigma{\cdot}d\sigma/d\cos\theta$',
        'xlim':     (-1, 1),
        'legend_1': r'$\cos\theta_1$',
        'legend_3': r'$\cos\theta_3$',
        'filename': 'cos_theta_test_histogram',
    },
    'phi': {
        'xlabel':   r'$\phi \; [\mathrm{rad}]$',
        'ylabel':   r'$1/\sigma{\cdot}d\sigma/d\phi$',
        'xlim':     (-np.pi, np.pi),
        'legend_1': r'$\phi_1$',
        'legend_3': r'$\phi_3$',
        'filename': 'phi_test_histogram',
    },
}


def plot_histogram(data_1, data_3, label):
    """ Plot two overlaid histograms (boson 1 and boson 3) for the given angular variable. """
    cfg = _LABEL_CONFIG[label]
    xlim = cfg['xlim']

    fig, ax = plt.subplots(figsize=(12, 8), dpi=800)

    ax.hist(data_1, bins=NUM_BINS, range=xlim, density=True,
            histtype='step', edgecolor='blue', label=cfg['legend_1'])
    ax.hist(data_3, bins=NUM_BINS, range=xlim, density=True,
            histtype='step', edgecolor='red', label=cfg['legend_3'])

    ax.set_xlabel(cfg['xlabel'], fontsize=20)
    ax.set_ylabel(cfg['ylabel'], fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='y', labelright=False, right=True)

    ax.grid(axis='y')
    ax.set_xlim(*xlim)

    ax.legend(loc='lower right', fontsize=20)

    fig.savefig(TESTS_PLOTS_DIR / f"{cfg['filename']}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {cfg['filename']}.pdf")


def calc_coefficients(theta_1, theta_3, phi_1, phi_3):
    """ Calculate the angular coefficients using the expectation values of spherical harmonics. 
        Due to conventions in the literature, the coefficients with azimuthal dependence (m != 0) 
        are multiplied by sqrt(2) to match the standard definitions.
    """
    a_110 = np.mean(sph_harm_y(1, 0, theta_1, phi_1))
    a_110_std = np.std(sph_harm_y(1, 0, theta_1, phi_1)) / np.sqrt(len(theta_1))
    a_120 = np.mean(sph_harm_y(2, 0, theta_1, phi_1))
    a_120_std = np.std(sph_harm_y(2, 0, theta_1, phi_1)) / np.sqrt(len(theta_1))
    a_310 = np.mean(sph_harm_y(1, 0, theta_3, phi_3))
    a_310_std = np.std(sph_harm_y(1, 0, theta_3, phi_3)) / np.sqrt(len(theta_3))
    a_320 = np.mean(sph_harm_y(2, 0, theta_3, phi_3))
    a_320_std = np.std(sph_harm_y(2, 0, theta_3, phi_3)) / np.sqrt(len(theta_3))
    a_12m2 = np.mean(sph_harm_y(2, -2, theta_1, phi_1)) * np.sqrt(2)
    a_12m2_std = np.std(sph_harm_y(2, -2, theta_1, phi_1) * np.sqrt(2)) / np.sqrt(len(theta_1))
    a_32m2 = np.mean(sph_harm_y(2, -2, theta_3, phi_3)) * np.sqrt(2)
    a_32m2_std = np.std(sph_harm_y(2, -2, theta_3, phi_3) * np.sqrt(2)) / np.sqrt(len(theta_3))
    g_1010 = np.mean(sph_harm_y(1, 0, theta_1, phi_1) * sph_harm_y(1, 0, theta_3, phi_3))
    g_1010_std = np.std(sph_harm_y(1, 0, theta_1, phi_1) * sph_harm_y(1, 0, theta_3, phi_3)) / np.sqrt(len(theta_1))
    g_2020 = np.mean(sph_harm_y(2, 0, theta_1, phi_1) * sph_harm_y(2, 0, theta_3, phi_3))
    g_2020_std = np.std(sph_harm_y(2, 0, theta_1, phi_1) * sph_harm_y(2, 0, theta_3, phi_3)) / np.sqrt(len(theta_1))
    return {
        'a_110': a_110,
        'a_120': a_120,
        'a_310': a_310,
        'a_320': a_320,
        'a_12m2': a_12m2,
        'a_32m2': a_32m2,
        'g_1010': g_1010,
        'g_2020': g_2020,
        'a_110_std': a_110_std,
        'a_120_std': a_120_std,
        'a_310_std': a_310_std,
        'a_320_std': a_320_std,
        'a_12m2_std': a_12m2_std,
        'a_32m2_std': a_32m2_std,
        'g_1010_std': g_1010_std,
        'g_2020_std': g_2020_std,
    }


if __name__ == "__main__":
    
    # Read angular distribution data for ZZ (whole-phase-space validation run)
    _data_dir = ZZ_RAW_DIR / "tests"
    theta_1 = np.load(_data_dir / "theta1.npy")
    theta_3 = np.load(_data_dir / "theta3.npy")
    cos_theta_1 = np.cos(theta_1)
    cos_theta_3 = np.cos(theta_3)
    phi_1 = np.load(_data_dir / "phi1.npy")
    phi_3 = np.load(_data_dir / "phi3.npy")

    plot_histogram(cos_theta_1, cos_theta_3, 'cos_theta')
    plot_histogram(phi_1, phi_3, 'phi')

    # Compare calculated coefficients to literature values
    calculated_coeffs = calc_coefficients(theta_1, theta_3, phi_1, phi_3)
    print("Calculated coefficients:")
    for key, _ in validate_dict.items():
        print(f"  {key}: {calculated_coeffs[key].real:.5f} +- {calculated_coeffs[f'{key}_std']:.5f} (target: {validate_dict[key]:.5f})")

import numpy as np
import matplotlib.pyplot as plt  # plotting
from astropy.io import ascii  # for table handling
from MCMC_inspiral_log import do_the_modelling as do_the_modelling_log
from MCMC_inspiral_log_variable_acc import do_the_modelling as do_the_modelling_log_var_acc
from MCMC_inspiral_log_variable_acc import calc_vals_simple
import os
import time
import warnings
import matplotlib
import pandas as pd
from Accreted_fracs_from_sims import get_value_for_mass

matplotlib.use('Agg')
warnings.filterwarnings('ignore')


def properties(eta, f_in, t_form, gal_vals, a=1, M_GC_min=10):
    M_GC_lim = a * gal_vals['R_gal']**2  # proportionality
    M_GC_lim = gal_vals['M_GC_diss']
    M_NSC_acc = eta*gal_vals['M_gal'] * \
        ((1+np.log(gal_vals['M_GC_max']/M_GC_lim))/(1+np.log10(M_GC_lim/M_GC_min)))
    M_NSC = M_NSC_acc/(1 - f_in)
    M_GC_tot = eta*gal_vals['M_gal'] - M_NSC_acc - gal_vals['M_GC_diss'] * \
        (1 + np.log10(gal_vals['M_GC_diss']/M_GC_min))
    R_NSC = 4.4e4 * f_in/t_form*M_NSC
    return M_NSC, R_NSC, M_GC_tot


def convert_to_log(val, e_val):
    log_val = np.log10(val)
    log_e_val = np.mean(
        [np.abs(np.log10(val + e_val) - log_val), np.abs(np.log10(val - e_val) - log_val)])
    return log_val, log_e_val


def do_for_galaxy(galaxy, redo=False, file='./Data/ACSFCS_sample.dat', mass_uncertainty=0.3, prefix='', steps=1000, parallel=False, cores=6):
    if os.path.isfile('./Results/{0}_results{1}.dat'.format(galaxy, prefix)) and not redo:
        print('done already')

        return 0
    # try:
    # if 1 = 1
    tab = ascii.read(file)

    gal = tab[tab['galaxy'] == galaxy]

    print('%%%%%%%%%%%%%%%%%%%%%% {0} %%%%%%%%%%%%%%%%%%%%%%%%%%%%'.format(galaxy))
    # do_the_modelling_log(np.log10(gal['M_NSC']), mass_uncertainty,
    #                     np.log10(gal['M_GCS']), mass_uncertainty,
    #                     galaxy=galaxy, file=file, prefix=prefix, steps=steps, parallel=parallel, cores=cores)
    do_the_modelling_log_var_acc(np.log10(gal['M_NSC']), mass_uncertainty,
                                 np.log10(gal['M_GCS']), mass_uncertainty,
                                 galaxy=galaxy, file=file, prefix=prefix+'_acc', steps=steps, parallel=parallel, cores=cores)

    # except:
    ##    print('Did not work for {0}'.format(galaxy))
    # return 0


def convert_seconds(t):
    hours = int(t/3600)
    minutes = int((t-hours*3600)/60)
    seconds = (t - hours*3600 - minutes*60)
    string = "{0} h {1} m {2:.2f} s".format(hours, minutes, seconds)
    return string


def calc_f_in_lims(theta):
    eta, f_in, f_acc, M_gal, M_GC_lim, M_GC_min, M_GC_max, M_GC_diss = theta
    M_gal_lin, M_GC_lim_lin, M_GC_min_lin, M_GC_max_lin, M_GC_diss_lin = np.power(
        10, theta[3:])
    eta = np.power(10, theta[0])
    # if not (0.00 < f_in <= 1):
    #    result = -np.inf
    M_NSC, M_GCS = calc_vals_simple(theta)

    M_NSC_lin, M_GCS_lin = np.power(10, M_NSC), np.power(10, M_GCS)

    f_in_lower_limit = 1 - ((eta*((1-f_acc)**2)*M_gal_lin*M_GCS_lin) /
                            (M_NSC_lin*M_GC_lim_lin*(1+np.log(M_GC_lim_lin/M_GC_diss_lin))))
    f_in_upper_limit = 1 - ((M_GC_lim_lin*M_GCS_lin)/(eta*M_gal_lin*M_NSC_lin))
    if f_in_lower_limit < 0:
        f_in_lower_limit = 0
    if f_in_upper_limit > 1:
        f_in_upper_limit = 1
    return f_in_lower_limit, f_in_upper_limit


if __name__ == "__main__":
    plt.close('all')

    #galaxy = 'FCC204'

    # for galaxy in tab['Name']:
    #    do_for_galaxy(galaxy)
    #    plt.close()s
    # plt.show()
    start = time.time()
    #file = './Data/LocalVolume_sample_to_fit2.dat'
    file = './Data/ACS_sample_to_fit3.dat'
    tab = ascii.read(file).to_pandas()

    tab = ascii.read(file)
    prefix = '_facc_lim'
    for galaxy in tab['galaxy']:
        start_i = time.time()
        do_for_galaxy(galaxy, file=file,
                      prefix=prefix, steps=1000, parallel=1, redo=1, cores=10)
        plt.close("all")
        end_i = time.time()
        duration_i = end_i-start_i
        print('This took {0}'.format(convert_seconds(duration_i)))
    end_i = time.time()
    duration_i = end_i-start
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print('This took {0}'.format(convert_seconds(duration_i)))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    #tab = ascii.read('./Data/Turner2012_NSCs_Fornax.txt')
    # for galaxy in tab['Name']:
    #    do_for_galaxy(galaxy, file='./Data/ACSVCS_sample_2nd_brightest.dat',  prefix='_2nd_brightest', steps=1000)
    #    plt.close("all")

    # FCC204
    # FCC335
    # print(np.random.randn(8))

    # do_the_modelling(np.log10(gal['M_NSC']), np.log10(gal['e_M_NSC']), np.log10(gal['M_GCS']),
    #                 np.log10(gal['e_M_GCS']), np.log10(gal['R_NSC']), np.log10(gal['e_R_NSC']))
    # do_the_modelling(gal['M_NSC'][0], gal['e_M_NSC'][0], gal['M_GCS'][0],
    #                 gal['e_M_GCS'][0], galaxy=galaxy)

    # log do_the_modelling
    #theta = np.array([0.01, 0.9, 10.8, 6.7, 2.5, 7.7, 4.7])

    # log_likelihood(theta, *convert_to_log(gal['M_NSC'][0], gal['e_M_NSC'][0]),
    #               *convert_to_log(gal['M_GCS'][0], gal['e_M_GCS'][0]))
    # log_likelihood(theta, np.log10(gal['M_NSC'][0]), 0.1,
    #               np.log10(gal['M_GCS'][0]), 0.1)
    # do_the_modelling_log(*convert_to_log(gal['M_NSC'], gal['e_M_NSC']),
    #                     *convert_to_log(gal['M_GCS'], gal['e_M_GCS']), galaxy=galaxy)
    #print(*convert_to_log(gal['M_NSC'], gal['e_M_NSC']))
    #print(*convert_to_log(gal['M_GCS'], gal['e_M_GCS']))
    #print(*convert_to_log(gal['M_GCS'], gal['e_M_GCS']))

"""
Project 2 in Ryan's presentation
"""

import numpy as np
import matplotlib.pyplot as plt  # plotting
from astropy.io import ascii  # for table handling
from MCMC_inspiral_log import do_the_modelling as do_the_modelling_log
from MCMC_inspiral_log import log_likelihood
import os
import warnings
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


def do_for_galaxy(galaxy, redo=False, file='./Data/ACSFCS_sample.dat', mass_uncertainty=0.3, prefix='', steps=1000):
    if os.path.isfile('./Results/{0}_results{1}.dat'.format(galaxy, prefix)) and not redo:
        print('done already')

        return 0
    # try:
    # if 1 = 1
    tab = ascii.read(file)

    gal = tab[tab['galaxy'] == galaxy]
    print(file)
    print('%%%%%%%%%%%%%%%%%%%%%% {0} %%%%%%%%%%%%%%%%%%%%%%%%%%%%'.format(galaxy))
    do_the_modelling_log(np.log10(gal['M_NSC']), mass_uncertainty,
                         np.log10(gal['M_GCS']), mass_uncertainty,
                         galaxy=galaxy, file=file, prefix=prefix, steps=steps)

    # except:
    ##    print('Did not work for {0}'.format(galaxy))
    # return 0


if __name__ == "__main__":
    plt.close('all')

    #galaxy = 'FCC204'
    tab = ascii.read('./Data/Cote2006_NSCs_Virgo.txt')
    # for galaxy in tab['Name']:
    #    do_for_galaxy(galaxy)
    #    plt.close()
    #plt.show()
    #do_for_galaxy('FCC47', redo=True, file='./Data/ACSFCS_sample_2nd_brightest.dat',
    #              prefix='_2nd_brightest', steps=1000)
    #tab = ascii.read('./Data/Turner2012_NSCs_Fornax.txt')
    for galaxy in tab['Name']:
        do_for_galaxy(galaxy, file='./Data/ACSVCS_sample_2nd_brightest.dat',  prefix='_2nd_brightest', steps=1000)
        plt.close("all")

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

from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt  # plotting
from astropy.io import ascii, fits
import os
from astropy.table import Table

from astropy.table import Column
from collections import OrderedDict
from scipy.optimize import minimize
import emcee
import corner
from scipy.interpolate import interp1d
import seaborn as sns

# fitting R and M (project 2) R >= 4.4e4 f_in/t_form * M_NSC


def get_value_for_mass(M_gal=10):
    m, f_ex = np.loadtxt('./Data/F_ex_situ.dat', unpack=1)
    m_p, f_ex_p = np.loadtxt('./Data/F_ex_situ_p.dat', unpack=1)
    m_m, f_ex_m = np.loadtxt('./Data/F_ex_situ_m.dat', unpack=1)

    m = np.log10(m)
    m_p = np.log10(m_p)
    m_m = np.log10(m_m)

    f_ex_ip = interp1d(m, f_ex)
    f_ex_ip_m = interp1d(m_m, f_ex_m)
    f_ex_ip_p = interp1d(m_p, f_ex_p)
    if M_gal >= 8.7 and M_gal <= 12.3:
        f_acc = f_ex_ip(M_gal)
    elif M_gal < 8.7:
        f_acc = f_ex_ip(8.7)
    elif M_gal > 12.3:
        f_acc = f_ex_ip(12.3)
    return f_acc


def get_value_for_mass_lims(M_gal=10):
    m, f_ex = np.loadtxt('./Data/F_ex_situ.dat', unpack=1)
    m_p, f_ex_p = np.loadtxt('./Data/F_ex_situ_p.dat', unpack=1)
    m_m, f_ex_m = np.loadtxt('./Data/F_ex_situ_m.dat', unpack=1)

    m = np.log10(m)
    m_p = np.log10(m_p)
    m_m = np.log10(m_m)

    f_ex_ip_m = interp1d(m_m, f_ex_m)
    f_ex_ip_p = interp1d(m_p, f_ex_p)
    if M_gal >= 8.7 and M_gal <= 12.3:
        f_acc_m = f_ex_ip_m(M_gal)
        f_acc_p = f_ex_ip_p(M_gal)
    elif M_gal < 8.7:
        f_acc_m = f_ex_ip_m(8.7)
        f_acc_p = f_ex_ip_p(8.7)
    elif M_gal > 12.3:
        f_acc_m = f_ex_ip_m(12.3)
        f_acc_p = f_ex_ip_p(12.3)
    return f_acc_m, f_acc_p


if __name__ == "__main__":
    plt.close('all')
    plt.rc('font', family='serif')

    # scale all the galaxy masses and globular system masses by f _acc
    file = './Data/ACS_sample_to_fit3.dat'
    tab = ascii.read(file)

    f_accs = np.zeros(len(tab))
    for i in range(len(f_accs)):
        f_accs[i] = get_value_for_mass(np.log10(tab['M_gal'][i]))

    tab['M_gal'] = tab['M_gal'] * f_accs
    tab['M_GCS'] = tab['M_GCS'] * f_accs
    tab.write('./Data/ACS_sample_scaled_accretion.dat', format='ascii')

    file = './Data/ACS_sample_to_fit3.dat'
    tab1 = ascii.read(file)

    fig, ax = plt.subplots()
    ax.scatter(np.log10(tab1['M_GCS']), np.log10(tab['M_GCS']))

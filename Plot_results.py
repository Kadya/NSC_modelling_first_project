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

# fitting R and M (project 2) R >= 4.4e4 f_in/t_form * M_NSC


def collect_res_ACSFCS():
    tab = ascii.read('../Data/Turner2012_NSCs_Fornax.txt')
    res_acsfcs = ascii.read('./Results/{0}_results_pGC05.dat'.format('FCC170'))
    names = res_acsfcs.colnames
    all_res = np.zeros((len(tab), len(names)))
    galaxies = []
    for i in range(len(tab['Name'])):
        if tab['Name'][i] not in ['FCC95']:
            gal = tab['Name'][i]
            res_acsfcs = ascii.read('./Results/{0}_results.dat'.format(gal))
            data = np.array([res_acsfcs[col][0] for col in res_acsfcs.colnames])
            all_res[i, :] = data
        galaxies.append(tab['Name'][i])

    tab = Table(all_res, names=names)

    col = Column(galaxies, 'galaxy')

    tab.add_column(col, index=0)
    tab.write('ACSFCS_sample_results_pGC05.dat', format='ascii')


def collect_res_ACSVCS():
    tab = ascii.read('../Data/Cote2006_NSCs_Virgo.txt')
    res_acsfcs = ascii.read('./Results/{0}_results.dat'.format('VCC538'))
    names = res_acsfcs.colnames
    all_res = np.zeros((len(tab), len(names)))
    galaxies = []
    for i in range(len(tab['Name'])):
        gal = tab['Name'][i]
        try:
            res_acsfcs = ascii.read('./Results/{0}_results.dat'.format(gal))
            data = np.array([res_acsfcs[col][0] for col in res_acsfcs.colnames])
            all_res[i, :] = data
            galaxies.append(tab['Name'][i])
        except:
            print('Not found for {0}'.format(gal))
            galaxies.append(tab['Name'][i])

    tab = Table(all_res, names=names)

    col = Column(galaxies, 'galaxy')

    tab.add_column(col, index=0)
    tab.write('ACSVCS_sample_results.dat', format='ascii')


if __name__ == "__main__":
    plt.close('all')

    #galaxy = 'FCC190'
    # collect_res_ACSVCS()
    # collect_res_ACSFCS()
    #a = b
    plt.rc('font', family='serif')
    #a = b

    fig, (ax) = plt.subplots(ncols=3, nrows=3, sharey=True, figsize=[13, 10])
    # fig2, ax2 = plt.subplots()
    res_acsfcs = ascii.read('ACSFCS_sample_results.dat')
    res_acsvcs = ascii.read('ACSVCS_sample_results.dat')

    res_acsvcs = ascii.read('ACSFCS_sample_results_pGC05.dat')

    gals_acsfcs = ascii.read('../Preparation/ACSFCS_sample.dat')
    gals_acsvcs = ascii.read('../Preparation/ACSVCS_sample.dat')

    gals_acsvcs = ascii.read('../Preparation/ACSFCS_sample_pGC05.dat')

    '''
    mask = res_acsvcs['galaxy'] != 'VCC230'
    res_acsvcs = res_acsvcs[mask]
    gals_acsvcs = gals_acsvcs[mask]
    '''

    # for i in range(len(gals_acsfcs)):
    #    print(gals_acsfcs['galaxy'][i], res_acsfcs['galaxy'][i])
    for i in range(len(gals_acsvcs)):
        print(gals_acsvcs['galaxy'][i], res_acsvcs['galaxy'][i])
    mask = res_acsfcs['galaxy'] != 'FCC95'
    res_acsfcs = res_acsfcs[mask]
    gals_acsfcs = gals_acsfcs[mask]

    color = 'orange'
    color2 = 'purple'
    alpha = 0.2
    marker = '^'
    marker2 = 'v'

    ax[0, 0].errorbar(np.log10(gals_acsfcs['M_gal']), res_acsfcs['f_in'], yerr=[
        res_acsfcs['f_in_m'], res_acsfcs['f_in_p']], color=color, alpha=alpha, fmt='o', markersize=0, label='')
    ax[0, 0].scatter(np.log10(gals_acsfcs['M_gal']),
                     res_acsfcs['f_in'], color=color, zorder=3, marker=marker, label='Fornax')
    ax[0, 0].errorbar(np.log10(gals_acsvcs['M_gal']), res_acsvcs['f_in'], yerr=[
        res_acsvcs['f_in_m'], res_acsvcs['f_in_p']], color=color2, alpha=alpha, fmt='o', markersize=0, label='')
    ax[0, 0].scatter(np.log10(gals_acsvcs['M_gal']),
                     res_acsvcs['f_in'], color=color2, zorder=3, marker=marker, label='Virgo')
    #
    ax[0, 1].errorbar(np.log10(gals_acsfcs['M_NSC']), res_acsfcs['f_in'], yerr=[
        res_acsfcs['f_in_m'], res_acsfcs['f_in_p']], color=color, alpha=alpha, fmt='o', markersize=0, label='')
    ax[0, 1].scatter(np.log10(gals_acsfcs['M_NSC']),
                     res_acsfcs['f_in'], color=color, zorder=3, marker=marker, label='Fornax')
    ax[0, 1].errorbar(np.log10(gals_acsvcs['M_NSC']), res_acsvcs['f_in'], yerr=[
        res_acsvcs['f_in_m'], res_acsvcs['f_in_p']], color=color2, alpha=alpha, fmt='o', markersize=0, label='')
    ax[0, 1].scatter(np.log10(gals_acsvcs['M_NSC']),
                     res_acsvcs['f_in'], color=color2, zorder=3, marker=marker, label='Virgo')
    #
    ax[0, 2].errorbar(np.log10(gals_acsfcs['M_GCS']), res_acsfcs['f_in'], yerr=[
        res_acsfcs['f_in_m'], res_acsfcs['f_in_p']], color=color, alpha=alpha, fmt='o', markersize=0, label='')
    ax[0, 2].scatter(np.log10(gals_acsfcs['M_GCS']),
                     res_acsfcs['f_in'], color=color, zorder=3, marker=marker, label='Fornax')
    ax[0, 2].errorbar(np.log10(gals_acsvcs['M_GCS']), res_acsvcs['f_in'], yerr=[
        res_acsvcs['f_in_m'], res_acsvcs['f_in_p']], color=color2, alpha=alpha, fmt='o', markersize=0, label='')
    ax[0, 2].scatter(np.log10(gals_acsvcs['M_GCS']),
                     res_acsvcs['f_in'], color=color2, zorder=3, marker=marker, label='Virgo')

    ax[2, 0].errorbar(res_acsfcs['M_GC_lim'], res_acsfcs['f_in'], yerr=[
        res_acsfcs['f_in_m'], res_acsfcs['f_in_p']], color=color, alpha=alpha, fmt='o', markersize=0, label='')
    ax[2, 0].scatter(res_acsfcs['M_GC_lim'],
                     res_acsfcs['f_in'], color=color, zorder=3, marker=marker, label='Fornax')
    ax[2, 0].errorbar(res_acsvcs['M_GC_lim'], res_acsvcs['f_in'], yerr=[
        res_acsvcs['f_in_m'], res_acsvcs['f_in_p']], color=color2, alpha=alpha, fmt='o', markersize=0, label='')
    ax[2, 0].scatter(res_acsvcs['M_GC_lim'],
                     res_acsvcs['f_in'], color=color2, zorder=3, marker=marker, label='Virgo')

    ax[1, 0].errorbar(np.log10(gals_acsfcs['M_NSC']/gals_acsfcs['M_gal']), res_acsfcs['f_in'], yerr=[
        res_acsfcs['f_in_m'], res_acsfcs['f_in_p']], color=color, alpha=alpha, fmt='o', markersize=0, label='')
    ax[1, 0].scatter(np.log10(gals_acsfcs['M_NSC']/gals_acsfcs['M_gal']),
                     res_acsfcs['f_in'], color=color, zorder=3, marker=marker, label='Fornax')
    ax[1, 0].errorbar(np.log10(gals_acsvcs['M_NSC']/gals_acsvcs['M_gal']), res_acsvcs['f_in'], yerr=[
        res_acsvcs['f_in_m'], res_acsvcs['f_in_p']], color=color2, alpha=alpha, fmt='o', markersize=0, label='')
    ax[1, 0].scatter(np.log10(gals_acsvcs['M_NSC']/gals_acsvcs['M_gal']),
                     res_acsvcs['f_in'], color=color2, zorder=3, marker=marker, label='Virgo')

    ax[1, 1].errorbar(np.log10(gals_acsfcs['M_GCS']/gals_acsfcs['M_gal']), res_acsfcs['f_in'], yerr=[
        res_acsfcs['f_in_m'], res_acsfcs['f_in_p']], color=color, alpha=alpha, fmt='o', markersize=0, label='')
    ax[1, 1].scatter(np.log10(gals_acsfcs['M_GCS']/gals_acsfcs['M_gal']),
                     res_acsfcs['f_in'], color=color, zorder=3, marker=marker, label='Fornax')
    ax[1, 1].errorbar(np.log10(gals_acsvcs['M_GCS']/gals_acsvcs['M_gal']), res_acsvcs['f_in'], yerr=[
        res_acsvcs['f_in_m'], res_acsvcs['f_in_p']], color=color2, alpha=alpha, fmt='o', markersize=0, label='')
    ax[1, 1].scatter(np.log10(gals_acsvcs['M_GCS']/gals_acsvcs['M_gal']),
                     res_acsvcs['f_in'], color=color2, zorder=3, marker=marker, label='Virgo')

    ax[2, 1].errorbar(res_acsfcs['M_cl_max'], res_acsfcs['f_in'], yerr=[
        res_acsfcs['f_in_m'], res_acsfcs['f_in_p']], xerr=[
            res_acsfcs['M_cl_max_m'], res_acsfcs['M_cl_max_p']], color=color, alpha=alpha, fmt='o', markersize=0, label='')
    ax[2, 1].scatter(res_acsfcs['M_cl_max'],
                     res_acsfcs['f_in'], color=color, zorder=3, marker=marker, label='Fornax')
    ax[2, 1].errorbar(res_acsvcs['M_cl_max'], res_acsvcs['f_in'], yerr=[
        res_acsvcs['f_in_m'], res_acsvcs['f_in_p']], xerr=[
            res_acsvcs['M_cl_max_m'], res_acsvcs['M_cl_max_p']], color=color2, alpha=alpha, fmt='o', markersize=0, label='')
    ax[2, 1].scatter(res_acsvcs['M_cl_max'],
                     res_acsvcs['f_in'], color=color2, zorder=3, marker=marker, label='Virgo')

    ax[1, 2].errorbar(np.log10(gals_acsfcs['M_NSC']/gals_acsfcs['M_GCS']), res_acsfcs['f_in'], yerr=[
        res_acsfcs['f_in_m'], res_acsfcs['f_in_p']], color=color, alpha=alpha, fmt='o', markersize=0, label='')
    ax[1, 2].scatter(np.log10(gals_acsfcs['M_NSC']/gals_acsfcs['M_GCS']),
                     res_acsfcs['f_in'], color=color, zorder=3, marker=marker, label='Fornax')
    ax[1, 2].errorbar(np.log10(gals_acsvcs['M_NSC']/gals_acsvcs['M_GCS']), res_acsvcs['f_in'], yerr=[
        res_acsvcs['f_in_m'], res_acsvcs['f_in_p']],  color=color2, alpha=alpha, fmt='o', markersize=0, label='')
    ax[1, 2].scatter(np.log10(gals_acsvcs['M_NSC']/gals_acsvcs['M_GCS']),
                     res_acsvcs['f_in'], color=color2, zorder=3, marker=marker, label='Virgo')
    '''
    ax[2, 1].errorbar(res_acsfcs['M_cl_min'], res_acsfcs['f_in'], yerr=[
        res_acsfcs['f_in_m'], res_acsfcs['f_in_p']],  color=color, alpha=alpha, fmt='o', markersize=0, label='')
    ax[2, 1].scatter(res_acsfcs['M_cl_min'],
                     res_acsfcs['f_in'], color=color, zorder=3, marker=marker, label='Fornax')
    ax[2, 1].errorbar(res_acsvcs['M_cl_min'], res_acsvcs['f_in'], yerr=[
        res_acsvcs['f_in_m'], res_acsvcs['f_in_p']],  color=color2, alpha=alpha, fmt='o', markersize=0, label='')
    ax[2, 1].scatter(res_acsvcs['M_cl_min'],
                     res_acsvcs['f_in'], color=color2, zorder=3, marker=marker, label='Virgo')
    '''

    ax[2, 2].errorbar(res_acsfcs['eta'], res_acsfcs['f_in'], yerr=[
        res_acsfcs['f_in_m'], res_acsfcs['f_in_p']], color=color, alpha=alpha, fmt='o', markersize=0, label='')
    ax[2, 2].scatter(res_acsfcs['eta'],
                     res_acsfcs['f_in'], color=color, zorder=3, marker=marker, label='Fornax')

    ax[2, 2].errorbar(res_acsvcs['eta'], res_acsvcs['f_in'], yerr=[
        res_acsvcs['f_in_m'], res_acsvcs['f_in_p']], color=color2, alpha=alpha, fmt='o', markersize=0, label='')
    ax[2, 2].scatter(res_acsvcs['eta'],
                     res_acsvcs['f_in'], color=color2, zorder=3, marker=marker, label='Virgo')

    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            ax[i, j].legend()

    ax[0, 0].set_ylim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[0, 0].set_ylabel(r'$f_{\rm{in}}$', fontsize=12)
    ax[1, 0].set_ylabel(r'$f_{\rm{in}}$', fontsize=12)
    ax[2, 0].set_ylabel(r'$f_{\rm{in}}$', fontsize=12)
    ax[2, 0].set_ylim(0, 1)
    ax[0, 0].set_xlabel(r'log($M_{\rm{gal}}/M_\odot)$', fontsize=12)

    # ax2.set_ylabel(r'$f_{\rm{in}}$', fontsize=12)
    ax[0, 1].set_xlabel(r'log($M_{\rm{NSC}}/M_\odot)$', fontsize=12)
    ax[0, 2].set_xlabel(r'log($M_{\rm{GCS}}/M_\odot)$', fontsize=12)

    ax[2, 0].set_xlabel(r'log($M_{\rm{GC, lim}})$', fontsize=12)
    ax[1, 0].set_xlabel(r'log($M_{\rm{NSC}}/M_{\rm{gal}}$)', fontsize=12)
    ax[1, 1].set_xlabel(r'log($M_{\rm{GCS}}/M_{\rm{gal}}$)', fontsize=12)

    ax[2, 1].set_xlabel(r'log($M_{\rm{cl, max}})$', fontsize=12)
    ax[1, 2].set_xlabel(r'log($M_{\rm{NSC}}/M_{\rm{GCS}}$)', fontsize=12)

    #ax[2, 1].set_xlabel(r'log($M_{\rm{cl, min}})$', fontsize=12)
    ax[2, 2].set_xlabel(r'$\eta$', fontsize=12)
    fig.tight_layout()
    plt.savefig('ACSFCS_ACSVCS_results.png', dpi=300)

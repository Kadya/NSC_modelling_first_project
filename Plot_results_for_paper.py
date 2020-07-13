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
import seaborn as sns

# fitting R and M (project 2) R >= 4.4e4 f_in/t_form * M_NSC


def collect_res_ACSFCS(prefix=''):
    tab = ascii.read('../Data/Turner2012_NSCs_Fornax.txt')
    res_acsfcs = ascii.read('./Results/{0}_results{1}.dat'.format('FCC170', prefix))
    names = res_acsfcs.colnames
    all_res = np.zeros((len(tab), len(names)))
    galaxies = []
    for i in range(len(tab['Name'])):
        if tab['Name'][i] not in ['FCC95']:
            gal = tab['Name'][i]
            res_acsfcs = ascii.read('./Results/{0}_results{1}.dat'.format(gal, prefix))
            data = np.array([res_acsfcs[col][0] for col in res_acsfcs.colnames])
            all_res[i, :] = data
        galaxies.append(tab['Name'][i])

    tab = Table(all_res, names=names)

    col = Column(galaxies, 'galaxy')

    tab.add_column(col, index=0)
    tab.write('ACSFCS_sample_results{0}.dat'.format(prefix), format='ascii')


def collect_res_ACSVCS(prefix=''):
    tab = ascii.read('../Data/Cote2006_NSCs_Virgo.txt')
    res_acsfcs = ascii.read('./Results/{0}_results{1}.dat'.format('VCC538', prefix))
    names = res_acsfcs.colnames
    all_res = np.zeros((len(tab), len(names)))
    galaxies = []
    for i in range(len(tab['Name'])):
        gal = tab['Name'][i]
        try:
            res_acsfcs = ascii.read('./Results/{0}_results{1}.dat'.format(gal, prefix))
            data = np.array([res_acsfcs[col][0] for col in res_acsfcs.colnames])
            all_res[i, :] = data
            galaxies.append(tab['Name'][i])
        except:
            print('Not found for {0}'.format(gal))
            galaxies.append(tab['Name'][i])

    tab = Table(all_res, names=names)

    col = Column(galaxies, 'galaxy')

    tab.add_column(col, index=0)
    tab.write('ACSVCS_sample_results{0}.dat'.format(prefix), format='ascii')


def collect_res_ACSFCS(prefix=''):
    tab = ascii.read('../Data/Turner2012_NSCs_Fornax.txt')
    res_acsfcs = ascii.read('./Results/{0}_results{1}.dat'.format('FCC170', prefix))
    names = res_acsfcs.colnames
    all_res = np.zeros((len(tab), len(names)))
    galaxies = []
    for i in range(len(tab['Name'])):
        if tab['Name'][i] not in ['FCC95']:
            gal = tab['Name'][i]
            res_acsfcs = ascii.read('./Results/{0}_results{1}.dat'.format(gal, prefix))
            data = np.array([res_acsfcs[col][0] for col in res_acsfcs.colnames])
            all_res[i, :] = data
        galaxies.append(tab['Name'][i])

    tab = Table(all_res, names=names)

    col = Column(galaxies, 'galaxy')

    tab.add_column(col, index=0)
    tab.write('ACSFCS_sample_results{0}.dat'.format(prefix), format='ascii')


def collect_res_acs(prefix=''):
    tab = ascii.read('./Data/ACS_sample_to_fit2.dat')
    res_acsfcs = ascii.read('./Results/{0}_results{1}.dat'.format('VCC538', prefix))
    names = res_acsfcs.colnames
    all_res = np.zeros((len(tab), len(names)))
    galaxies = []
    environment = []
    for i in range(len(tab['galaxy'])):
        gal = tab['galaxy'][i]
        if gal[0] == 'V':
            environment.append('Virgo')
        else:
            environment.append('Fornax')
        try:
            res_acsfcs = ascii.read('./Results/{0}_results{1}.dat'.format(gal, prefix))
            data = np.array([res_acsfcs[col][0] for col in res_acsfcs.colnames])
            all_res[i, :] = data
            galaxies.append(tab['galaxy'][i])
        except:
            print('Not found for {0}'.format(gal))
            galaxies.append(tab['galaxy'][i])

    tab = Table(all_res, names=names)

    col = Column(galaxies, 'galaxy')
    environment_col = Column(environment, 'Environment')

    tab.add_column(col, index=0)
    #environment = Column(np.full_like(galaxies, 'Fornax'), 'Environment')
    tab.add_column(environment_col, index=-1)
    #tab['Environment']['VCC' in tab['galaxy']] = 'Virgo'
    tab.write('ACS_sample_results{0}.dat'.format(prefix), format='ascii')


def collect_results_local(prefix=''):
    tab = ascii.read('./Data/LocalVolume_sample_to_fit.dat')
    res_local = ascii.read('./Results/{0}_results{1}.dat'.format('UGC8638', prefix))
    names = res_local.colnames
    all_res = np.zeros((len(tab), len(names)))
    galaxies = []
    for i in range(len(tab['galaxy'])):
        gal = tab['galaxy'][i]
        try:
            res_local = ascii.read('./Results/{0}_results{1}.dat'.format(gal, prefix))
            data = np.array([res_local[col][0] for col in res_local.colnames])
            all_res[i, :] = data
            galaxies.append(tab['galaxy'][i])
        except:
            print('Not found for {0}'.format(gal))
            galaxies.append(tab['galaxy'][i])

    tab = Table(all_res, names=names)
    col = Column(galaxies, 'galaxy')

    tab.add_column(col, index=0)
    tab.write('LocalVolume_sample_results{0}.dat'.format(prefix), format='ascii')


def create_big_table(prefix=''):
    res_acsfcs = ascii.read('ACSFCS_sample_results{0}.dat'.format(prefix)).to_pandas()
    res_acsvcs = ascii.read('ACSVCS_sample_results{0}.dat'.format(prefix)).to_pandas()

    gals_acsfcs = ascii.read('../Preparation/ACSFCS_sample.dat').to_pandas()
    gals_acsvcs = ascii.read('../Preparation/ACSVCS_sample.dat').to_pandas()

    acsfcs = gals_acsfcs.merge(res_acsfcs, on='galaxy', suffixes=('', '_fit'))
    acsfcs['Environment'] = 'Fornax'
    acsvcs = gals_acsvcs.merge(res_acsvcs, on='galaxy', suffixes=('', '_fit'))
    acsvcs['Environment'] = 'Virgo'

    tab = acsfcs.append(acsvcs, sort=1)
    tab['f_in'][tab['f_in'] == 0] = np.nan
    return tab


def plot(tab, ax, s1='M_gal', s2='f_in', errorbar=True, x_errs=True, alpha=0.3, label='default', x_label=r'log($M_{\rm{gal}}/M_\odot)$',
         y_label=r'$f_{\rm{in}}$', color='darkred', marker='o', logi=True):
    # check for fracs
    if '/' in s1:
        s1_0, s1_1 = s1.split('/')
        x = np.log10(tab[s1_0]) - np.log10(tab[s1_1])
        yerr = [
            tab['{0}_m'.format(s2)], tab['{0}_p'.format(s2)]]
        # xerr = x*np.sqrt((0.3/np.log10(tab[s1_0]))**2 + (0.3/np.log10(tab[s1_1]))**2)
        xerr = np.sqrt(2*0.3**2)
        if not logi:
            x = tab[s1_0] - tab[s1_1]
    else:
        x = np.log10(tab[s1])
        yerr = [
            tab['{0}_m'.format(s2)], tab['{0}_p'.format(s2)]]
        try:
            xerr = [tab['{0}_m'.format(s1)], tab['{0}_p'.format(s1)]]
        except KeyError:
            xerr = 0.3
        if not logi:
            x = tab[s1]

    ax.scatter(x, tab[s2], edgecolor='k',
               facecolor=color, label=label, marker=marker, zorder=3)
    if errorbar:
        if not x_errs:
            xerr = None
        ax.errorbar(x, tab[s2], yerr=yerr, xerr=xerr, fmt='o', markersize=0,
                    label='', alpha=alpha, color=color, zorder=0)

    return ax


def stellar_pop_stuff():

    tab = create_big_table()
    tab_2nd = create_big_table('_2nd_brightest')

    fig, ax = plt.subplots()
    ax = plot(tab, label='brightest GC')
    ax = plot(tab_2nd, label='2nd brightest GC', color='mediumblue')

    fig, ax = plt.subplots()
    ax = plot(tab, s1='M_NSC', label='brightest GC', x_label=r'log($M_{\rm{NSC}}/M_\odot)$')
    ax = plot(tab_2nd, s1='M_NSC', label='2nd brightest GC',
              color='mediumblue', x_label=r'log($M_{\rm{NSC}}/M_\odot)$')

    tab_sp = ascii.read('F3D_NSC_metallicities.dat')

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots(ncols=2, sharey=True, figsize=[8, 4])

    ax = plot(tab, s1='M_NSC', label='', color='dimgrey', x_label=r'log($M_{\rm{NSC}}/M_\odot)$')
    for i in range(len(tab_sp)):
        mask = tab['galaxy'] == tab_sp['galaxy'][i]
        if len(tab[mask]) > 0:
            ax.scatter(np.log10(tab['M_NSC'][mask]), tab['f_in'][mask],
                       c=tab_sp['metals'][i], vmin=-0.8, vmax=0.4, cmap='plasma', zorder=100)
            scat = ax2.scatter(tab_sp['ages'][i] - tab_sp['ages_gal'][i], tab_sp['metals']
                               [i] - tab_sp['metals_gal'][i], c=tab['f_in'][mask], vmin=0.2, vmax=1)
            # ax2.annotate(tab_sp['galaxy'][i], (tab_sp['ages'][i] - tab_sp['ages_gal']
            #                                   [i], tab_sp['metals'][i] - tab_sp['metals_gal'][i]))
            ax3[0].scatter(tab_sp['metals'][i] - tab_sp['metals_gal'][i], np.log10(tab['M_NSC'][mask]/tab['M_gal'][mask]),
                           c=tab['f_in'][mask], vmin=0.2, vmax=1, cmap='plasma', zorder=100)
            ax3[1].scatter(tab_sp['ages'][i] - tab_sp['ages_gal'][i], np.log10(tab['M_NSC'][mask]/tab['M_gal'][mask]),
                           c=tab['f_in'][mask], vmin=0.2, vmax=1, cmap='plasma', zorder=100)
    fig2.colorbar(scat, label=r'$f_{\rm{in}}$')
    ax2.set_xlabel(r'Age$_{\rm{NSC}}$ -  Age$_{\rm{gal}}$ [Gyr]', fontsize=14)
    ax2.set_ylabel(r'[M/H]$_{\rm{NSC}}$ -  [M/H]$_{\rm{gal}}$', fontsize=14)
    ax2.set_xlim(-6.5, 6.5)
    ax2.set_ylim(-0.65, 0.65)
    ax2.axvline(0, c='k', ls='--')
    ax2.axhline(0, c='k', ls='--')
    ax3[0].axvline(0, c='k', ls='--')
    ax3[1].axvline(0, c='k', ls='--')

    ax3[0].set_xlim(-0.65, 0.65)
    ax3[1].set_xlim(-6.5, 6.5)

    tab_phot_acsfcs = ascii.read('../PredictMDF/ACSFCS_photometric_metallicities.dat').to_pandas()
    tab_phot_acsvcs = ascii.read('../PredictMDF/ACSVCS_photometric_metallicities.dat').to_pandas()
    tab_phot = tab_phot_acsfcs.append(tab_phot_acsvcs, sort=1)

    tab = tab.merge(tab_phot, on='galaxy')

    fig, ax = plt.subplots()
    for i in range(len(tab)):
        mask = tab_sp['galaxy'] == tab['galaxy'][i]
        if len(tab_sp[mask]) > 0:
            ax.scatter(tab_sp['metals'][mask], tab['f_in'][i], marker='*', s=80, color='darkorange')
            # ax.annotate(tab['galaxy'][i], (tab_sp['metals'][mask], tab['f_in'][i]))
        else:
            ax.scatter(tab['MH_NSC'][i], tab['f_in'][i],
                       marker='o', facecolor='None', edgecolor='k')

    fig, ax = plt.subplots()
    for i in range(len(tab)):
        mask = tab_sp['galaxy'] == tab['galaxy'][i]
        if len(tab_sp[mask]) > 0:
            ax.scatter(tab['gz_NSC'][i], tab_sp['metals'][mask],
                       marker='*', s=80, color='darkorange')
            # ax.annotate(tab['galaxy'][i], (tab_sp['metals'][mask], tab['f_in'][i]))
        else:
            ax.scatter(tab['gz_NSC'][i], tab['MH_NSC'][i],
                       marker='o', facecolor='None', edgecolor='k')

    fig, ax = plt.subplots()
    ax.scatter(tab['gz_NSC'], tab['f_in'])


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)


def log(x):
    return np.log10(x)


def plot_for_paper():
    mask_f = tab['Environment'] == 'Fornax'
    marker_f = '^'
    color_f = CB_color_cycle[1]
    label_f = 'Fornax'
    mask_v = tab['Environment'] == 'Virgo'
    marker_v = 'v'
    color_v = CB_color_cycle[2]
    label_v = 'Virgo'
    mask_l = tab['Environment'] == 'Local Volume'
    marker_l = '>'
    color_l = CB_color_cycle[5]
    label_l = 'Local Volume'

    props = ['M_NSC',  'M_GCS', 'M_NSC/M_gal', 'M_gal', 'M_NSC/M_GCS', 'M_GCS/M_gal',
             'M_GC_lim', 'M_GC_diss', 'M_GC_lim/M_GC_diss',
             'eta', 'M_cl_min', 'M_cl_max', ]
    labels = [r'log($M_{\rm{NSC}}/M_\odot$)', r'log($M_{\rm{GCS}}/M_\odot$)',
              r'log($M_{\rm{NSC}}/M_{\rm{gal}}$)', r'log($M_{\rm{gal}}/M_\odot$)', r'log($M_{\rm{NSC}}/M_{\rm{GCS}}$)',
              r'log($M_{\rm{GCS}}/M_{\rm{gal}}$)',
              r'log($M_{\rm{GC, lim}}/M_\odot$)', r'log($M_{\rm{diss}}/M_\odot$)', r'log($M_{\rm{GC, lim}}/M_{\rm{diss}}$)',
              r'log($\eta$)', r'log($M_{\rm{cl, min}}/M_\odot$)', r'log($M_{\rm{cl, max}}/M_\odot$)']
    logs = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]).astype('bool')

    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=[6.5, 6], sharey=True)
    fig2, axes2 = plt.subplots(ncols=2, nrows=3, figsize=[6.5, 6], sharey=True)
    k = 0
    l = 0
    for i in range(3):
        axes[i, 0].set_ylabel(r'$f_{\rm{in-situ}}$', fontsize=12)
        for j in range(2):
            ax = axes[i, j]
            plot(tab[mask_f], ax, s1=props[k], s2='f_in',
                 label=label_f, marker=marker_f, color=color_f, logi=logs[k])
            plot(tab[mask_v], ax, s1=props[k], s2='f_in',
                 label=label_v, marker=marker_v, color=color_v, logi=logs[k])
            plot(tab[mask_l], ax, s1=props[k], s2='f_in',
                 label=label_l, marker=marker_l, color=color_l, logi=logs[k])
            # ax.set_xlabel(props[k])
            ax.set_xlabel(labels[k], fontsize=12)

            k += 1

    for i in range(3):
        axes2[i, 0].set_ylabel(r'$f_{\rm{in-situ}}$', fontsize=12)
        for j in range(2):
            ax = axes2[i, j]
            plot(tab[mask_f], ax, s1=props[k], s2='f_in',
                 label=label_f, marker=marker_f, color=color_f, logi=logs[k])
            plot(tab[mask_v], ax, s1=props[k], s2='f_in',
                 label=label_v, marker=marker_v, color=color_v, logi=logs[k])
            plot(tab[mask_l], ax, s1=props[k], s2='f_in',
                 label=label_l, marker=marker_l, color=color_l, logi=logs[k])
            # ax.set_xlabel(props[k])
            ax.set_xlabel(labels[k], fontsize=12)
            if props[k] == 'M_cl_min':
                ax.set_xlim(1.33, 1.63)
            k += 1

    fig.tight_layout()
    fig.savefig('Results.png', dpi=300)
    fig2.tight_layout()
    fig2.savefig('Results2.png', dpi=300)


def plot_for_paper_horizontal(tab, CB_color_cycle):
    mask_f = tab['Environment'] == 'Fornax'
    marker_f = '^'
    color_f = CB_color_cycle[1]
    label_f = 'Fornax'
    mask_v = tab['Environment'] == 'Virgo'
    marker_v = 'v'
    color_v = CB_color_cycle[2]
    label_v = 'Virgo'
    mask_l = tab['Environment'] == 'Local Volume'
    marker_l = '>'
    color_l = CB_color_cycle[5]
    label_l = 'Local Volume'

    props = ['M_NSC',   'M_NSC/M_gal', 'eta', 'M_GC_lim/M_GC_diss',
             'M_GCS', 'M_GCS/M_gal',  'M_GC_diss',  'M_cl_min',
             'M_gal', 'M_NSC/M_GCS', 'M_GC_lim',   'M_cl_max', ]

    labels = [r'log($M_{\rm{NSC}}/M_\odot$)', r'log($M_{\rm{NSC}}/M_{\rm{gal}}$)', r'log($\eta$)', r'log($M_{\rm{GC, lim}}/M_{\rm{diss}}$)',
              r'log($M_{\rm{GCS}}/M_\odot$)',  r'log($M_{\rm{GCS}}/M_{\rm{gal}}$)',  r'log($M_{\rm{diss}}/M_\odot$)',  r'log($M_{\rm{cl, min}}/M_\odot$)',
              r'log($M_{\rm{gal}}/M_\odot$)', r'log($M_{\rm{NSC}}/M_{\rm{GCS}}$)',  r'log($M_{\rm{GC, lim}}/M_\odot$)', r'log($M_{\rm{cl, max}}/M_\odot$)']

    logs = np.array([1, 1, 0, 0,
                     1, 1, 0, 0,
                     1, 1, 0, 0]).astype('bool')

    fig, axes = plt.subplots(ncols=4, nrows=3, figsize=[13, 6], sharey=True)
    #fig2, axes2 = plt.subplots(ncols=2, nrows=3, figsize=[6.5, 6], sharey=True)
    k = 0
    l = 0
    for i in range(3):
        axes[i, 0].set_ylabel(r'$f_{\rm{in-situ}}$', fontsize=12)
        for j in range(4):
            ax = axes[i, j]

            plot(tab[mask_l], ax, s1=props[k], s2='f_in',
                 label=label_l, marker=marker_l, color=color_l, logi=logs[k])
            plot(tab[mask_v], ax, s1=props[k], s2='f_in',
                 label=label_v, marker=marker_v, color=color_v, logi=logs[k])
            plot(tab[mask_f], ax, s1=props[k], s2='f_in',
                 label=label_f, marker=marker_f, color=color_f, logi=logs[k])
            # ax.set_xlabel(props[k])
            ax.set_xlabel(labels[k], fontsize=12)

            k += 1
            if props[k-1] == 'M_cl_min':
                ax.set_xlim(1.33, 1.63)

    fig.tight_layout()
    fig.savefig('Results_hori.png', dpi=300)
    # fig2.tight_layout()
    #fig2.savefig('Results2.png', dpi=300)


def stackedhist(data, stackby, **kwds):
    # number of groups:
    print(data)
    groups = np.unique(data[stackby])
    print(groups)
    for i in range(len(groups)):
        mask = data[stackby] == groups[i]
        plt.hist(data[mask], stacked=True)


if __name__ == "__main__":
    plt.close('all')
    plt.rc('font', family='serif')

    # galaxy = 'FCC190'
    #prefix = '_M_GC_max_2nd'
    # collect_res_acs(prefix)
    #prefix = '_M_GC_max'
    # collect_res_acs(prefix)
    #prefix = '_M_GC_lim'
    # collect_res_acs(prefix)
    # collect_results_local(prefix)
    # a = b

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']

    colors = ['darkred', 'mediumblue', 'darkorange', 'purple', 'forestgreen']
    colors = [CB_color_cycle[5], CB_color_cycle[2], CB_color_cycle[1]]
    fig, ax = plt.subplots()
    for i in range(len(CB_color_cycle)):
        ax.scatter(i, 0, color=CB_color_cycle[i])
    # a = b
    #acs = create_big_table(prefix='_2nd_brightest')

    acs = ascii.read('./ACS_sample_results_M_GC_max.dat').to_pandas()
    acs_in = ascii.read('./Data/ACS_sample_to_fit2.dat').to_pandas()
    acs_in.rename(columns={'M_GC_lim': 'M_GC_lim_input'}, inplace=1)

    acs = acs_in.merge(acs, on='galaxy', suffixes=('', '_fit'))
    local_res = ascii.read('LocalVolume_sample_results.dat').to_pandas()
    local_gal = ascii.read('./Data/LocalVolume_sample_to_fit.dat').to_pandas()
    local = local_gal.merge(local_res, on='galaxy', suffixes=('', '_fit'))
    local['Environment'] = 'Local Volume'
    tab = local.append(acs,  sort=1)

    #plot_for_paper_horizontal(tab, CB_color_cycle)

    #a = b

    tab_rel = tab[['M_NSC',  'M_GCS', 'M_gal', 'Environment',
                   'M_GC_lim', 'M_GC_diss',
                   'eta', 'M_cl_min', 'M_cl_max', ]]
    labels = [r'log($M_{\rm{NSC}}/M_\odot$)', r'log($M_{\rm{GCS}}/M_\odot$)',
              r'log($M_{\rm{gal}}/M_\odot$)',
              r'log($M_{\rm{GC, lim}}/M_\odot$)', r'log($M_{\rm{diss}}/M_\odot$)',
              r'log($\eta$)', r'log($M_{\rm{cl, min}}/M_\odot$)', r'log($M_{\rm{cl, max}}/M_\odot$)']

    tab_rel['M_NSC'] = np.log10(tab_rel['M_NSC'])
    tab_rel['M_gal'] = np.log10(tab_rel['M_gal'])
    tab_rel['M_GCS'] = np.log10(tab_rel['M_GCS'])

    # sns.pairplot(tab_rel)

    #fig, ax = plt.subplots(constrained_layout=True, figsize=[13, 6])
    groups = np.unique(tab_rel['Environment'])
    palette = sns.color_palette(colors)
    g = sns.PairGrid(tab_rel, diag_sharey=False, despine=True, palette=palette, corner=True, hue='Environment', hue_kws={
                     "marker": [">", "v", "^"]}, hue_order=['Local Volume', 'Virgo', 'Fornax'])
    g.map_lower(sns.scatterplot, data=tab_rel, alpha=0.95, edgecolor='k')
    # g.__init__(tab_rel, diag_sharey=False, palette=palette, corner=True, hue='Environment')
    g.map_diag(sns.distplot, kde=False, hist=True, hist_kws={'stacked': True})
    # g.map_upper(hide_current_axis)
    axes = g.axes
    g.fig.set_size_inches(13, 11)
    for i in range(len(axes[:, 0])):
        axes[-1, i].set_xlabel(labels[i], fontsize=11)
        axes[i, 0].set_ylabel(labels[i], fontsize=11)
    # sns.despine(bottom=True)
    g.add_legend(bbox_to_anchor=(0.7, 0.8))
    plt.tight_layout()

    g.savefig('Full_comparison.png', dpi=300)
    cols = tab_rel.columns

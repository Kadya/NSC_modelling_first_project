from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt  # plotting
from astropy.io import ascii, fits
import os
from astropy.table import Table

from astropy.table import Column
from collections import OrderedDict
from scipy.optimize import minimize, curve_fit
import emcee
import corner
import numpy.polynomial.polynomial as poly
from scipy import special
import seaborn as sns
import glob

# fitting R and M (project 2) R >= 4.4e4 f_in/t_form * M_NSC


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


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)


def log(x):
    return np.log10(x)


def stackedhist(data, stackby, **kwds):
    # number of groups:
    print(data)
    groups = np.unique(data[stackby])
    print(groups)
    for i in range(len(groups)):
        mask = data[stackby] == groups[i]
        plt.hist(data[mask], stacked=True)


def build_giant_table(files, prefixes, input_file='./Data/ACS_sample_to_fit2.dat'):
    acs_in = ascii.read(input_file).to_pandas()
    acs_in.rename(columns={'M_GC_lim': 'M_GC_lim_input'}, inplace=1)
    for i in range(len(files)):
        acs = ascii.read(files[i]).to_pandas()
        acs = acs_in.merge(acs, on='galaxy', suffixes=('', '_fit'))
        acs['M_NSC/M_GCS'] = np.log10(acs['M_NSC'])-np.log10(acs['M_GCS'])
        acs['prefix'] = prefixes[i]
        if i == 0:
            acs_out = acs
        else:
            acs_out = acs_out.append(acs)
    return acs_out


def fsigmoid(x, a, b, c):
    return 1.0 / (1.0 + np.exp(-a*(x-b))) + c


def func(z, a, b):
    return a*special.erf(z)+b


def plot_diff_setups(tab):
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']

    colors = [CB_color_cycle[5], CB_color_cycle[2], CB_color_cycle[1], CB_color_cycle[0]]
    labels = [r'log($M_{\rm{GC, lim}}/M_\odot$)', r'log($M_{\rm{diss}}/M_\odot$)',
              r'log($\eta$)', r'log($M_{\rm{cl, min}}/M_\odot$)', r'log($M_{\rm{cl, max}}/M_\odot$)',
              r'log($M_{\rm{NSC}}/M_{\rm{GCS}}$)', r'$f_{\rm{in-situ}}$', ]

    #fig, ax = plt.subplots(constrained_layout=True, figsize=[13, 6])
    groups = np.unique(tab_rel['Environment'])
    palette = sns.color_palette(colors)
    g = sns.PairGrid(tab_rel, diag_sharey=False, despine=True, palette=palette, corner=True, hue='prefix', hue_kws={
                     "marker": ["^", ">", "v", "<", 'o', 'p', 'P']})
    g.map_lower(sns.scatterplot, data=tab_rel, alpha=0.7, edgecolor='None', marker='^')
    #g.map_lower(sns.scatterplot, data=tab_rel_clmax, alpha=0.5, edgecolor='k', color='orange', marker='>')
    # g.__init__(tab_rel, diag_sharey=False, palette=palette, corner=True, hue='Environment')
    g.map_diag(sns.distplot, kde=False, hist=True, hist_kws={'stacked': True})
    # g.map_upper(hide_current_axis)
    axes = g.axes
    g.fig.set_size_inches(13, 11)
    for i in range(len(axes[:, 0])):
        axes[-1, i].set_xlabel(labels[i], fontsize=12)
        axes[i, 0].set_ylabel(labels[i], fontsize=12)
    # sns.despine(bottom=True)
    g.add_legend(bbox_to_anchor=(0.7, 0.8))
    plt.tight_layout()
    #plt.savefig('Different_setups_full.png', dpi=300)

    #colors = ['darkred', 'mediumblue', 'darkorange', 'cyan', 'purple', 'forestgreen', 'cyan']

    fig, ax = plt.subplots(figsize=[6.5, 4], constrained_layout=True)
    markers = ['^', '>', 'v', '<']
    markers = ['o', '^', 'd', 'P']
    label_prefixes = [r'$M_{\rm{GC,\,lim}}$ = $M_{\rm{GC,\,max}}$', r'$M_{\rm{GC,\,lim}}$ = $M_{\rm{GC, 2nd\,max}}$',
                      r'$M_{\rm{GC,\,lim}}$ from model', r'$M_{\rm{cl,\,max}} \in$ ($M_{\rm{GC,\,max}}$, $10^{8.5} M_\odot$)']
    for i, prefix in enumerate(['M_GC_max', 'M_GC_max_2nd', 'M_GC_lim', 'M_GC_max_cl_max']):
        mask = tab_rel['prefix'] == prefix
        ax.scatter(tab_rel['M_NSC/M_GCS'][mask], tab_rel['f_in'][mask],
                   marker=markers[i], color=colors[i], label=label_prefixes[i], alpha=0.6)
    ax.set_xlabel(r'log($M_{\rm{NSC}}/M_{\rm{GCS}}$)', fontsize=12)
    ax.set_ylabel(r'$f_{\rm{in-situ}}$', fontsize=12)
    ax.legend(fontsize=10)
    #plt.savefig('Different_setups.png', dpi=300)


if __name__ == "__main__":
    plt.close('all')
    plt.rc('font', family='serif')
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']

    colors = [CB_color_cycle[5], CB_color_cycle[2], CB_color_cycle[1], CB_color_cycle[0]]

    # a = b
    #acs = create_big_table(prefix='_2nd_brightest')

    files = np.sort(np.array(glob.glob('./Results/ACS_sample_results*.dat')))

    prefixes = []
    for i in range(len(files)):
        file = files[i]
        prefixes.append(file.split('.dat')[0].split('_results_')[-1])

    giant_tab = build_giant_table(files, prefixes)
    #tab = Table.from_pandas(giant_tab)
    #tab.write('ACS_sample_results.dat', format='ascii')

    #a = b
    keys = ['Environment', 'prefix',
            'M_GC_lim', 'M_GC_diss',
            'eta', 'M_cl_min', 'M_cl_max',
            'M_NSC/M_GCS', 'f_in', ]

    tab_rel = giant_tab[keys]

    fig, ax = plt.subplots()
    i = 0
    mask = tab_rel['prefix'] == 'M_GC_max_2nd'

    xdata, ydata = tab_rel['M_NSC/M_GCS'][mask], tab_rel['f_in'][mask]
    ax.scatter(xdata, ydata,
               marker='^', color='purple', label='', alpha=0.6)

    # , bounds=([-2., 2.], [0.01, 1200.]))
    popt, pcov = curve_fit(fsigmoid, xdata, ydata, method='dogbox')
    xrange = np.linspace(-1.5, 1.5, 1000)
    yrange = np.linspace(0, 1, 1000)
    ax.plot(xrange, fsigmoid(xrange, *popt), c='k')
    # , bounds=([-2., 2.], [0.01, 1200.]))
    popt, pcov = curve_fit(func, xdata, ydata, method='dogbox')
    xrange = np.linspace(-1.5, 1.5, 1000)
    ax.plot(xrange, func(xrange, *popt), c='k', ls='--')

    coefs = poly.polyfit(ydata, xdata, 3)
    ffit = poly.Polynomial(coefs)
    ax.plot(ffit(yrange), yrange)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.1, 1.1)

    fig, ax = plt.subplots(figsize=[6.5, 4], constrained_layout=True)
    markers = ['^', '>', 'v', '<']

    markers = ['o', '^', 'd', 'P']
    label_prefixes = [r'$M_{\rm{GC,\,lim}}$ = $M_{\rm{GC,\,max}}$', r'$M_{\rm{GC,\,lim}}$ = $M_{\rm{GC, 2nd\,max}}$',
                      r'$M_{\rm{GC,\,lim}}$ from model', r'$M_{\rm{cl,\,max}} \in$ ($M_{\rm{GC,\,max}}$, $10^{8.5} M_\odot$)']
    for i, prefix in enumerate(['M_GC_max', 'M_GC_max_2nd', 'M_GC_lim', 'M_GC_max_cl_max']):
        mask = tab_rel['prefix'] == prefix
        xdata, ydata = tab_rel['M_NSC/M_GCS'][mask], tab_rel['f_in'][mask]
        ax.scatter(xdata, ydata, color=colors[i], label=label_prefixes[i],
                   marker=markers[i], alpha=0.6, zorder=1)
        coefs = poly.polyfit(ydata, xdata, 3)
        ffit = poly.Polynomial(coefs)
        ax.plot(ffit(yrange), yrange, color=colors[i], zorder=2, lw=3)

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(0.1, 1.01)
    ax.set_xlabel(r'log($M_{\rm{NSC}}/M_{\rm{GCS}}$)', fontsize=12)
    ax.set_ylabel(r'$f_{\rm{in-situ}}$', fontsize=12)
    ax.legend(fontsize=10)
    #plt.savefig('Different_setups_with_fits.png', dpi=300)

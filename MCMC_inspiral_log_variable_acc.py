import numpy as np
import matplotlib.pyplot as plt  # plotting
from astropy.io import ascii, fits
import os
from astropy.table import Table
from collections import OrderedDict
from scipy.optimize import minimize
import emcee
import time
from multiprocessing import Pool
import corner

# fitting R and M (project 2) R >= 4.4e4 f_in/t_form * M_NSC


def calc_vals(theta):
    eta, f_in, f_acc, M_gal, M_GC_lim, M_GC_min, M_GC_max, M_GC_diss = theta.T
    M_gal_lin, M_GC_lim_lin, M_GC_min_lin, M_GC_max_lin, M_GC_diss_lin = np.power(
        10, theta[:, 3:].T)
    eta = np.power(10, theta[:, 0])
    M_NSC_acc_m = eta*M_gal_lin*(1-f_acc) * ((1+np.log(M_GC_max_lin/M_GC_lim_lin)) /
                                             (1+np.log(M_GC_max_lin/M_GC_min_lin)))
    M_GCS_in = eta*M_gal_lin*(1-f_acc) - M_NSC_acc_m - M_GC_diss_lin * \
        (1 + np.log(M_GC_diss_lin/M_GC_min_lin))
    M_GCS_m = M_GCS_in/(1-f_acc)

    M_NSC_m = M_NSC_acc_m/(1 - f_in)
    return np.log10(M_NSC_m), np.log10(M_GCS_m)


def calc_vals_simple(theta):
    eta, f_in, f_acc, M_gal, M_GC_lim, M_GC_min, M_GC_max, M_GC_diss = theta
    M_gal_lin, M_GC_lim_lin, M_GC_min_lin, M_GC_max_lin, M_GC_diss_lin = np.power(
        10, theta[3:])
    eta = np.power(10, theta[0])
    M_NSC_acc_m = eta*M_gal_lin*(1-f_acc) * ((1+np.log(M_GC_max_lin/M_GC_lim_lin)) /
                                             (1+np.log(M_GC_max_lin/M_GC_min_lin)))
    M_GCS_in = eta*M_gal_lin*(1-f_acc) - M_NSC_acc_m - M_GC_diss_lin * \
        (1 + np.log(M_GC_diss_lin/M_GC_min_lin))
    M_GCS_m = M_GCS_in/(1-f_acc)
    M_NSC_m = M_NSC_acc_m/(1 - f_in)
    return np.log10(M_NSC_m), np.log10(M_GCS_m)


def log_likelihood(theta, M_NSC, e_M_NSC, M_GCS, e_M_GCS):
    eta, f_in, f_acc, M_gal, M_GC_lim, M_GC_min, M_GC_max, M_GC_diss = theta  # log units

    # the main equations
    M_gal_lin, M_GC_lim_lin, M_GC_min_lin, M_GC_max_lin, M_GC_diss_lin = np.power(
        10, theta[3:])  # to linear units for the calculation
    eta = np.power(10, theta[0])

    # the main equations

    M_NSC_acc_m = eta*M_gal_lin*(1-f_acc) * ((1+np.log(M_GC_max_lin/M_GC_lim_lin)) /
                                             (1+np.log(M_GC_max_lin/M_GC_min_lin)))
    # M_NSC = M_acc + M_insitu,
    # M_insitu = f_in*M_NSC #fraction of total
    # M_NSC = Macc + f_in * M_NSC => M_acc = (1 - f_in)*M_NSC
    M_NSC_m = M_NSC_acc_m/(1 - f_in)
    M_GCS_in_m = (eta*M_gal_lin*(1-f_acc) - M_NSC_acc_m - M_GC_diss_lin *
                  (1 + np.log(M_GC_diss_lin/M_GC_min_lin)))
    M_GCS_m = (1 - f_acc)*M_GCS_in_m
    # sigmas
    M_NSC_m = np.log10(M_NSC_m)  # back to log
    M_GCS_m = np.log10(M_GCS_m)

    sigma_squared_M_NSC = e_M_NSC**2  # additional terms??
    sigma_squared_M_GCS = e_M_GCS**2

    log_P_M_NSC = np.log(1/np.sqrt(2*np.pi*sigma_squared_M_NSC)) - (0.5 *
                                                                    (M_NSC - M_NSC_m)**2/(sigma_squared_M_NSC))
    log_P_M_GCS = np.log(1/np.sqrt(2*np.pi*sigma_squared_M_GCS)) - (0.5 *
                                                                    (M_GCS - M_GCS_m)**2/(sigma_squared_M_GCS))

    result_log = log_P_M_NSC + log_P_M_GCS
    if np.isnan(result_log) or not np.isfinite(result_log):
        return -np.inf
    else:
        return result_log


def convert_to_log(val, e_val):
    """
    Convert linear val + uncertainty to log values (uncertainty in dex)
    """
    log_val = np.log10(val)
    log_e_val = np.mean(
        [np.abs(np.log10(val + e_val) - log_val), np.abs(np.log10(val - e_val) - log_val)])
    return log_val, log_e_val


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


def log_prior(theta, galaxy='FCC47', file='../Data/ACSVCS_sample.dat', mass_uncertainty=0.3):
    """
    Priors on the parameters
    """
    eta, f_in, f_acc, M_gal, M_GC_lim, M_GC_min, M_GC_max, M_GC_diss = theta
    # flat priors for eta and f_in and t_form
    tab = ascii.read(file)
    gal = tab[tab['galaxy'] == galaxy]
    result = 0

    #f_in_lower_limit, f_in_upper_lim = calc_f_in_lims(theta)

    # if not (f_in_lower_limit <= f_in <= f_in_upper_lim):
    if not (0 <= f_in <= 1):
        result = -np.inf
    elif not (0.00 < f_acc <= 1):
        result = -np.inf
    elif not (-5 < eta < np.log10(0.5)):
        result = -np.inf
    elif not (1 < M_GC_min < 2):
        result = -np.inf

    # most massive ever formed, Norris et al 2015 give limit of ~ 5x10^7 (default is 7.8)
    elif not (np.log10(gal['M_GC_max']) < M_GC_max < 7.8):
        result = -np.inf
    # elif not (5 < M_GC_lim < np.log10(gal['M_GC_max'])):  # flat prior on mass
    #    result = -np.inf
    else:
        # gaussian priors for the others

        # 0.3 dex as reasonable uncertainty on the galaxy mass
        prior_M_gal = simple_gauss(M_gal, np.log10(gal['M_gal']), mass_uncertainty)
        # MGC lim is the most massive surviving today
        # prior_M_GC_lim = simple_gauss(
        #    M_GC_lim, *convert_to_log(gal['M_GC_max'], gal['e_M_GC_max']))  # max GC today
        prior_M_GC_lim = simple_gauss(
            M_GC_lim, np.log10(gal['M_GC_max']), mass_uncertainty)  # max GC today

        prior_M_GC_diss = simple_gauss(
            M_GC_diss, np.log10(gal['M_GC_min']), mass_uncertainty)  # diss: min GC today
        result = prior_M_GC_diss + prior_M_gal + prior_M_GC_lim
    return result


def simple_gauss(val, mu, sig):
    """
    Gauss function in log
    """
    return np.log(1.0/np.sqrt(2 * np.pi * sig**2))-0.5*(val-mu)**2/sig**2


def log_probability(theta,  M_NSC, e_M_NSC, M_GCS, e_M_GCS, galaxy='FCC47', file='../Data/ACSFCS_sample.dat', mass_uncertainty=0.3):
    """
    probablity in log
    """
    lp = log_prior(theta, galaxy=galaxy, file=file, mass_uncertainty=mass_uncertainty)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, M_NSC, e_M_NSC, M_GCS, e_M_GCS)


def do_the_modelling(M_NSC, e_M_NSC, M_GCS, e_M_GCS, eta_true=0.05, f_in_true=0.8, galaxy='FCC47',
                     file='../Data/ACSFCS_sample.dat', mass_uncertainty=0.3, prefix='', steps=1000, parallel=False, cores=6):
    tab = ascii.read(file)

    gal = tab[tab['galaxy'] == galaxy]

    nll = lambda *args: -log_likelihood(*args)
    # initial = np.array([eta_true, f_in_true, np.log10(gal['M_gal'][0]), np.log10(gal['M_GC_max'][0]),
    #                    np.log10(gal['M_GC_min'][0]), np.log10(5e7), np.log10(gal['M_GC_diss'][0])])  # * np.random.randn(1)
    if galaxy == 'FCC47':
        initial = np.array([-1, 0.9, 0.5, 10.2, 6.7, 1.5, 7.7, 4.7]
                           )  # initial guess for the parameters
    if galaxy == 'FCC177':
        initial = np.array([-1, 0.57, 0.5, 9.6, 6.4, 1.5, 7.7, 4.7])
    if galaxy == 'FCC170':
        initial = np.array([-1, 0.8, 0.5, 9.8, 6.8, 1.5, 7.5, 4.7])
    else:
        initial = np.array([-1, 0.5, 0.5, np.log10(gal['M_gal'][0]), 6.5, 1.5, 7.5, 4.7])
    # print(initial)
    prob_initial = log_prior(initial, galaxy=galaxy, file=file)
    # print(prob_initial)
    if ~np.isfinite(prob_initial):
        print('Initial values outside priors!')
        return 0
    # print(initial)
    # soln = minimize(nll, initial, args=(M_NSC, e_M_NSC, M_GCS, e_M_GCS))

    nwalkers = 50
    ndim = len(initial)
    rand = (np.zeros_like(initial) + 0.01)

    pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)
    # nwalkers, ndim = pos.shape
    if not parallel:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=(M_NSC, e_M_NSC, M_GCS, e_M_GCS, galaxy, file, mass_uncertainty))
        sampler.run_mcmc(pos, steps)
        print('Walkers: {0}'.format(nwalkers))
        print('Steps: {0}'.format(steps))
    else:
        with Pool(processes=cores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                            args=(M_NSC, e_M_NSC, M_GCS, e_M_GCS, galaxy, file, mass_uncertainty), pool=pool)
            sampler.run_mcmc(pos, steps)
            print('Walkers: {0}'.format(nwalkers))
            print('Steps: {0}'.format(steps))

    # tau = sampler.get_autocorr_time()
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    labels = [r"$\eta$", r'$f_{\rm{in}}$', r'$f_{\rm{acc}}$', r'log($M_{\rm{gal}}$)', r'log($M_{\rm{GC, lim}}$)',
              r'log($M_{\rm{cl, min}}$)', r'log($M_{\rm{cl, max}}$)', r'log($M_{\rm{GC, diss}}$)']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    M_NSC_m, M_GCS_m = calc_vals(flat_samples)
    s = np.shape(flat_samples)
    to_plot = np.zeros([s[0], 5])
    to_plot[:, 0] = flat_samples[:, 0]
    to_plot[:, 1] = flat_samples[:, 1]
    to_plot[:, 2] = flat_samples[:, 2]
    to_plot[:, 3] = flat_samples[:, 4]
    to_plot[:, 4] = flat_samples[:, 5]
    # to_plot[:, -2] = M_NSC_m
    # to_plot[:, -1] = M_GCS_m

    labels_full = [r"log($\eta$)", r'$f_{\rm{in}}$', r'$f_{\rm{acc}}$', r'log($M_{\rm{gal}}$)', r'log($M_{\rm{GC, lim}}$)',
                   r'log($M_{\rm{cl, min}}$)', r'log($M_{\rm{cl, max}}$)', r'log($M_{\rm{GC, diss}}$)']
    fig = corner.corner(flat_samples, labels=labels_full, truths=None, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12}, label_kwargs={'fontsize': 12})
    fig.savefig('./Plots/{0}_corner{1}.png'.format(galaxy, prefix), dpi=300)

    labels = [r"log($\eta$)", r'$f_{\rm{in}}$', r'$f_{\rm{acc}}$',
              r'log($M_{\rm{cl, min}}$)', r'log($M_{\rm{cl, max}}$)']
    fig = corner.corner(to_plot, labels=labels, truths=None, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 14}, label_kwargs={'fontsize': 14}, title_fmt='.1f')
    fig.savefig('./Plots/{0}_corner{1}_for_paper.png'.format(galaxy, prefix), dpi=300)

    results = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        results.append(np.round(mcmc[1], 4))
        results.append(np.round(q[1], 4))
        results.append(np.round(q[0], 4))
    results = np.array(results)
    names = ["eta", "eta_p", 'eta_m', 'f_in', 'f_in_p', 'f_in_m', 'f_acc', 'f_acc_p', 'f_acc_m', 'M_gal', 'M_gal_p', 'M_gal_m',
             'M_GC_lim', 'M_GC_lim_p', 'M_GC_lim_m', 'M_cl_min', 'M_cl_min_p', 'M_cl_min_m',
             'M_cl_max', 'M_cl_max_p', 'M_cl_max_m', 'M_GC_diss', 'M_GC_diss_p', 'M_GC_diss_m']

    tab = Table(results, names=names)
    tab.write('./Results/{0}_results{1}.dat'.format(galaxy, prefix), format='ascii')

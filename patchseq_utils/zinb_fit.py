import scipy
import multiprocessing
import warnings
import logging
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argschema as ags
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.api import NegativeBinomial, ZeroInflatedNegativeBinomialP
from sklearn.neighbors import KernelDensity
from scipy.special import expit
from tqdm.contrib.concurrent import process_map
from collections import namedtuple


class ZinbFitParameters(ags.ArgSchema):
    data_file = ags.fields.InputFile(
        description="feather file with count data (genes as rows, cells as columns)")
    output_file = ags.fields.OutputFile(
        description="HDF5 file containing the fitting results")
    scale_factor_file = ags.fields.InputFile(
        default=None,
        allow_none=True,
        description="file with per cell scaling factors for count data")
    sample_id_file = ags.fields.InputFile(
        default=None,
        description="list of samples ids to use")
    n_cells_for_reg = ags.fields.Integer(
        default=5000,
        description='number of randomly-sampled cells to use for ZINB fitting')
    n_genes_for_reg = ags.fields.Integer(
        default=2000,
        description='number of random density-sampled genes to use for ZINB fitting')
    min_cells_expressing = ags.fields.Integer(
        default=5,
        description='minimum number of cells expressing gene for inclusion')


def fit_single_gene(y, display=False):
    X = np.ones_like(y)
    zinb_model = ZeroInflatedNegativeBinomialP(y, X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        zinb_results = zinb_model.fit(
            start_params=np.array([10, np.log(y.mean()), 1]),
            method="nm",
            maxiter=1000,
            disp=display,
        )

    mu = np.exp(zinb_results.params[1])
    phi = zinb_results.params[2]
    pi = expit(zinb_results.params[0])
    llik = zinb_results.llf

    return mu, phi, pi, llik


def fit_genes_with_zinb(counts_df, n_cells=5000, n_genes=2000, min_cells_expressing=5):

    logging.info(f"Keeping genes with expression in at least {min_cells_expressing} cell{'s' if min_cells_expressing != 1 else ''}")
    genes_cell_count = (counts_df > 0).sum(axis=1)
    genes = genes_cell_count.loc[genes_cell_count >= min_cells_expressing].index.values
    counts_df = counts_df.loc[genes, :]

    logging.info("Calculating all gene means")
    genes_log_mean = np.log10(counts_df.values.mean(axis=1))

    # Downsample cells for regularization fits
    rng = np.random.default_rng()
    if n_cells < counts_df.shape[1]:
        logging.info(f"Downsampling to {n_cells} cells")
        cell_sample_reg = rng.choice(counts_df.columns.values, n_cells)
        # Ensure that genes are expressed in at least `min_cells_expressing` cells in subset of cells
        logging.info("Re-checking genes for minimum expression")
        genes_cell_count_reg = (counts_df.loc[:, cell_sample_reg] > 0).sum(axis=1)
        genes_reg = genes_cell_count_reg.loc[genes_cell_count_reg >= min_cells_expressing].index.values

        logging.info("Re-calculating all gene means for subset of cells")
        genes_log_mean_reg = np.log10(counts_df.loc[genes_reg, cell_sample_reg].values.mean(axis=1))
    else:
        cell_sample_reg = counts_df.columns.values
        genes_reg = genes
        genes_log_mean_reg = genes_log_mean

    logging.info("Determining bandwidth")
    iqr = scipy.stats.iqr(genes_log_mean_reg)
    bw = 1.06 * (len(genes_log_mean_reg) ** (-1/5)) * min(np.std(genes_log_mean_reg), iqr * 1.34)

    if n_genes < counts_df.shape[0]:
        logging.info("Density sampling genes")
        # density-sample genes

        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(genes_log_mean_reg[:, np.newaxis])
        log_dens = kde.score_samples(genes_log_mean_reg[:, np.newaxis])

        sampling_prob = 1 / np.exp(log_dens)
        sampling_prob = sampling_prob / sampling_prob.sum()
        genes_reg = rng.choice(genes_reg, size=n_genes, replace=False, p=sampling_prob)

        logging.info(f"Selected {len(genes_reg)} genes for regularization step")
        genes_log_mean_reg = np.log10(counts_df.loc[genes_reg, cell_sample_reg].values.mean(axis=1))


    logging.info("Fitting selected genes with ZINBs")
    results = process_map(
        fit_single_gene,
        [counts_df.loc[g, cell_sample_reg].values for g in genes_reg],
        chunksize=1,
    )

    pg_mu_vec, pg_phi_vec, pg_pi_vec, pg_llik_vec = zip(*results)
    pg_mu_vec = np.array(pg_mu_vec)
    pg_phi_vec = np.array(pg_phi_vec)
    pg_pi_vec = np.array(pg_pi_vec)
    pg_llik_vec = np.array(pg_llik_vec)

    ZinbGeneFits = namedtuple("ZinbGeneFits",
        ["mu", "phi", "pi", "llik",
        "bw", "genes", "genes_log_mean",
        "genes_reg", "genes_log_mean_reg",
        "cells"])

    return ZinbGeneFits(
        pg_mu_vec,
        pg_phi_vec,
        pg_pi_vec,
        pg_llik_vec,
        bw,
        genes,
        genes_log_mean,
        genes_reg,
        genes_log_mean_reg,
        cell_sample_reg
    )


def fit_regressions_for_zinb_regularization(zinb_gene_fits, n_kernel_points=2000):
    bw = zinb_gene_fits.bw

    log_mu = np.log10(zinb_gene_fits.mu)
    spaced_values_log_mu = np.linspace(log_mu.min(), log_mu.max(), num=n_kernel_points)

    logging.info("Fitting phi kernel regression")
    dispersion = np.log10(1 + zinb_gene_fits.mu * zinb_gene_fits.phi)
    kr_disp = KernelReg(
        dispersion,
        log_mu,
        var_type="c",
        bw=[bw]
    )
    fit_disp, _ = kr_disp.fit(spaced_values_log_mu)

    # Ensure positive minimum value for dispersion
    min_disp_allowed = np.percentile(dispersion, 5)
    fit_disp[fit_disp < min_disp_allowed] = min_disp_allowed

    fit_phi = (10 ** fit_disp - 1) / (10 ** spaced_values_log_mu)

    logging.info("Fitting pi sigmoid function")

    # Rule of thumb to estimate "useful" section of pi estimates to fit, since
    # pi estimation gets unstable for genes with low mean expression

    # Take the smoothed average of pi and use the maximum as the lower bound,
    # since we'd expect it to monotonically decrease

    sorter = np.argsort(log_mu)
    window = 200
    smoothed_pi = np.convolve(zinb_gene_fits.pi[sorter], np.ones(window), mode='same') / window
    max_pi_inds = scipy.signal.argrelmax(smoothed_pi, order=window)[0]
    min_log_mu_for_pi_fit = log_mu[sorter][max_pi_inds[-1]]
    logging.info(f"Starting pi fit at log(mu) = {min_log_mu_for_pi_fit}")

    sig_fit = scipy.optimize.curve_fit(
        logistic_func,
        log_mu[log_mu > min_log_mu_for_pi_fit],
        zinb_gene_fits.pi[log_mu > min_log_mu_for_pi_fit],
        bounds=([0, -np.inf], [np.inf, np.inf]),
    )

    ZinbRegularizedRegressions = namedtuple("ZinbRegularizedRegressions",
        ["spaced_log_mu", "phi_kr_fit", "pi_sig_fit", "pi_sig_params", "pi_fit_log_mu_start"]
    )

    return ZinbRegularizedRegressions(
        spaced_values_log_mu,
        fit_phi,
        logistic_func(spaced_values_log_mu, *sig_fit[0]),
        sig_fit[0],
        min_log_mu_for_pi_fit,
    )


def save_results_to_h5(filename, zinb_results, reg_results):
    with h5py.File(filename, "w") as f:
        zinb_fits_group = f.create_group("zinb_fits")
        for i, n in enumerate(zinb_results._fields):
            data = zinb_results[i]
            if isinstance(data, np.ndarray):
                if n in ['genes', 'genes_reg', 'cells']:
                    data = data.astype('S')
                zinb_fits_group.create_dataset(n, data=data)
            else:
                zinb_fits_group.create_dataset(n, data=np.array(data))

        regularizations_group = f.create_group("regularizations")
        for i, n in enumerate(reg_results._fields):
            data = reg_results[i]
            if isinstance(data, np.ndarray):
                regularizations_group.create_dataset(n, data=data)
            else:
                regularizations_group.create_dataset(n, data=np.array(data))


def logistic_func(x, a, x0):
    return 1 / (1 + np.exp(a * (x - x0)))


def main():
    module = ags.ArgSchemaParser(schema_type=ZinbFitParameters)

    # Load the data
    data = pd.read_feather(module.args["data_file"]).set_index("sample_id").T

    if module.args["scale_factor_file"] is not None:
        logging.info("Scaling data")
        scaling_factor = np.loadtxt(module.args["scale_factor_file"])
        data = data * scaling_factor

    # Subset the data if requested
    if module.args["sample_id_file"] is not None:
        sample_ids = np.genfromtxt(module.args["sample_id_file"], dtype=str)
        data = data.loc[:, data.columns.intersection(sample_ids)]
    logging.info(f"Using data file with {data.shape[1]} cells and {data.shape[0]} genes")


    zinb_results = fit_genes_with_zinb(
        data,
        n_cells=module.args["n_cells_for_reg"],
        n_genes=module.args["n_genes_for_reg"],
        min_cells_expressing=module.args["min_cells_expressing"],
    )

    reg_results = fit_regressions_for_zinb_regularization(
        zinb_results)

    save_results_to_h5(module.args["output_file"], zinb_results, reg_results)


if __name__ == "__main__":
    main()
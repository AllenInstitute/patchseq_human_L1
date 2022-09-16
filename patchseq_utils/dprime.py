import numpy as np
from sklearn.model_selection import KFold
import itertools

def negative_binomial_dprime(data, type_list, type_labels,
        n_folds=5, r=3, reg_num=1e-4, reg_den=1):
    """ Calculate cross-validated d-prime values using negative binomial model

    The means of each parameter for each type are regularized as
        (reg_num + sum(expression_across_samples)) / (reg_den + n_samples)

    Parameters
    ----------
    data: array, shape (n_samples, n_features)

    type_list: list
        Types to compare

    type_labels: array, shape (n_samples)
        Types of samples in data

    n_folds: int (optional, default 5)
        Number of cross-validation folds

    r: float (optional, default 3)
        Fixed dispersion parameter for negative binomial model

    reg_num: float (optional, default 1e-4)
        Regularization for numerator of mean calculation to avoid zero means

    reg_den: float (optional, default 1)
        Regularization for denominator of mean calculation to avoid zero means

    Returns
    -------
    dict with keys (type_1, type_2) tuples
        Each element of dict has members:
            dprime: d-prime value for paired comparison
            ll_ratio1: list of log-likelihood ratios for cells in type_1
            ll_ratio2: list of log-likelihood ratios for cells in type_2
    """

    return cv_dprime(data, type_list, type_labels,
        _regularized_nb_prob, _negative_binomial_ll_ratios,
        n_folds=n_folds,
        r=r, reg_num=reg_num, reg_den=reg_den)


def zinb_dprime(data, type_list, type_labels,
        n_folds=5, r=3, a=8.0, mu0=50.0, reg_num=1e-4, reg_den=1):
    """ Calculate cross-validated d-prime values using negative binomial model

    The means of each parameter for each type are regularized as
        (reg_num + sum(expression_across_samples)) / (reg_den + n_samples)

    Parameters
    ----------
    data: array, shape (n_samples, n_features)

    type_list: list
        Types to compare

    type_labels: array, shape (n_samples)
        Types of samples in data

    n_folds: int (optional, default 5)
        Number of cross-validation folds

    r: float (optional, default 3)
        Fixed dispersion parameter for negative binomial model

    a: float (optional, default 8)
        Fixed shape parameter for logistic function of inflation parameter

    m0: float (optional, default 50)
        Fixed midpoint parameter for logistic function of inflation parameter

    reg_num: float (optional, default 1e-4)
        Regularization for numerator of mean calculation to avoid zero means

    reg_den: float (optional, default 1)
        Regularization for denominator of mean calculation to avoid zero means

    Returns
    -------
    dict with keys (type_1, type_2) tuples
        Each element of dict has members:
            dprime: d-prime value for paired comparison
            ll_ratio1: list of log-likelihood ratios for cells in type_1
            ll_ratio2: list of log-likelihood ratios for cells in type_2
    """

    return cv_dprime(data, type_list, type_labels,
        _regularized_zinb_prob, _zi_negative_binomial_ll_ratios,
        n_folds=n_folds,
        r=r, a=a, mu0=mu0, reg_num=reg_num, reg_den=reg_den)

def zinb_dprime_fit_phi(data, type_list, type_labels,
        spaced_log_mu=None, pi_lookup=None, phi_lookup=None,
        n_folds=5, reg_num=1e-4, reg_den=1):

    return cv_dprime(data, type_list, type_labels,
        params_from_reg_fit, _zi_negative_binomial_ll_ratios_var_phi,
        spaced_log_mu=spaced_log_mu, pi_lookup=pi_lookup, phi_lookup=phi_lookup,
        n_folds=n_folds, reg_num=reg_num, reg_den=reg_den)

def mv_gaussian_dprime(data, type_list, type_labels,
        n_folds=5, gaussian_type="full"):
    """ Calculate cross-validated d-prime values using multivariate Guassian model

    Parameters
    ----------
    data: array, shape (n_samples, n_features)

    type_list: list
        Types to compare

    type_labels: array, shape (n_samples)
        Types of samples in data

    n_folds: int (optional, default 5)
        Number of cross-validation folds

    gaussian_type: str (optional, default "full")
        Type of Gaussian covariance matrix to estimate ("full" or "diag")

    Returns
    -------
    dict with keys (type_1, type_2) tuples
        Each element of dict has members:
            dprime: d-prime value for paired comparison
            ll_ratio_1: list of log-likelihood ratios for cells in type_1
            ll_ratio_2: list of log-likelihood ratios for cells in type_2
    """

    if gaussian_type == "full":
        estimator = _estimate_mv_gaussian_full
    elif gaussian_type == "diag":
        estimator = _estimate_mv_gaussian_diag
    else:
        raise TypeError("gaussian_type parameter must be 'full' or 'diag'")


    return cv_dprime(data, type_list, type_labels,
        estimator, _mv_gaussian_ll_ratios,
        n_folds=n_folds)


def cv_dprime(data, type_list, type_labels, estimator, ll_ratio_func,
        n_folds, **kwargs):
    """ Calculate cross-validated d-prime values using specified functions

    Parameters
    ----------
    data: array, shape (n_samples, n_features)

    type_list: list
        Types to compare

    type_labels: array, shape (n_samples)
        Types of samples in data

    n_folds: int (optional, default 5)
        Number of cross-validation folds

    kwargs: dict
        Keyword arguments to pass to `estimator` and `ll_ratio_func`

    Returns
    -------
    dict with keys (type_1, type_2) tuples
        Each element of dict has members:
            dprime: d-prime value for paired comparison
            ll_ratio_1: list of log-likelihood ratios for cells in type_1
            ll_ratio_2: list of log-likelihood ratios for cells in type_2
    """
    results = {}
    kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)

    for counter, (t1, t2) in enumerate(itertools.combinations(type_list, 2)):
        if counter % 50 == 0:
            print(counter)

        t1_mask = type_labels == t1
        ind1 = np.arange(data.shape[0])[t1_mask]

        t2_mask = type_labels == t2
        ind2 = np.arange(data.shape[0])[t2_mask]

        # Cross-validated discriminant analysis
        ll_ratio1 = []
        ll_ratio2 = []
        for (train1, test1), (train2, test2) in zip(kf.split(ind1), kf.split(ind2)):
            train1_ind = ind1[train1]
            train2_ind = ind2[train2]
            test1_ind = ind1[test1]
            test2_ind = ind2[test2]

            est_out1 = estimator(data[train1_ind, :], **kwargs)
            est_out2 = estimator(data[train2_ind, :], **kwargs)

            x1 = data[test1_ind, :]
            x2 = data[test2_ind, :]
            
            ll_ratio1 += ll_ratio_func(x1, est_out1, est_out2, **kwargs)
            ll_ratio2 += ll_ratio_func(x2, est_out1, est_out2, **kwargs)

        # ll_ratio1 = [x for x in ll_ratio1 if x is not None]
        # ll_ratio2 = [x for x in ll_ratio2 if x is not None]
        # llr1_mean = np.mean(ll_ratio1)
        # llr1_var = np.var(ll_ratio1)
        # llr2_mean = np.mean(ll_ratio2)
        # llr2_var = np.var(ll_ratio2)
        # dprime = ((llr1_mean - llr2_mean) /
        #     np.sqrt(0.5 * (llr1_var + llr2_var)))
        
        from scipy.stats import norm
        from sklearn.metrics import roc_auc_score
        y_true = np.concatenate([np.zeros_like(ll_ratio1), np.ones_like(ll_ratio2)])
        y_score = np.concatenate([ll_ratio1, ll_ratio2])
        area = roc_auc_score(y_true, y_score, average=None, multi_class='ovo')
        n = len(y_true)
        if area==1:
            area = (n-1)/n
        if area==0:
            area = 1/n
        dprime = np.sqrt(2) * norm.ppf(area)
        
        results[(t1, t2)] = {
            "ll_ratio1": ll_ratio1,
            "ll_ratio2": ll_ratio2,
            "dprime": dprime,
        }
    return results


def _regularized_nb_prob(train_data, reg_num, reg_den, r, **kwargs):
    """Negative binomial probability based on regularized mean expression"""
    train_sum = np.reshape(train_data.sum(axis=0), (-1, 1))
    train_n = train_data.shape[1]
    mu = (reg_num + train_sum) / (reg_den + train_n)
    return mu / (r + mu)

def params_from_reg_fit(train_data, reg_num, reg_den, spaced_log_mu=None, pi_lookup=None, phi_lookup=None, **kwargs):
    train_sum = train_data.sum(axis=0)
    train_n = train_data.shape[0]

    # Regularized estimates of mean and variance
    mu_obs = (reg_num + train_sum) / (reg_den + train_n)
    # var_obs = train_data.var(axis=0) + reg_num

    x = 10**spaced_log_mu
    obs_x = (1 - pi_lookup) * x
    # x_inds = np.searchsorted(x, mu_obs)
    obs_inds = np.searchsorted(obs_x, mu_obs)
    obs_inds[obs_inds == len(x)] = len(x) - 1
    fit_mu = x[obs_inds]
    fit_phi = phi_lookup[obs_inds]
    fit_pi = pi_lookup[obs_inds]
    # pi = logistic_func(np.log10(mu), *params)
    fit_r = 1/fit_phi
    fit_p = fit_mu / (fit_r + fit_mu)
    return fit_p, fit_pi, fit_r

def _regularized_zinb_prob(train_data, reg_num, reg_den, r, a, mu0, **kwargs):
    """Zero-inflated negative binomial probability based on regularized mean expression.

    Since some genes appear to follow a non-inflated NB and others are
    zero-inflated, we compare the expected variance for both models and assign
    the one closer to the (log) sample variance.

    The inflation parameter depends on the mean expression and
    follows a logistic distribution defined by a and mu0.
    """

    # Define fixed parameters for zero-inflated negative binomial
    x = 10 ** np.linspace(-3, 5, 10000)
    y = (x ** 2) / r + x
    pi_lookup = 1 / (1 + (x / mu0) ** (a / np.log(10)))
    obs_x = (1 - pi_lookup) * x
    obs_y = (1 - pi_lookup) * x * (1 + x * (pi_lookup + 1/r))

    train_sum = train_data.sum(axis=0)
    train_n = train_data.shape[0]

    # Regularized estimates of mean and variance
    mu = (reg_num + train_sum) / (reg_den + train_n)
    s2 = train_data.var(axis=0) + reg_num

    # Determine if sample variance is closer to NB or ZINB model
    x_inds = np.searchsorted(x, mu)
    x_inds[x_inds == len(x)] = len(x) - 1
    obs_inds = np.searchsorted(obs_x, mu)
    obs_inds[obs_inds == len(obs_x)] = len(obs_x) - 1
    diff_nb = (np.log10(s2) - np.log10(y[x_inds])) ** 2
    diff_zinb = (np.log10(s2) - np.log10(obs_y[obs_inds])) ** 2
    fit_type_mask = diff_nb < diff_zinb

    # Figure out inflation parameters per gene
    fit_pi = pi_lookup[obs_inds]
    fit_pi[fit_type_mask] = 0 # Fits with regular NB have inflation parameter of zero

    fit_mu = x[obs_inds]
    fit_mu[fit_type_mask] = mu[fit_type_mask]

    fit_p = fit_mu / (r + fit_mu)

    return fit_p, fit_pi


def _negative_binomial_ll_ratios(x, p1, p2, r, **kwargs):
    """Log-likehood ratios for each sample in x using negative binomial model"""
    ll_ratios = []
    for row in range(x.shape[0]):
        xr = np.reshape(x[row, :], (-1, 1))
        ll_ratio = np.sum(xr * (np.log(p1) - np.log(p2)) + r * (np.log(1 - p1) - np.log(1 - p2)))
        ll_ratios.append(ll_ratio)
    return ll_ratios


def _zi_negative_binomial_ll_ratios(x, zinb1, zinb2, r, **kwargs):
    """Log-likehood ratios for each sample in x using two zero-inflated negative binomial models"""
    p1, pi1 = zinb1
    p2, pi2 = zinb2

    mask_for_nonzeros = x == 0
    mask_for_zeros = x > 0

    # log-likelihood of zero counts doesn't depend on x (since x is zero)
    ll_ratio_z_per_gene = (
        np.log(pi1 + (1 - pi1) * np.power(1 - p1, r)) -
        np.log(pi2 + (1 - pi2) * np.power(1 - p2, r))
    )
    # log-likelihood of nonzero counts
    ll_ratio_nz_per_gene = (
        x * (np.log(p1) - np.log(p2)) +
        r * (np.log(1 - p1) - np.log(1 - p2)) +
        (np.log(1 - pi1) - np.log(1 - pi2))
    )

    ll_ratio_total = (np.ma.array(np.broadcast_to(ll_ratio_z_per_gene, x.shape), mask=mask_for_zeros).sum(axis=1) +
        np.ma.array(ll_ratio_nz_per_gene, mask=mask_for_nonzeros).sum(axis=1))
    return ll_ratio_total.tolist()

from scipy.special import loggamma
def _zi_negative_binomial_ll_ratios_var_phi(x, zinb1, zinb2, **kwargs):
    """Log-likehood ratios for each sample in x using two zero-inflated negative binomial models"""
    p1, pi1, r1 = zinb1
    p2, pi2, r2 = zinb2

    mask_for_nonzeros = x == 0
    mask_for_zeros = x > 0

    # log-likelihood of zero counts doesn't depend on x (since x is zero)
    ll_ratio_z_per_gene = (
        np.log(pi1 + (1 - pi1) * np.power(1 - p1, r1)) -
        np.log(pi2 + (1 - pi2) * np.power(1 - p2, r2))
    )
    # log-likelihood of nonzero counts
    ll_ratio_nz_per_gene = (
        loggamma(x+r1) - loggamma(x+r2)
        -(loggamma(r1) - loggamma(r2)) +
        x * (np.log(p1) - np.log(p2)) +
        (r1*np.log(1 - p1) - r2*np.log(1 - p2)) +
        (np.log(1 - pi1) - np.log(1 - pi2))
    )

    ll_ratio_total = (np.ma.array(np.broadcast_to(ll_ratio_z_per_gene, x.shape), mask=mask_for_zeros).sum(axis=1) +
        np.ma.array(ll_ratio_nz_per_gene, mask=mask_for_nonzeros).sum(axis=1))
    return ll_ratio_total.tolist()

def _estimate_mv_gaussian_full(train_data, reg_covar=1e-6, **kwargs):
    """Gaussian multivariate parameter estimates (full covariance matrix)"""
    mu = np.reshape(train_data.mean(axis=0), (-1, 1))
    n_features = mu.shape[0]
    sigma = np.cov(train_data.T)
    sigma.flat[::n_features + 1] += reg_covar

    return (mu, sigma)


def _estimate_mv_gaussian_diag(train_data, reg_covar=1e-6, **kwargs):
    """Gaussian multivariate parameter estimates (diagonal covariance matrix)"""
    mu = np.reshape(train_data.mean(axis=0), (-1, 1))
    n_features = mu.shape[0]
    sigma = np.diag(train_data.var(axis=0))
    sigma.flat[::n_features + 1] += reg_covar

    return (mu, sigma)


def _mv_gaussian_ll_ratios(x, gauss1, gauss2, **kwargs):
    """Log-likehood ratios for each sample in x using multivariate Gaussian model"""
    mu1, sigma1 = gauss1
    mu2, sigma2 = gauss2

    ll_ratios = []

    for r in range(x.shape[0]):
        if len(sigma1.shape) < 2:
            ll1 = -0.5 * (np.log(2 * np.pi * sigma1)
                    + ((x[r, :] - mu1) ** 2) / sigma1)
            ll2 = -0.5 * (np.log(2 * np.pi * sigma2)
                    + ((x[r, :] - mu2) ** 2) / sigma2)
        else:
            xr = np.reshape(x[r, :], (-1, 1))
            ll1 = -0.5 * (np.log(np.linalg.det(sigma1))
                     + (xr - mu1).T @ np.linalg.inv(sigma1) @ (xr - mu1))
            ll2 = -0.5 * (np.log(np.linalg.det(sigma2))
                     + (xr - mu2).T @ np.linalg.inv(sigma2) @ (xr - mu2))

        ll_ratio = ll1 - ll2
        ll_ratios.append(ll_ratio[0, 0])
    return ll_ratios


"""
helpers.py  —  ST3247 shared utilities
Import this in any notebook instead of copy-pasting functions around.

Usage:
    from helpers import (calc_summaries, SUMMARY_NAMES, load_observed_summaries,
                         sample_prior_theta, generate_prior_bank,
                         abc_rejection, PRIOR_BETA, PRIOR_GAMMA, PRIOR_RHO,
                         distance_fn, sample_prior, weighted_cov, ess, smc_abc)
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

# ── Prior bounds ───────────────────────────────────────────────────────────────
PRIOR_BETA  = (0.05, 0.50)
PRIOR_GAMMA = (0.02, 0.20)
PRIOR_RHO   = (0.00, 0.80)

# ── Summary statistic names ────────────────────────────────────────────────────
SUMMARY_NAMES = [
    "peak", "t_peak", "auc", "early_mass",
    "t_5", "t_10", "t_20", "duration_10",
    "inc_peak", "t_inc_peak",
    "slope_growth", "slope_decay",
    "total_rewire", "max_rewire", "t_rewire_peak", "early_rewire",
    "lag_rewire_infect", "rewire_to_auc", "corr_I_R", "corr_dI_R",
    "mean_degree", "var_degree", "entropy_degree", "tail_mass",
]

# Best summary subset index (from forward selection in ST3247_ABC_v2)
FINAL_JOINT = [3, 15, 10, 5, 4, 6, 2, 7, 12, 0, 17, 18, 13, 8, 1, 19]

def calc_summaries(infected_fraction, rewire_counts, degree_histogram):
    """
    Compute 24 summary statistics from one simulation replicate.

    Parameters
    ----------
    infected_fraction  : array of shape (201,)
    rewire_counts      : array of shape (201,)
    degree_histogram   : array of shape (31,)

    Returns
    -------
    summaries : 1D numpy array of length 24
    """
    I   = infected_fraction
    Rw  = rewire_counts
    T   = len(I)
    eps = 1e-8

    # Basic
    peak       = np.max(I)
    t_peak     = np.argmax(I)
    auc        = np.sum(I)
    early_mass = np.sum(I[:max(1, T // 5)])

    # Threshold crossing times
    def first_crossing(arr, thresh):
        idx = np.where(arr >= thresh)[0]
        return idx[0] if len(idx) > 0 else T

    t_5         = first_crossing(I, 0.05)
    t_10        = first_crossing(I, 0.10)
    t_20        = first_crossing(I, 0.20)
    duration_10 = np.sum(I >= 0.10)

    # Incidence
    dI         = np.diff(I, prepend=I[0])
    incidence  = np.maximum(dI, 0)
    inc_peak   = np.max(incidence)
    t_inc_peak = np.argmax(incidence)

    # Growth / decay slopes
    t_growth_end = max(2, T // 5)
    x            = np.arange(t_growth_end)
    y            = np.log(I[:t_growth_end] + eps)
    slope_growth = np.polyfit(x, y, 1)[0]

    if t_peak < T - 2:
        x2          = np.arange(t_peak, T)
        y2          = np.log(I[t_peak:] + eps)
        slope_decay = np.polyfit(x2, y2, 1)[0]
    else:
        slope_decay = 0.0

    # Rewiring summaries
    total_rewire  = np.sum(Rw)
    max_rewire    = np.max(Rw)
    t_rewire_peak = np.argmax(Rw)
    early_rewire  = np.sum(Rw[:max(1, T // 5)])

    # Joint infection-rewiring
    lag_rewire_infect = t_rewire_peak - t_peak
    rewire_to_auc     = total_rewire / (auc + eps)

    corr_I_R  = np.corrcoef(I, Rw)[0, 1]        if (np.std(I) > 0 and np.std(Rw) > 0)        else 0.0
    corr_dI_R = np.corrcoef(incidence, Rw)[0, 1] if (np.std(incidence) > 0 and np.std(Rw) > 0) else 0.0

    # Degree summaries
    degrees        = np.repeat(np.arange(len(degree_histogram)), degree_histogram)
    mean_degree    = np.mean(degrees) if len(degrees) > 0 else 0.0
    var_degree     = np.var(degrees)  if len(degrees) > 0 else 0.0
    p              = degree_histogram / (np.sum(degree_histogram) + eps)
    entropy_degree = -np.sum(p * np.log(p + eps))
    tail_mass      = np.sum(degree_histogram[10:]) / (np.sum(degree_histogram) + eps)

    return np.array([
        peak, t_peak, auc, early_mass,
        t_5, t_10, t_20, duration_10,
        inc_peak, t_inc_peak,
        slope_growth, slope_decay,
        total_rewire, max_rewire, t_rewire_peak, early_rewire,
        lag_rewire_infect, rewire_to_auc, corr_I_R, corr_dI_R,
        mean_degree, var_degree, entropy_degree, tail_mass,
    ], dtype=float)


def load_observed_summaries(infected_df, rewire_df, degree_df):
    """Average summary vector across all observed replicates."""
    rep_summaries = []
    for rep_id in infected_df["replicate_id"].unique():
        inf = infected_df[infected_df["replicate_id"] == rep_id]["infected_fraction"].values
        rew = rewire_df  [rewire_df  ["replicate_id"] == rep_id]["rewire_count"].values
        deg_rows = degree_df[degree_df["replicate_id"] == rep_id]
        deg = (deg_rows.set_index("degree")["count"]
                       .reindex(range(31), fill_value=0).values)
        rep_summaries.append(calc_summaries(inf, rew, deg))
    return np.mean(rep_summaries, axis=0)


def sample_prior_theta(rng):
    """Draw one (beta, gamma, rho) from the joint uniform prior."""
    return np.array([
        rng.uniform(*PRIOR_BETA),
        rng.uniform(*PRIOR_GAMMA),
        rng.uniform(*PRIOR_RHO),
    ], dtype=float)


def generate_prior_bank(simulate_fn, n_sim, rng, verbose=True):
    """
    Draw n_sim parameter sets from the prior and simulate each one.

    Parameters
    ----------
    simulate_fn : callable  — the simulate() function from updated_simulator
    n_sim       : int       — number of simulations
    rng         : numpy Generator

    Returns
    -------
    params : (n_sim, 3)            — beta, gamma, rho draws
    sims   : (n_sim, n_summaries)  — summary vectors
    """
    params = np.empty((n_sim, 3),                  dtype=float)
    sims   = np.empty((n_sim, len(SUMMARY_NAMES)), dtype=float)

    for i in range(n_sim):
        theta          = sample_prior_theta(rng)
        params[i]      = theta
        beta, gamma, rho = theta
        inf, rew, deg  = simulate_fn(beta=beta, gamma=gamma, rho=rho, rng=rng)
        sims[i]        = calc_summaries(inf, rew, deg)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  {i+1:>6,}/{n_sim:,}", flush=True)

    return params, sims


def abc_rejection(s_obs_norm, sims_norm, params, stat_indices, acceptance_rate=0.02):
    """
    Rejection ABC: accept the closest acceptance_rate fraction of simulations.

    Returns
    -------
    params_accepted : accepted parameter draws
    sims_accepted   : their summary vectors (un-normalised)
    mask            : boolean index into the full prior bank
    threshold       : distance cutoff used
    """
    diff      = sims_norm[:, stat_indices] - s_obs_norm[stat_indices]
    distances = np.linalg.norm(diff, axis=1)
    n_accept  = max(1, int(acceptance_rate * len(params)))
    threshold = np.sort(distances)[n_accept]
    mask      = distances <= threshold
    return params[mask], mask, threshold


# ── SMC-ABC utilities ─────────────────────────────────────────────────────────
def distance_fn(s_sim, s_obs, scale=None):
    """Euclidean distance, with optional per-summary scaling."""
    s_sim = np.asarray(s_sim, dtype=float)
    s_obs = np.asarray(s_obs, dtype=float)
    diff = s_sim - s_obs

    if scale is not None:
        scale = np.asarray(scale, dtype=float)
        scale = np.where(scale == 0, 1.0, scale)
        diff = diff / scale

    return float(np.linalg.norm(diff))


def sample_prior(rng, prior_bounds=None):
    """Draw one parameter vector from independent uniform priors."""
    if prior_bounds is None:
        prior_bounds = np.array([PRIOR_BETA, PRIOR_GAMMA, PRIOR_RHO], dtype=float)

    return np.array([
        rng.uniform(low, high) for low, high in prior_bounds
    ], dtype=float)


def in_prior_support(theta, prior_bounds=None):
    """Check whether theta lies inside prior support."""
    if prior_bounds is None:
        prior_bounds = np.array([PRIOR_BETA, PRIOR_GAMMA, PRIOR_RHO], dtype=float)

    theta = np.asarray(theta, dtype=float)
    return bool(np.all((theta >= prior_bounds[:, 0]) & (theta <= prior_bounds[:, 1])))


def prior_pdf(theta, prior_bounds=None):
    """Joint prior density for independent uniforms."""
    if prior_bounds is None:
        prior_bounds = np.array([PRIOR_BETA, PRIOR_GAMMA, PRIOR_RHO], dtype=float)

    if not in_prior_support(theta, prior_bounds=prior_bounds):
        return 0.0

    widths = prior_bounds[:, 1] - prior_bounds[:, 0]
    return float(np.prod(1.0 / widths))


def perturb_theta(theta, cov, rng, prior_bounds=None):
    """Gaussian random-walk proposal with support check."""
    if prior_bounds is None:
        prior_bounds = np.array([PRIOR_BETA, PRIOR_GAMMA, PRIOR_RHO], dtype=float)

    theta = np.asarray(theta, dtype=float)
    try:
        proposal = rng.multivariate_normal(theta, cov, check_valid="ignore")
    except Exception:
        return None

    if not in_prior_support(proposal, prior_bounds=prior_bounds):
        return None

    return proposal


def weighted_cov(X, w):
    """Numerically stable weighted covariance."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    total = np.sum(w)
    if not np.isfinite(total) or total <= 0:
        w = np.full(len(w), 1.0 / len(w), dtype=float)
    else:
        w = w / total

    mean = np.sum(X * w[:, None], axis=0)
    Xm = X - mean
    denom = 1.0 - np.sum(w ** 2)
    if denom <= 1e-12:
        denom = 1.0

    cov = (Xm * w[:, None]).T @ Xm / denom
    return 0.5 * (cov + cov.T)


def ess(weights):
    """Effective sample size from normalized importance weights."""
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        return 0.0

    weights = weights / total
    return float(1.0 / np.sum(weights ** 2))


def _stabilize_cov(cov, jitter=1e-9):
    """Symmetrize and jitter covariance matrix."""
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    scale = max(1.0, float(np.trace(cov)) / cov.shape[0])
    return cov + (jitter * scale) * np.eye(cov.shape[0])


def _kernel_mixture_logpdf(theta, particles_prev, weights_prev, cov):
    """Log density of SMC proposal mixture at theta."""
    log_terms = []
    for theta_prev, weight_prev in zip(particles_prev, weights_prev):
        if weight_prev <= 0:
            continue
        try:
            log_density = multivariate_normal.logpdf(
                theta,
                mean=theta_prev,
                cov=cov,
                allow_singular=True,
            )
        except Exception:
            continue

        if np.isfinite(log_density):
            log_terms.append(np.log(weight_prev) + log_density)

    if not log_terms:
        return -np.inf

    return float(logsumexp(log_terms))


def smc_abc(
    n_particles,
    eps_schedule,
    simulate_fn,
    summary_fn,
    s_obs,
    rng,
    scale=None,
    max_tries_per_particle=10000,
    eps_quantiles=None,
    prior_bounds=None,
):
    """
    Sequential Monte Carlo ABC with optional adaptive epsilon quantiles.

    eps_schedule:
        - Full sequence [eps0, eps1, ...], or
        - [eps0] with eps_quantiles controlling later populations.
    """
    if prior_bounds is None:
        prior_bounds = np.array([PRIOR_BETA, PRIOR_GAMMA, PRIOR_RHO], dtype=float)

    eps_schedule = np.atleast_1d(np.asarray(eps_schedule, dtype=float))
    if eps_schedule.ndim != 1 or len(eps_schedule) == 0:
        raise ValueError("eps_schedule must contain at least one tolerance")

    if eps_quantiles is None:
        eps_quantiles = []
    eps_quantiles = np.asarray(eps_quantiles, dtype=float)

    fixed_mode = len(eps_schedule) > 1
    if fixed_mode and len(eps_quantiles) > 0:
        raise ValueError("Use either a full eps_schedule or eps_schedule[0] + eps_quantiles")

    if fixed_mode:
        n_pops = len(eps_schedule)
    else:
        n_pops = 1 + len(eps_quantiles)

    s_obs = np.asarray(s_obs, dtype=float)

    particles_list = []
    weights_list = []
    distances_list = []
    summaries_list = []
    acceptance_rates = []
    covariances = []
    eps_used = []

    # Population 0: rejection from prior.
    eps0 = float(eps_schedule[0])
    eps_used.append(eps0)

    pop_particles = []
    pop_distances = []
    pop_summaries = []
    total_tries = 0

    for i in range(n_particles):
        accepted = False
        tries = 0

        while not accepted:
            tries += 1
            total_tries += 1
            if tries > max_tries_per_particle:
                raise RuntimeError(
                    f"Population 0, particle {i}: exceeded max_tries_per_particle={max_tries_per_particle}"
                )

            theta = sample_prior(rng, prior_bounds=prior_bounds)
            sim_data = simulate_fn(theta, rng)
            s_sim = np.asarray(summary_fn(sim_data), dtype=float)
            dist = distance_fn(s_sim, s_obs, scale=scale)

            if dist <= eps0:
                pop_particles.append(theta)
                pop_distances.append(dist)
                pop_summaries.append(s_sim)
                accepted = True

    particles = np.asarray(pop_particles, dtype=float)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    distances = np.asarray(pop_distances, dtype=float)
    summaries = np.asarray(pop_summaries, dtype=float)

    particles_list.append(particles)
    weights_list.append(weights)
    distances_list.append(distances)
    summaries_list.append(summaries)
    acceptance_rates.append(n_particles / total_tries)
    covariances.append(None)

    # Populations t >= 1: perturb + reweight.
    for t in range(1, n_pops):
        prev_particles = particles_list[-1]
        prev_weights = weights_list[-1]
        kernel_cov = _stabilize_cov(2.0 * weighted_cov(prev_particles, prev_weights))

        if fixed_mode:
            eps_t = float(eps_schedule[t])
        else:
            q = float(eps_quantiles[t - 1])
            eps_t = float(np.quantile(distances_list[-1], q))
        eps_used.append(eps_t)

        pop_particles = []
        pop_log_weights = []
        pop_distances = []
        pop_summaries = []
        total_tries = 0

        for i in range(n_particles):
            accepted = False
            tries = 0

            while not accepted:
                tries += 1
                total_tries += 1
                if tries > max_tries_per_particle:
                    raise RuntimeError(
                        f"Population {t}, particle {i}: exceeded max_tries_per_particle={max_tries_per_particle}"
                    )

                parent_idx = rng.choice(len(prev_particles), p=prev_weights)
                theta = perturb_theta(
                    prev_particles[parent_idx],
                    kernel_cov,
                    rng,
                    prior_bounds=prior_bounds,
                )
                if theta is None:
                    continue

                sim_data = simulate_fn(theta, rng)
                s_sim = np.asarray(summary_fn(sim_data), dtype=float)
                dist = distance_fn(s_sim, s_obs, scale=scale)

                if dist <= eps_t:
                    log_prior = np.log(max(prior_pdf(theta, prior_bounds=prior_bounds), 1e-300))
                    log_denom = _kernel_mixture_logpdf(theta, prev_particles, prev_weights, kernel_cov)
                    log_weight = log_prior - log_denom

                    pop_particles.append(theta)
                    pop_log_weights.append(log_weight)
                    pop_distances.append(dist)
                    pop_summaries.append(s_sim)
                    accepted = True

        pop_particles = np.asarray(pop_particles, dtype=float)
        pop_log_weights = np.asarray(pop_log_weights, dtype=float)
        pop_distances = np.asarray(pop_distances, dtype=float)
        pop_summaries = np.asarray(pop_summaries, dtype=float)

        if np.any(~np.isfinite(pop_log_weights)):
            pop_weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
        else:
            pop_log_weights = pop_log_weights - logsumexp(pop_log_weights)
            pop_weights = np.exp(pop_log_weights)

        pop_weights = np.clip(pop_weights, 0.0, None)
        weight_total = np.sum(pop_weights)
        if not np.isfinite(weight_total) or weight_total <= 0:
            pop_weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
        else:
            pop_weights = pop_weights / weight_total

        particles_list.append(pop_particles)
        weights_list.append(pop_weights)
        distances_list.append(pop_distances)
        summaries_list.append(pop_summaries)
        acceptance_rates.append(n_particles / total_tries)
        covariances.append(kernel_cov)

    return {
        "particles": particles_list,
        "weights": weights_list,
        "distances": distances_list,
        "summaries": summaries_list,
        "acceptance_rates": acceptance_rates,
        "covariances": covariances,
        "eps_schedule": np.asarray(eps_used, dtype=float),
    }

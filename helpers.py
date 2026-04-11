"""
helpers.py  —  ST3247 shared utilities
Import this in any notebook instead of copy-pasting functions around.

Usage:
    from helpers import (calc_summaries, SUMMARY_NAMES, load_observed_summaries,
                         sample_prior_theta, generate_prior_bank,
                         abc_rejection, PRIOR_BETA, PRIOR_GAMMA, PRIOR_RHO)
"""

import numpy as np
import pandas as pd

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
FINAL_JOINT = [17, 13, 1, 9, 8, 10, 3, 4, 16]


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

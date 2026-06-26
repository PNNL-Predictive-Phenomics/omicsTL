"""Bayesian optimization-style experiment recommendation for omicsTL models.

Works with any trained TransferVAE, TransferMLP, or TransferForest and any
dataset. The interface is fully domain-agnostic. The main entry point is
`recommend_next_batch`.

Uncertainty Sources:
    RF: Variance across individual transfer trees (`pred_0` through `pred_N`).
    DL: Monte Carlo dropout. The model is temporarily set to training mode so
        dropout remains active, and `n_mc_samples` forward passes are averaged.

Acquisition Functions:
    EI: Expected Improvement. Recommended default.
    UCB: Upper Confidence Bound. Robust when uncertainty is poorly calibrated.
    POI: Probability of Improvement. Included for completeness; EI is usually
        better in practice.

Batch Diversity:
    Uses a two-stage selection procedure:
    1. Shortlist the top `shortlist_pct` fraction of candidates by acquisition
       score, with a default of the top 10%.
    2. From that shortlist, greedily pick `batch_size` points that maximize the
       minimum pairwise Euclidean distance.

    This greedy max-min strategy is a 2-approximation to the NP-hard optimum.
    It is stronger than greedy exclusion (`min_dist_pct`) because it actively
    optimizes spread within the high-EI region rather than only preventing the
    worst clustering.

Candidate Grid Resolution:
    By default, all features are sampled as continuous floats. Pass
    `step_sizes` to snap individual features to experimental grid levels. An
    integer step gives integer values, `0.5` gives one decimal place, `0.25`
    gives two, and so on. Mixed precision is supported, so each feature may
    declare its own granularity.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from scipy.stats import qmc as _qmc

from omicstl.transfer_model import TransferModel
from omicstl.simulation_utils.model_utils import predict_rf_model
from omicstl.transfer_forest import TransferForest
from omicstl.transfer_networks import TransferMLP, TransferVAE

logger = logging.getLogger(__name__)

AcquisitionName = Literal["EI", "UCB", "POI"]


def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 that is >= n (required by Sobol sampler)."""
    return 1 << max(n - 1, 0).bit_length()


def _maximin_select(
    norm_candidates: np.ndarray,
    acq_scores: np.ndarray,
    batch_size: int,
    shortlist_pct: float = 0.10,
) -> np.ndarray:
    """Select a batch using a top-EI shortlist and greedy max-min diversity.

    Args:
        norm_candidates: Feature matrix normalized to [0, 1] with shape (n, d).
        acq_scores: Acquisition score for each candidate with shape (n,).
        batch_size: Number of points to select.
        shortlist_pct: Fraction of candidates to keep before applying max-min
            selection. Default is 0.10 (top 10%). The shortlist size is always at
            least `batch_size`.

    Returns:
        Indices into `norm_candidates` and `acq_scores` with shape
        `(batch_size,)`, ordered by selection. The first selected point is the
        highest-EI seed.

    Notes:
        Algorithm:
            1. Shortlist k = max(batch_size, ceil(n * shortlist_pct)) candidates by
            EI.
            2. Precompute all pairwise Euclidean distances within the shortlist.
            3. Seed with the highest-EI shortlisted point.
            4. Greedily add the point whose minimum distance to the selected set is
            largest. Update minimum distances after each pick in O(k) time.

        Total complexity is O(n log k) for shortlisting, O(k^2 d) for distance
        precomputation, and O(batch_size * k) for the greedy loop. This is fast
        for typical values such as k ~= 800 and d <= 20.
    """
    n = len(acq_scores)
    k = min(n, max(batch_size, int(np.ceil(n * shortlist_pct))))

    shortlist_idx = (
        np.arange(n) if k >= n
        else np.argpartition(acq_scores, -k)[-k:]
    )
    shortlist  = norm_candidates[shortlist_idx]   # (k, d)
    sh_scores  = acq_scores[shortlist_idx]        # (k,)

    logger.debug(
        "max-min shortlist: top %.0f%% → k=%d from n=%d candidates.",
        shortlist_pct * 100, k, n,
    )

    # Precompute all pairwise distances in shortlist
    diff     = shortlist[:, None, :] - shortlist[None, :, :]  # (k, k, d)
    pairwise = np.sqrt((diff ** 2).sum(axis=2))               # (k, k)

    # Greedy max-min selection
    first    = int(np.argmax(sh_scores))
    selected = [first]
    min_dist = pairwise[:, first].copy() # distance to the only selected point
    min_dist[first] = -np.inf # mark as selected

    for _ in range(batch_size - 1):
        nxt = int(np.argmax(min_dist)) # point farthest from selected set
        selected.append(nxt)
        min_dist = np.minimum(min_dist, pairwise[:, nxt])
        min_dist[nxt] = -np.inf

    return shortlist_idx[np.array(selected)]


def _infer_decimals(step: float) -> int:
    """Return the number of decimal places implied by `step`.

    Examples:
        1 -> 0
        25 -> 0
        0.5 -> 1
        0.25 -> 2
        0.1 -> 1
        0.05 -> 2

    Uses the string representation of `step` to avoid floating-point
    ambiguity.
    """
    s = f"{step:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def _make_allowed_values(lo: float, hi: float, step: float) -> np.ndarray:
    """Enumerate all discrete allowed values in [lo, hi] at `step` increments.

    Args:
        lo: Inclusive lower bound.
        hi: Inclusive upper bound.
        step: Grid spacing, for example 1 for integers or 0.5 for half-steps.

    Returns:
        A 1-D array of values rounded to the precision implied by `step`.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step!r}")
    n_steps = max(1, round((hi - lo) / step))
    vals = lo + np.arange(n_steps + 1) * step
    vals = vals[vals <= hi + step * 1e-6]   # trim float overshoot
    return np.round(vals, _infer_decimals(step))


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.0,
) -> np.ndarray:
    """Compute Expected Improvement over the current best observation `f_best`.

    EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z), where
    Z = (mu - f_best - xi) / sigma.

    The first term is exploitation, rewarding predictions above `f_best`. The
    second is exploration, rewarding uncertainty. No hyperparameter tuning is
    usually needed because the balance self-regulates as more data are
    collected.

    Args:
        mu: Predicted mean for each candidate.
        sigma: Predicted standard deviation for each candidate.
        f_best: Best response value observed so far.
        xi: Small positive jitter to encourage exploration. Default is 0.

    Returns:
        The EI score for each candidate.
    """
    sigma = np.clip(sigma, 1e-9, None)
    Z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return np.clip(ei, 0.0, None)


def upper_confidence_bound(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappa: float | None = None,
    n_obs: int | None = None,
) -> np.ndarray:
    """Compute the Upper Confidence Bound, `mu + kappa * sigma`.

    `kappa` controls the explore-exploit tradeoff. Higher values increase
    exploration. UCB is often more robust than EI when uncertainty estimates are
    poorly calibrated, but it requires a sensible choice of `kappa`.

    If `kappa` is None, the default schedule is
    `kappa = sqrt(2 * ln(n_obs))`, where `n_obs` is the number of existing
    observations. This increases with dataset size so exploration is maintained
    as more rounds are collected. Passing a fixed float overrides this behavior.

    Args:
        mu: Predicted mean for each candidate.
        sigma: Predicted standard deviation for each candidate.
        kappa: Exploration weight. If None, a default schedule based on `n_obs`
            is used.
        n_obs: Number of existing observations. Required when `kappa` is None.

    Returns:
        The UCB score for each candidate.
    """
    if kappa is None:
        if n_obs is None or n_obs < 1:
            raise ValueError(
                "UCB requires either a kappa value or n_obs (number of existing "
                "observations) to auto-compute kappa = sqrt(2 * ln(n_obs))."
            )
        kappa = float(np.sqrt(2.0 * np.log(max(n_obs, 2))))
        logger.debug("UCB: auto-computed kappa = %.4f (n_obs=%d)", kappa, n_obs)
    return mu + kappa * np.clip(sigma, 0.0, None)


def probability_of_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute Probability of Improvement over `f_best + xi`.

    POI(x) = Phi((mu - f_best - xi) / sigma).

    `xi` is a required improvement buffer. With `xi = 0`, POI degenerates because
    any point infinitesimally above `f_best` gets probability near 1 regardless of
    magnitude, so the search clusters around the current best and stops exploring.
    The default `xi = 0.01` forces the method to prefer points that are
    meaningfully better.

    EI is generally preferred because it accounts for improvement magnitude
    automatically, but POI can be useful when you want a simple probability
    threshold.

    Args:
        mu: Predicted mean for each candidate.
        sigma: Predicted standard deviation for each candidate.
        f_best: Best response value observed so far.
        xi: Improvement buffer. Default is 0.01.

    Returns:
        The POI score for each candidate.
    """
    if xi <= 0.0:
        logger.warning(
            "POI called with xi=%.4f. With xi<=0, POI is degenerate. It "
            "treats a tiny improvement identically to a large one and clusters "
            "near the current best. Consider xi=0.01 or use EI instead.", xi
        )
    sigma = np.clip(sigma, 1e-9, None)
    Z = (mu - f_best - xi) / sigma
    return norm.cdf(Z)


def compute_acquisition(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    method: AcquisitionName = "EI",
    kappa: float | None = None,
    xi: float | None = None,
    n_obs: int | None = None,
) -> np.ndarray:
    """Dispatch to the requested acquisition function.

    Args:
        mu: Predicted mean for each candidate.
        sigma: Predicted standard deviation for each candidate.
        f_best: Best response value observed so far.
        method: Acquisition function to use. Must be "EI", "UCB", or "POI".
        kappa: UCB exploration weight. If None, it is auto-computed as
            `sqrt(2 * ln(n_obs))`. Ignored for EI and POI.
        xi: Improvement buffer for EI and POI. Defaults are 0.0 for EI and 0.01
            for POI. Ignored for UCB.
        n_obs: Number of existing observations. Required by UCB when `kappa` is
            None.

    Returns:
        The acquisition score for each candidate.

    Notes:
        EI uses `xi = 0` by default and usually needs no tuning.
        UCB requires either an explicit `kappa` or `n_obs` for auto-scheduling.
        POI should use `xi > 0` to avoid degenerate behavior.
    """
    if method == "EI":
        return expected_improvement(mu, sigma, f_best, xi if xi is not None else 0.0)
    if method == "UCB":
        return upper_confidence_bound(mu, sigma, kappa=kappa, n_obs=n_obs)
    if method == "POI":
        return probability_of_improvement(mu, sigma, f_best, xi if xi is not None else 0.01)
    raise ValueError(f"Unknown acquisition method '{method}'. Choose EI, UCB, or POI.")


def _predict_rf_with_uncertainty(
    rf_model: TransferForest,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(mu, sigma)` for an RF model using per-tree prediction variance."""

    preds_dict = predict_rf_model(rf_model, X)
    pred_df = pd.DataFrame(preds_dict)

    # Individual transfer-tree columns: pred_0, pred_1, … (not source, ensemble, val)
    tree_cols = [
        c for c in pred_df.columns
        if c.startswith("pred_")
        and c not in ("pred_source", "pred_ensemble")
        and not c.endswith("_val")
        and not c.endswith("_prob_1")   # classification probabilities
        and not c.endswith("_prob_2")
        and not c.endswith("_prob_3")
    ]

    if "pred_ensemble" in pred_df.columns:
        mu = pred_df["pred_ensemble"].to_numpy(dtype=np.float64)
    elif tree_cols:
        mu = pred_df[tree_cols].mean(axis=1).to_numpy(dtype=np.float64)
    else:
        mu = pred_df.iloc[:, -1].to_numpy(dtype=np.float64)

    if len(tree_cols) >= 2:
        sigma = pred_df[tree_cols].std(axis=1).to_numpy(dtype=np.float64)
    else:
        # Fallback: no variance available. Use a small constant
        logger.warning("RF has fewer than 2 tree columns; setting sigma = 0.01.")
        sigma = np.full(len(mu), 0.01)

    return mu, sigma


def _predict_dl_with_uncertainty(
    dl_model,
    X: pd.DataFrame,
    model_id: str = "target",
    n_mc_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(mu, sigma)` for a DL model using Monte Carlo dropout.

    The model is temporarily set to training mode so dropout remains active
    during inference. `n_mc_samples` stochastic forward passes are averaged.
    """
    model_obj = getattr(dl_model, model_id)
    net = model_obj.model

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    if net.device is not None:
        X_tensor = X_tensor.to(net.device)

    # MC Dropout: keep network in train mode
    net.train()
    samples: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_mc_samples):
            out = net([X_tensor])
            yhat = out[0]  # first output is always the prediction
            samples.append(yhat.detach().cpu().numpy().flatten())
    net.eval()

    arr = np.stack(samples, axis=0)   # (n_mc_samples, n_points)
    mu    = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    return mu, sigma


def predict_with_uncertainty(
    model: TransferModel,
    X: pd.DataFrame,
    feature_cols: list[str],
    n_mc_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(mu, sigma)` for each row of `X`.

    Args:
        model: The TransferModel to predict with.
        X: Candidate DataFrame. Must contain `feature_cols`.
        feature_cols: Feature columns to pass to the model.
        n_mc_samples: Number of stochastic passes for DL Monte Carlo dropout.

    Returns:
        Predicted mean and standard deviation arrays for each row of `X`.
    """
    X_in = X[feature_cols].copy()

    if model.type == "rf":
        assert type(model.model) is TransferForest
        return _predict_rf_with_uncertainty(model.model, X_in)

    if model.type == "dl":
        assert type(model.model) is TransferVAE or type(model.model) is TransferMLP
        return _predict_dl_with_uncertainty(
            model.model, X_in, n_mc_samples=n_mc_samples
        )

    raise ValueError(f"Unknown model type '{model.model}'. Expected 'dl' or 'rf'.")


def recommend_next_batch(
    model_info: TransferModel,
    existing_data: pd.DataFrame,
    response_col: str,
    feature_cols: list[str],
    feature_ranges: dict[str, tuple[float, float]] | None = None,
    expansion_pct: float = 0.0,
    step_sizes: dict[str, float] | None = None,
    n_candidates: int = 5_000,
    batch_size: int = 20,
    acquisition: AcquisitionName = "EI",
    kappa: float | None = None,
    xi: float | None = None,
    n_mc_samples: int = 50,
    shortlist_pct: float = 0.10,
    seed: int = 42,
    return_candidates: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Recommend the next batch of experiments using Bayesian optimization.

    This is a domain-agnostic function that works with any omicsTL model and
    any dataset. It accounts for all existing observations and uses an
    acquisition function to balance exploitation and exploration.

    Args:
        model_info: The TransferModel to predict with.
        existing_data: All observations collected so far. Must contain the
            feature columns and the response column. The response column is
            used to determine the current best value.
        response_col: Name of the response column in `existing_data`.
        feature_cols: Names of the feature columns. Must match what the model
            expects.
        feature_ranges: Search bounds of the form `{col: (min, max)}`
            representing physically feasible limits for each feature. This is
            the recommended way to control the search space. Define it from
            what is experimentally possible, not only from what has been
            observed so far. If None, the observed range in `existing_data` is
            used.
        expansion_pct: Fractional expansion applied symmetrically to each
            feature range after the range is resolved. `0.0` means no
            expansion. For example, `0.1` expands each side by 10% of the span,
            so `[0, 100]` becomes `[-10, 110]`. Useful as a lightweight
            exploration knob when physical bounds are not known in advance.
        step_sizes: Optional per-feature grid spacing of the form
            `{col: step}`. Omit a feature to keep it continuous.

            Example:
                step_sizes = {
                    "concentration_mM": 25,
                    "pH": 0.5,
                    "temperature": 0.25,
                }

            The step value also determines output precision. Integer steps give
            integer values, 0.5 gives one decimal place, and 0.25W gives two.
            Features absent from this dict are sampled as continuous floats.
        n_candidates: Number of candidate points to score.
        batch_size: Number of experiments to recommend.
        acquisition: Acquisition function to use: "EI" for Expected
            Improvement, "UCB" for Upper Confidence Bound, or "POI" for
            Probability of Improvement. EI is the recommended default.
        kappa: UCB exploration weight. Ignored for EI and POI. If None, it is
            auto-computed as `sqrt(2 * ln(n_obs))`, which increases with
            dataset size so exploration is maintained.
        xi: Improvement buffer for EI and POI. Ignored for UCB. If None, EI
            uses 0.0 and POI uses 0.01. With xi = 0, POI becomes degenerate
            because it treats any positive improvement, however small, the
            same.
        n_mc_samples: Number of Monte Carlo dropout passes for DL models.
        shortlist_pct: Fraction of Sobol candidates to shortlist by acquisition
            score before applying max-min diversity selection. Default is 0.10
            (top 10%). The shortlist size is always at least `batch_size`.
        seed: Random seed for reproducible candidate sampling.
        return_candidates: If True, return `(batch, candidates)`, where
            `candidates` is the full scored candidate pool with
            `predicted_mean`, `predicted_std`, and `acquisition_score`. Useful
            for diagnostic plots.

    Returns:
        Either the recommended batch alone, or `(batch, candidates)` if
        `return_candidates` is True. The batch includes `feature_cols` plus
        `predicted_mean`, `predicted_std`, `acquisition_score`, and
        `batch_rank`, sorted by `batch_rank`.
    """
    rng = np.random.default_rng(seed)

    if feature_ranges is None:
        logger.warning(
            "feature_ranges was not provided. Candidate search space is being "
            "inferred from the observed data range, which restricts the optimizer "
            "to interpolation only. It cannot suggest experiments outside the "
            "values already seen. Pass feature_ranges={col: (min, max)} with "
            "physically meaningful bounds (like BacterAI's pre-specified "
            "concentration levels) to allow exploration of the full feasible "
            "space. Use expansion_pct to expand the observed range by a fixed "
            "fraction if explicit bounds are not available."
        )
        feature_ranges = {
            col: (float(existing_data[col].min()), float(existing_data[col].max()))
            for col in feature_cols
        }

    if expansion_pct > 0.0:
        expanded = {}
        for col, (lo, hi) in feature_ranges.items():
            margin = (hi - lo) * expansion_pct
            expanded[col] = (lo - margin, hi + margin)
            logger.debug(
                "Feature '%s': expanded [%.4g, %.4g] → [%.4g, %.4g] "
                "(expansion_pct=%.2f)",
                col, lo, hi, lo - margin, hi + margin, expansion_pct,
            )
        feature_ranges = expanded

    for col in feature_cols:
        lo, hi = feature_ranges[col]
        if lo == hi:
            feature_ranges[col] = (lo - 1e-6, hi + 1e-6)

    # Generate candidates (Sobol low-discrepancy sequences)
    #
    # Sobol requires power-of-2 sample sizes for optimal coverage.
    # Discrete features (step_sizes) are mapped from the Sobol [0,1) dimension
    # to allowed levels proportionally, ensuring uniform level distribution.
    step_sizes = step_sizes or {}
    _allowed: dict[str, np.ndarray] = {}
    for col in feature_cols:
        if col in step_sizes:
            lo, hi = feature_ranges[col]
            lvls = _make_allowed_values(lo, hi, step_sizes[col])
            if len(lvls) == 0:
                raise ValueError(
                    f"step_sizes['{col}']={step_sizes[col]} produces no allowed "
                    f"values in range [{lo}, {hi}]."
                )
            _allowed[col] = lvls
            logger.debug(
                "Feature '%s': %d discrete levels (step=%s, range=[%s, %s])",
                col, len(lvls), step_sizes[col], lo, hi,
            )

    n_sobol = _next_power_of_2(n_candidates)
    if n_sobol != n_candidates:
        logger.info(
            "Sobol requires power-of-2 sample size; rounding n_candidates %d → %d.",
            n_candidates, n_sobol,
        )

    sampler    = _qmc.Sobol(d=len(feature_cols), scramble=True, rng=rng)
    sobol_unit = sampler.random(n_sobol) # shape (n_sobol, n_features), values in [0, 1)

    cand_dict: dict[str, np.ndarray] = {}
    for j, col in enumerate(feature_cols):
        lo, hi = feature_ranges[col]
        u = sobol_unit[:, j]
        if col in _allowed:
            # Discrete: map [0, 1) uniformly to allowed levels. Gives exactly
            # n_sobol / n_levels candidates per level (perfectly stratified).
            lvls = _allowed[col]
            idx  = np.minimum((u * len(lvls)).astype(int), len(lvls) - 1)
            cand_dict[col] = lvls[idx]
        else:
            # Continuous: linear scale from [0, 1) to [lo, hi).
            cand_dict[col] = lo + u * (hi - lo)

    candidates = pd.DataFrame(cand_dict)

    logger.info(
        "Scoring %d Sobol candidates with %s acquisition (model type: %s)",
        n_sobol, acquisition, model_info.type,
    )
    mu, sigma = predict_with_uncertainty(
        model_info, candidates, feature_cols, n_mc_samples
    )
    candidates["predicted_mean"] = mu
    candidates["predicted_std"]  = sigma

    f_best = float(existing_data[response_col].max())
    logger.info("Current best observed response: %.6f", f_best)

    n_obs = len(existing_data)
    scores = compute_acquisition(
        mu, sigma, f_best,
        method=acquisition, kappa=kappa, xi=xi, n_obs=n_obs,
    )
    candidates["acquisition_score"] = scores

    # Select batch: top-EI shortlist + greedy max-min diversity
    norm_candidates = np.column_stack([
        (np.asarray(candidates[col], dtype=np.float64) - feature_ranges[col][0])
        / (feature_ranges[col][1] - feature_ranges[col][0])
        for col in feature_cols
    ])

    acq_vals    = candidates["acquisition_score"].values
    selected_idx = _maximin_select(
        norm_candidates, np.asarray(acq_vals, dtype=np.float64), batch_size, shortlist_pct,
    )

    logger.info(
        "Selected %d candidates via top-%.0f%% EI shortlist + max-min diversity.",
        len(selected_idx), shortlist_pct * 100,
    )

    empty_batch = pd.DataFrame(
        columns=feature_cols
        + ["predicted_mean", "predicted_std", "acquisition_score", "batch_rank"]
    )
    if len(selected_idx) == 0:
        logger.error("No candidates were selected.")
        return (empty_batch, candidates) if return_candidates else empty_batch

    # Rank within the batch by acquisition score (rank 1 = highest EI)
    batch_acq  = np.asarray(acq_vals[selected_idx], dtype=np.float64)
    rank_order = np.argsort(-batch_acq)
    sorted_idx = selected_idx[rank_order]

    result = candidates.iloc[sorted_idx][
        feature_cols + ["predicted_mean", "predicted_std", "acquisition_score"]
    ].copy()
    result["batch_rank"] = np.arange(1, len(sorted_idx) + 1)
    result = result.reset_index(drop=True)

    logger.info(
        "Best acquisition score: %.6f   Best predicted mean: %.6f",
        result["acquisition_score"].iloc[0],
        result["predicted_mean"].max(),
    )

    return (result, candidates) if return_candidates else result

"""
Diagnostic and visualization utilities for omicsTL Bayesian optimization.

Three main plots are provided:

plot_candidate_scores
    Scatter of predicted mean vs uncertainty, colored by acquisition score,
    with the selected batch highlighted. Shows the explore/exploit trade-off.

plot_acquisition_landscape
    2-D heatmap over any two features (all others fixed at the best-observed
    values). Two panels: predicted mean surface and acquisition score surface.
    Existing observations and the recommended batch are overlaid.

plot_batch_coverage
    Parallel-coordinates view comparing the recommended batch to existing data.
    Quickly reveals whether the batch is diverse or collapsing to one cluster.

Typical workflow:
>>> batch, candidates = recommend_next_batch(..., return_candidates=True)
>>> fig1 = plot_candidate_scores(candidates, batch)
>>> fig2 = plot_acquisition_landscape(model_info, existing, "response", feats,
...                                   x_col="pH", y_col="temperature", batch=batch)
>>> fig3 = plot_batch_coverage(batch, existing, feats)
"""

import logging
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

from omicstl.transfer_model import TransferModel

logger = logging.getLogger(__name__)


def plot_candidate_scores(
    candidates: pd.DataFrame,
    batch: pd.DataFrame | None = None,
    acquisition_col: str = "acquisition_score",
    figsize: tuple[float, float] = (12, 4.5),
) -> Figure:
    """Predicted mean vs uncertainty, colored by acquisition score.

    Args:
        candidates: full scored candidate pool returned by
            `recommend_next_batch(..., return_candidates=True)`.
        batch: recommended batch DataFrame (output of recommend_next_batch). If
            provided, batch points are overlaid as red stars.
    acquisition_col : column name for the acquisition score in candidates.
    figsize    : figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Panel 1: mu vs sigma scatter ──────────────────────────────────────────
    sc = axes[0].scatter(
        candidates["predicted_std"],
        candidates["predicted_mean"],
        c=candidates[acquisition_col],
        cmap="viridis",
        alpha=0.35,
        s=10,
        rasterized=True,
    )
    plt.colorbar(sc, ax=axes[0], label="Acquisition score")

    if batch is not None and len(batch) > 0:
        axes[0].scatter(
            batch["predicted_std"],
            batch["predicted_mean"],
            c="red",
            marker="*",
            s=180,
            zorder=6,
            label=f"Batch (n={len(batch)})",
        )
        axes[0].legend(fontsize=9)

    axes[0].set_xlabel("Predicted std  (uncertainty)")
    axes[0].set_ylabel("Predicted mean")
    axes[0].set_title("Explore-exploit landscape")

    # ── Panel 2: acquisition score histogram ──────────────────────────────────
    axes[1].hist(
        candidates[acquisition_col],
        bins=60,
        color="steelblue",
        edgecolor="white",
        linewidth=0.4,
    )
    if batch is not None and len(batch) > 0:
        threshold = float(batch[acquisition_col].min())
        axes[1].axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Batch minimum  ({threshold:.4f})",
        )
        axes[1].legend(fontsize=9)

    axes[1].set_xlabel("Acquisition score")
    axes[1].set_ylabel("Candidate count")
    axes[1].set_title("Acquisition score distribution")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    return fig


def plot_acquisition_landscape(
    model: TransferModel,
    existing_data: pd.DataFrame,
    response_col: str,
    feature_cols: list[str],
    x_col: str,
    y_col: str,
    batch: pd.DataFrame | None = None,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
    n_grid: int = 50,
    acquisition: Literal['EI', 'UCB', 'POI'] = "EI",
    kappa: float | None = None,
    xi: float | None = None,
    n_mc_samples: int = 20,
    figsize: tuple[float, float] = (13, 5),
) -> Figure:
    """2-D heatmap of predicted mean and acquisition score over two features.

    All features not on the axes are fixed at the values observed in the
    best-performing row of existing_data (i.e. the row with the highest
    response).

    Args:
        model: the TransferModel to predict with.
        existing_data: observations so far (must contain feature_cols +
            response_col).
        response_col: name of the response column.
        feature_cols: full feature column list.
        x_col: the feature column to plot on the x axis
        y_col: the feature column to plot on the y axis.
        batch: if provided, recommended batch points are overlaid.
        feature_ranges: `{col: (min, max)}`.  Defaults to observed range.
        n_grid: grid resolution per axis (n_grid^2 points scored).
        acquisition: 'EI' (default), 'UCB', or 'POI'.
        kappa: UCB exploration weight. See `recommend_next_batch` for details.
        xi: Improvement buffer for EI and POI. See `recommend_next_batch` for
            details.
        n_mc_samples: number of stochastic passes for DL MC Dropout.
        figsize: tuple specifying the figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from omicstl.simulation_utils.recommendation_utils import (
        predict_with_uncertainty,
        compute_acquisition,
    )

    if feature_ranges is None:
        feature_ranges = {
            col: (float(existing_data[col].min()), float(existing_data[col].max()))
            for col in feature_cols
        }

    best_idx = existing_data[response_col].idxmax()
    best_row = existing_data.loc[best_idx]

    x_vals = np.linspace(*feature_ranges[x_col], n_grid)
    y_vals = np.linspace(*feature_ranges[y_col], n_grid)
    XX, YY = np.meshgrid(x_vals, y_vals)
    n_pts = n_grid * n_grid

    best_feature_vals = np.asarray(
        existing_data[feature_cols].loc[best_idx],
        dtype=np.float64,
    )

    grid = pd.DataFrame({
        col: np.full(n_pts, best_feature_vals[i], dtype=np.float64)
        for i, col in enumerate(feature_cols)
    })
    grid[x_col] = XX.ravel()
    grid[y_col] = YY.ravel()

    logger.info(
        "Scoring %d grid points for acquisition landscape (%s vs %s).",
        n_pts, x_col, y_col,
    )
    mu, sigma = predict_with_uncertainty(model, grid, feature_cols, n_mc_samples)

    f_best = float(existing_data[response_col].max())
    n_obs  = len(existing_data)
    acq    = compute_acquisition(
        mu, sigma, f_best,
        method=acquisition, kappa=kappa, xi=xi, n_obs=n_obs,
    )

    MU  = mu.reshape(n_grid, n_grid)
    ACQ = acq.reshape(n_grid, n_grid)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: predicted mean surface
    cf1 = axes[0].contourf(XX, YY, MU, levels=25, cmap="RdYlGn")
    plt.colorbar(cf1, ax=axes[0], label="Predicted mean")

    sc1 = axes[0].scatter(
        existing_data[x_col], existing_data[y_col],
        c=existing_data[response_col],
        cmap="RdYlGn",
        edgecolors="black",
        linewidths=0.6,
        s=55,
        zorder=5,
        label="Existing obs",
    )
    if batch is not None and len(batch) > 0:
        axes[0].scatter(
            batch[x_col], batch[y_col],
            marker="*", color="red", s=220, zorder=6, label="Recommended batch",
        )
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)
    axes[0].set_title("Predicted mean surface")
    axes[0].legend(fontsize=8, loc="best")

    # Panel 2: acquisition score surface
    cf2 = axes[1].contourf(XX, YY, ACQ, levels=25, cmap="plasma")
    plt.colorbar(cf2, ax=axes[1], label=f"{acquisition} score")

    axes[1].scatter(
        existing_data[x_col], existing_data[y_col],
        c="white", edgecolors="black", linewidths=0.6,
        s=55, zorder=5, label="Existing obs",
    )
    if batch is not None and len(batch) > 0:
        axes[1].scatter(
            batch[x_col], batch[y_col],
            marker="*", color="red", s=220, zorder=6, label="Recommended batch",
        )
    axes[1].set_xlabel(x_col)
    axes[1].set_ylabel(y_col)
    axes[1].set_title(f"{acquisition} acquisition landscape")
    axes[1].legend(fontsize=8, loc="best")

    # Shared subtitle showing which features are held fixed
    fixed_cols = [c for c in feature_cols if c not in (x_col, y_col)]
    if fixed_cols:
        fixed_str = ", ".join(
            f"{c}={best_row[c]:.3g}" for c in fixed_cols[:6]
        )
        if len(fixed_cols) > 6:
            fixed_str += f" … (+{len(fixed_cols)-6} more)"
        fig.text(
            0.5, -0.01,
            f"Other features fixed at best-observed values: {fixed_str}",
            ha="center", va="top", fontsize=7.5, color="dimgray",
        )

    fig.tight_layout()
    return fig


def plot_batch_coverage(
    batch: pd.DataFrame,
    existing_data: pd.DataFrame,
    feature_cols: list[str],
    response_col: str | None = None,
    figsize: tuple[float, float] = (14, 4.5),
) -> Figure:
    """Parallel-coordinates view of the recommended batch vs existing data.

    Each line represents one experiment. Existing observations are shown in
    blue (shaded); the recommended batch in red. A good batch should spread
    across the feature axes rather than cluster together.

    Args:
        batch: recommended batch (output of `recommend_next_batch`).
        existing_data: all observations so far.
        feature_cols: feature columns to plot.
        response_col: if provided, existing data lines are colored by response
            value rather than a uniform blue.
        figsize: tuple specifying the figure size in inches.

    Returns:
        A matplotlib Figure describing the batch coverage.
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_feat = len(feature_cols)
    x_pos  = np.arange(n_feat, dtype=float)

    mins  = existing_data[feature_cols].min()
    maxs  = existing_data[feature_cols].max()
    spans = (maxs - mins).replace(0.0, 1.0)

    def _norm(row: pd.Series) -> np.typing.NDArray[np.float64]:
        return np.asarray((row[feature_cols] - mins) / spans, dtype=np.float64)

    # Existing observations
    if response_col is not None and response_col in existing_data.columns:
        resp = np.asarray(existing_data[response_col], dtype=float)
        r_min, r_max = resp.min(), resp.max()
        r_span = max(r_max - r_min, 1e-9)
        cmap = plt.get_cmap("Blues")
        for i, (_, row) in enumerate(existing_data.iterrows()):
            intensity = 0.25 + 0.65 * (resp[i] - r_min) / r_span
            ax.plot(x_pos, _norm(row), color=cmap(intensity), alpha=0.5, linewidth=0.9)
    else:
        for _, row in existing_data.iterrows():
            ax.plot(x_pos, _norm(row), color="steelblue", alpha=0.20, linewidth=0.8)

    for _, row in batch.iterrows():
        ax.plot(x_pos, _norm(row), color="red", alpha=0.80, linewidth=2.0, zorder=5)

    # Legend
    legend_handles = [
        Line2D([0], [0], color="steelblue", alpha=0.6, linewidth=1.5,
               label=f"Existing data  (n={len(existing_data)})"),
        Line2D([0], [0], color="red", linewidth=2,
               label=f"Recommended batch  (n={len(batch)})"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    # Axes decoration
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_cols, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Normalized value  [0 = min, 1 = max]", fontsize=9)
    ax.set_title("Batch coverage vs existing data (parallel coordinates)")
    ax.set_ylim(-0.05, 1.05)

    for xp in x_pos:
        ax.axvline(xp, color="lightgray", linewidth=0.8, zorder=0)

    fig.tight_layout()
    return fig

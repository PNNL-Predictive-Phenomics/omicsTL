import os
import typing
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.rinterface_lib.sexp import NULLType
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.special import expit
from .._defs import _PKG_ROOT
from ..r_utils import df2pd, pd2df

def response_function(func_str : str) -> ro._reval:
    """Create a response function from a string.

    Args:
        func_str: a response function in R syntax. Input data will be passed as
            a data.frame called `df`.

    Returns:
        A response function object
    """
    return ro.reval(f"\\(df) {func_str}")

def generate_synth_data(
        data : pd.DataFrame,
        num_features : int,
        num_samples : int,
        response_fn : ro._reval,
        response_parameters : dict | None = None,
        snr : float = 1,
        prior_lc_info : pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, typing.Union[ro.FloatVector, NULLType]]:
    """Generate synthetic data based on real data

    Args:
        data: a DataFrame containing the real data to generate synthetic data from
        num_features: the number of features the resulting synthetic DataFrame should have
        num_samples: the number of samples the resulting synthetic DataFrame should have
        response_fn: the function used to generate the response column. This can be created
            using `response_function` function
        response_parameters: (optional) if set to None, a continuous response will be
            generated. Otherwise, this should be a dict with an "ncats" key specifying
            the number of categories, and a "quantile" key which should be "quantile",
            "random", or a list of cut points. Default is None (continuous response)
        snr: (optional) the desired signal to noise ratio. Lower values inject more
            random noise into the synthetic data. Default is 1 (equal parts signal and noise)
        prior_lc_info: (optional) linear combination info returned from a previous call to
            this function to ensure target and source synthetic datasets are paired.
            Default is None (no prior linear combination info)
    
    Returns:
        A tuple containing the generated synthetic data, the linear combination info for the generated synthetic data, and a list of cut points if a categorical response was used (else NULL)
    """
    
    ro.r["Sys.setenv"](OMICSTL_PKG_ROOT = _PKG_ROOT()) # type: ignore
    ro.r["source"](os.path.join(_PKG_ROOT(), "r", "requirements.R")) # type: ignore
    ro.r["source"](os.path.join(_PKG_ROOT(), "r", "data_simulation_utils.R")) # type: ignore

    response_parameters_r = None
    if response_parameters is not None:
        ncats = response_parameters.get("ncats", None)
        quantile = response_parameters.get("quantile", None)
        if type(quantile) is list:
            quantile = ro.FloatVector(quantile)
        if ncats is not None:
            response_parameters_r = ro.ListVector({"ncats": ncats, "quantile": quantile})

    if prior_lc_info is None:
        result_data = ro.r["data_generator_wrapper"](
        data=pd2df(data),
        num_features=num_features,
        num_samples=num_samples,
        response_fn=response_fn,
        response_parameters=ro.NULL if response_parameters is None else response_parameters_r,
        snr=snr
        ) # type: ignore
    else:
        result_data = ro.r["data_generator_wrapper"](
        data=pd2df(data),
        num_features=num_features,
        num_samples=num_samples,
        response_fn=response_fn,
        response_parameters=ro.NULL if response_parameters is None else response_parameters_r,
        snr=snr,
        # Including this is crucial. Otherwise, target and source datasets are
        # not paired!
        prior_lc_info=pd2df(prior_lc_info),
        )  # type: ignore

    return pd.DataFrame(df2pd(result_data.rx2("data"))), df2pd(result_data.rx2("lc_info")), result_data.rx2("cut_point")


def generate_synth_data_pca(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    n_output_samps_source: int,
    n_output_samps_target: int,
    n_output_features: int | None = None,
    n_components: int | None = None,
    n_active_components: int = 4,
    pc_selection: str = "activity",
    alpha: float = 1.0,
    regularization: float = 1e-3,
    snr: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate paired synthetic source and target data via Gaussian PCA simulation.

    Fits a shared PCA basis on the real source, projects both real domains
    into that basis, estimates a Gaussian score distribution per domain, then
    samples from an alpha-interpolated target distribution.

    alpha=0 targets the same distribution as source (no domain shift);
    alpha=1 targets the full observed domain shift.

    Args:
        source_data: Feature-only DataFrame for the source domain.
        target_data: Feature-only DataFrame for the target domain.
        n_output_samps_source: Number of synthetic source samples to generate.
        n_output_samps_target: Number of synthetic target samples to generate.
        n_output_features: Unused; kept for signature compatibility. Must be None.
        n_components: Total PCA components to fit. Defaults to min(n_src-1, p).
        n_active_components: Number of PCs to sample from and return as scores
            (K_active). Default 4.
        pc_selection: How to choose the K_active PCs.
            "variance" — first K by source variance explained (standard PCA order).
            "activity" — K PCs ranked by source_var / (1 + normalised_shift²),
                         deprioritising components dominated by domain shift.
            Default "activity".
        alpha: Domain-shift interpolation in [0, 1]. 0 = same as source;
            1 = full observed target shift. Default 1.0.
        regularization: Ridge term added to PC-score covariance matrices for
            positive-definiteness. Default 1e-3.
        snr: Feature-level signal-to-noise ratio. If provided, per-feature
            Gaussian noise is injected so that noise_var_j = signal_var_j / snr.
            None (default) adds no noise.
        random_state: Seed for numpy RNG. Default None.

    Returns:
        (synth_source, synth_target, source_scores_syn, target_scores_syn) where
        source_scores_syn and target_scores_syn are K_active-dimensional sampled
        PC score arrays. Pass directly to generate_pca_response.
    """
    if pc_selection not in ("variance", "activity"):
        raise ValueError(f"pc_selection must be 'variance' or 'activity', got {pc_selection!r}")
    if n_output_features is not None:
        raise NotImplementedError(
            "n_output_features is not supported; pass None to keep all common features."
        )

    rng = np.random.default_rng(random_state)

    src_num = source_data.select_dtypes(include="number")
    tgt_num = target_data.select_dtypes(include="number")
    common_cols = src_num.columns.intersection(tgt_num.columns).tolist()
    src_arr = src_num[common_cols].values.astype(float)
    tgt_arr = tgt_num[common_cols].values.astype(float)
    n_src, p = src_arr.shape

    feat_mean = src_arr.mean(axis=0)
    feat_sd = src_arr.std(axis=0, ddof=1)
    feat_sd[feat_sd < 1e-10] = 1.0
    src_scaled = (src_arr - feat_mean) / feat_sd
    tgt_scaled = (tgt_arr - feat_mean) / feat_sd

    max_comp = min(n_src - 1, p)
    n_comp = max_comp if n_components is None else min(int(n_components), max_comp)
    pca = PCA(n_components=n_comp).fit(src_scaled)
    V = pca.components_.T   # (p, n_comp)

    Z_src = pca.transform(src_scaled)            # (n_src, n_comp)
    Z_tgt = (tgt_scaled - pca.mean_) @ V        # (n_tgt, n_comp)

    K = min(n_active_components, n_comp)

    if pc_selection == "variance":
        active_idx = np.arange(K)   # first K by source variance (standard PCA order)
    else:
        # "activity": rank by source_var / (1 + normalised_shift²) so components
        # dominated by between-domain mean difference score lower.
        var_S_full    = Z_src.var(axis=0, ddof=1).clip(min=1e-10)
        shift_sq_full = (Z_tgt.mean(axis=0) - Z_src.mean(axis=0)) ** 2
        activity_score = var_S_full / (1.0 + shift_sq_full / var_S_full)
        active_idx = np.argsort(activity_score)[::-1][:K]

    Z_src_k = Z_src[:, active_idx]
    Z_tgt_k = Z_tgt[:, active_idx]

    mu_S = Z_src_k.mean(axis=0)
    mu_T = Z_tgt_k.mean(axis=0)
    Sigma_S = np.atleast_2d(np.cov(Z_src_k, rowvar=False, ddof=1)) + regularization * np.eye(K)
    Sigma_T = np.atleast_2d(np.cov(Z_tgt_k, rowvar=False, ddof=1)) + regularization * np.eye(K)

    mu_T_alpha    = (1.0 - alpha) * mu_S    + alpha * mu_T
    Sigma_T_alpha = (1.0 - alpha) * Sigma_S + alpha * Sigma_T

    Z_src_syn = rng.multivariate_normal(mu_S, Sigma_S, size=n_output_samps_source)
    Z_tgt_syn = rng.multivariate_normal(mu_T_alpha, Sigma_T_alpha, size=n_output_samps_target)

    V_k = V[:, active_idx]   # source loadings for the selected active PCs
    src_rec_scaled = Z_src_syn @ V_k.T + pca.mean_
    tgt_rec_scaled = Z_tgt_syn @ V_k.T + pca.mean_

    if snr is not None:
        sig_var = src_rec_scaled.var(axis=0)
        sig_var[sig_var < 1e-10] = 1e-10
        noise_sd = np.sqrt(sig_var / snr)
        src_rec_scaled += rng.normal(0.0, noise_sd, size=src_rec_scaled.shape)
        tgt_rec_scaled += rng.normal(0.0, noise_sd, size=tgt_rec_scaled.shape)

    src_rec = src_rec_scaled * feat_sd + feat_mean
    tgt_rec = tgt_rec_scaled * feat_sd + feat_mean

    synth_source = pd.DataFrame(src_rec, columns=common_cols)
    synth_target = pd.DataFrame(tgt_rec, columns=common_cols)
    return synth_source, synth_target, Z_src_syn, Z_tgt_syn


def _pca_response_scores(scores: np.ndarray, complexity: str, index) -> pd.Series:
    z1, z2, z3, z4 = scores[:, 0], scores[:, 1], scores[:, 2], scores[:, 3]
    if complexity == "linear":
        raw = z1 + 0.8 * z2 - 0.5 * z3 + 0.35 * z4
    elif complexity == "interaction":
        raw = z1 + 0.5 * z4 + 0.6 * np.tanh(z1 * z2) + 0.4 * np.tanh(z3 * z4)
    elif complexity == "nonlinear":
        raw = np.tanh(z1) + 0.5 * np.sin(z2) + 0.5 * np.tanh(z3 * z4) + 0.3 * np.sin(z4)
    else:
        raise ValueError(f"Unknown complexity: {complexity!r}")
    return pd.Series(raw, index=index)


def _add_response(
    features: pd.DataFrame,
    raw: pd.Series,
    is_categorical: bool,
    snr: float | None = None,
    intercept: float = 0.0,
    gamma: float = 1.0,
) -> tuple[pd.DataFrame, None]:
    if is_categorical:
        # Logistic model: P(Y=1|η) = σ(intercept + gamma·η)
        # gamma controls separation strength; intercept controls base rate.
        # SNR noise is not applied — gamma already governs signal strength.
        p = expit(intercept + gamma * raw.values)
        response = pd.Series(
            np.random.binomial(1, p).astype(np.int64), index=raw.index
        )
    else:
        if snr is not None:
            signal_var = float(np.var(raw.values))
            noise_sd = np.sqrt(signal_var / snr) if signal_var > 0 else 0.0
            raw = raw + np.random.normal(0, noise_sd, size=len(raw))
        response = raw
    df = features.copy()
    df.insert(0, "response", response.values)
    return df, None


def generate_pca_response(
    source_features: pd.DataFrame,
    target_features: pd.DataFrame,
    complexity: str,
    is_categorical: bool = False,
    snr: float | None = None,
    variance_threshold: float = 0.80,
    min_components: int = 3,
    source_scores: np.ndarray | None = None,
    target_scores: np.ndarray | None = None,
    intercept: float = 0.0,
    gamma: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, None]:
    """Generate response variables for source and target datasets using PCA scores.

    When source_scores and target_scores are provided (returned by
    generate_synth_data_pca), they are used directly and no PCA refit is
    performed.  Both domains therefore share the same source loading matrix
    that was used to generate the features, so the response is coherent with
    the synthetic features.

    When scores are not provided, PCA is fitted on source_features and both
    domains are projected onto the source PC space (legacy behaviour, kept for
    backward compatibility).

    For categorical responses, source and target are each standardized by their
    own distribution before the sigmoid threshold, so both domains maintain
    balanced class proportions regardless of domain shift (alpha). Domain shift
    manifests as feature covariate shift, not label imbalance. For continuous
    responses, shared source standardization is used to keep both domains on
    the same response scale.

    Args:
        source_features: Feature-only DataFrame for the source domain.
        target_features: Feature-only DataFrame for the target domain.
        complexity: "linear", "interaction", or "nonlinear" response formula.
        is_categorical: If True, generate binary labels via a logistic model
            P(Y=1|η) = σ(intercept + gamma·η).
        snr: Signal-to-noise ratio applied to continuous responses only.
        variance_threshold: Proportion of source variance the selected PCs must
            explain (only used when scores are not provided). Default 0.80.
        min_components: Minimum PCs to retain (only used when scores are not
            provided). Default 3.
        source_scores: Pre-computed source PC scores (n_source × n_comp) from
            generate_synth_data_pca. When provided together with target_scores,
            skips PCA refit entirely.
        target_scores: Pre-computed target PC scores (n_target × n_comp) from
            generate_synth_data_pca.
        intercept: Log-odds intercept b0. Controls the base class rate when η=0.
            Default 0.0 (balanced source when scores are zero-mean).
        gamma: Log-odds slope γ. Controls separation strength; higher values give
            cleaner class separation. Default 1.0 keeps probabilities away from
            0/1 extremes even under large domain shift, reducing single-class
            target outcomes.

    Returns:
        (source_with_response, target_with_response, None). The third element is
        always None and kept only for a consistent call signature.
    """
    if source_scores is not None and target_scores is not None:
        K = min(source_scores.shape[1], 4)
        src_sc = source_scores[:, :K]
        tgt_sc = target_scores[:, :K]
        if K < 4:
            pad_cols = 4 - K
            src_sc = np.hstack([src_sc, np.zeros((src_sc.shape[0], pad_cols))])
            tgt_sc = np.hstack([tgt_sc, np.zeros((tgt_sc.shape[0], pad_cols))])
    else:
        cumvar = np.cumsum(PCA().fit(source_features).explained_variance_ratio_)
        K = min(
            max(int(np.searchsorted(cumvar, variance_threshold)) + 1, min_components),
            4,                           # formula uses z1..z4
            source_features.shape[1],
        )
        pca_resp = PCA(n_components=K).fit(source_features)
        src_sc = pca_resp.transform(source_features)
        tgt_sc = pca_resp.transform(target_features)
        if K < 4:
            pad_cols = 4 - K
            src_sc = np.hstack([src_sc, np.zeros((src_sc.shape[0], pad_cols))])
            tgt_sc = np.hstack([tgt_sc, np.zeros((tgt_sc.shape[0], pad_cols))])

    # Z-score PC scores using source sample mean and SD.
    # The sampled source scores have population mean 0 but non-zero sample mean
    # due to finite-n randomness; subtracting the estimate is more principled.
    src_mean_sc = src_sc.mean(axis=0)
    src_sd_sc   = src_sc.std(axis=0, ddof=1)
    src_sd_sc[src_sd_sc < 1e-10] = 1.0
    src_sc = (src_sc - src_mean_sc) / src_sd_sc
    tgt_sc = (tgt_sc - src_mean_sc) / src_sd_sc   # same mean+SD for both

    src_raw = _pca_response_scores(src_sc, complexity, source_features.index)
    tgt_raw = _pca_response_scores(tgt_sc, complexity, target_features.index)

    # Z-score latent signal using source sample mean and SD.
    # Guarantees expit(intercept) = exact source class balance when intercept=0,
    # and makes gamma = log-odds per source-SD of eta for all complexity types.
    eta_mean = float(src_raw.mean())
    eta_sd   = float(src_raw.std(ddof=1))
    if eta_sd > 1e-10:
        src_raw = (src_raw - eta_mean) / eta_sd
        if is_categorical:
            # For categorical: standardize target by its own distribution so
            # class balance is preserved regardless of domain shift (alpha).
            # Domain shift manifests as feature covariate shift, not label imbalance.
            tgt_eta_mean = float(tgt_raw.mean())
            tgt_eta_sd   = float(tgt_raw.std(ddof=1))
            if tgt_eta_sd > 1e-10:
                tgt_raw = (tgt_raw - tgt_eta_mean) / tgt_eta_sd
        else:
            # Continuous: shared source standardization keeps source and target
            # response on the same scale for direct comparison.
            tgt_raw = (tgt_raw - eta_mean) / eta_sd

    source_df, _ = _add_response(source_features, src_raw, is_categorical,
                                  snr=snr, intercept=intercept, gamma=gamma)
    target_df, _ = _add_response(target_features, tgt_raw, is_categorical,
                                  snr=snr, intercept=intercept, gamma=gamma)
    return source_df, target_df, None


def generate_synth_data_multiomics_latent(
    source_omics: dict[str, pd.DataFrame],
    target_omics: dict[str, pd.DataFrame],
    n_output_samps_source: int,
    n_output_samps_target: int,
    n_shared_factors: int = 10,
    shared_var_ratio: float = 0.6,
    alpha: float = 0.0,
    snr: float | None = None,
    regularization: float = 1e-3,
    random_state: int | None = None,
    top_n_per_omic: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    """Generate paired synthetic multi-omics source and target data via shared latent factors.

    A joint PCA is fit on the standardized, concatenated source omics to learn K
    shared latent factors that capture cross-omic correlations. Synthetic samples
    are drawn from an alpha-interpolated Gaussian in factor space, then projected
    back into each omic's feature space. Layer-specific noise (controlled by
    shared_var_ratio) is added independently per omic so the cross-omic correlation
    strength is directly tunable.

    When a single omic is passed the function degrades gracefully to a per-omic
    PCA simulation equivalent to generate_synth_data_pca.

    Args:
        source_omics: Dict mapping omic name to feature-only DataFrame (source domain).
        target_omics: Dict mapping omic name to feature-only DataFrame (target domain).
            Must share the same keys as source_omics.
        n_output_samps_source: Number of synthetic source samples to generate.
        n_output_samps_target: Number of synthetic target samples to generate.
        n_shared_factors: Number of shared latent PCs learned from the concatenated
            omics. Controls how many cross-omic directions drive the response.
            Default 10.
        shared_var_ratio: Fraction of total feature variance from the shared PCA
            reconstruction vs. omic-specific noise. 0.9 = highly correlated layers;
            0.3 = mostly independent layers. Default 0.6.
        alpha: Domain-shift interpolation in [0, 1]. 0 = synthetic target drawn from
            same distribution as source. Default 0.0.
        snr: Optional feature-level signal-to-noise ratio applied on top of
            shared+layer noise (same convention as generate_synth_data_pca).
            Default None (no extra noise).
        regularization: Ridge term for PC-score covariance matrices. Default 1e-3.
        random_state: Seed for numpy RNG. Default None.
        top_n_per_omic: If set, retain only the top-N most variable features
            (by source variance) per omic before concatenation. Use this to
            prevent high-dimensional omics (e.g. proteomics) from dominating
            the joint PCA. Features are selected from the common source/target
            columns, so target columns are filtered to match. Default None
            (keep all common features).

    Returns:
        (synth_source, synth_target, Z_src_syn, Z_tgt_syn) where synth_source and
        synth_target are dicts mapping omic name to a feature-only DataFrame, and
        Z_src_syn / Z_tgt_syn are the sampled shared factor score arrays
        (n_samples x n_shared_factors). Pass Z arrays directly to generate_pca_response.
    """
    rng = np.random.default_rng(random_state)
    omic_names = list(source_omics.keys())

    # ── 0. Align samples across omics (real data may have different sample sets) ──
    src_common_samples = source_omics[omic_names[0]].index
    tgt_common_samples = target_omics[omic_names[0]].index
    for k in omic_names[1:]:
        src_common_samples = src_common_samples.intersection(source_omics[k].index)
        tgt_common_samples = tgt_common_samples.intersection(target_omics[k].index)
    source_omics = {k: v.loc[src_common_samples] for k, v in source_omics.items()}
    target_omics = {k: v.loc[tgt_common_samples] for k, v in target_omics.items()}

    # ── 1. Per-omic: find common features, compute source stats, standardize ──
    common_cols_per_omic: dict[str, list[str]] = {}
    src_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    src_scaled_parts: list[np.ndarray] = []
    tgt_scaled_parts: list[np.ndarray] = []
    col_slices: dict[str, tuple[int, int]] = {}
    cursor = 0

    for k in omic_names:
        src_num = source_omics[k].select_dtypes(include="number")
        tgt_num = target_omics[k].select_dtypes(include="number")
        common = src_num.columns.intersection(tgt_num.columns).tolist()

        # Optional: keep only the top-N most variable features (by source variance)
        # to prevent high-feature-count omics from dominating the joint PCA.
        if top_n_per_omic is not None and len(common) > top_n_per_omic:
            src_var = src_num[common].var(axis=0, ddof=1)
            common = src_var.nlargest(top_n_per_omic).index.tolist()
        common_cols_per_omic[k] = common

        src_arr = src_num[common].values.astype(float)
        tgt_arr = tgt_num[common].values.astype(float)

        mu_k = src_arr.mean(axis=0)
        sd_k = src_arr.std(axis=0, ddof=1)
        sd_k[sd_k < 1e-10] = 1.0
        src_stats[k] = (mu_k, sd_k)

        # Target standardized with SOURCE stats to put both domains on the same scale
        src_scaled_parts.append((src_arr - mu_k) / sd_k)
        tgt_scaled_parts.append((tgt_arr - mu_k) / sd_k)

        col_slices[k] = (cursor, cursor + len(common))
        cursor += len(common)

    # ── 2. Concatenate all omics in standardized space ──────────────────────
    src_full = np.hstack(src_scaled_parts)   # (n_src, total_p)
    tgt_full = np.hstack(tgt_scaled_parts)   # (n_tgt, total_p)
    n_src, total_p = src_full.shape

    # ── 3. Fit shared PCA on concatenated source data ───────────────────────
    K = min(n_shared_factors, n_src - 1, total_p)
    pca = PCA(n_components=K).fit(src_full)
    V_k = pca.components_.T                           # (total_p, K)

    Z_src = pca.transform(src_full)                   # (n_src, K)
    Z_tgt = (tgt_full - pca.mean_) @ V_k             # (n_tgt, K)

    # ── 4. Gaussian parameters with alpha-interpolated target distribution ──
    mu_S    = Z_src.mean(axis=0)
    mu_T    = Z_tgt.mean(axis=0)
    Sigma_S = np.atleast_2d(np.cov(Z_src, rowvar=False, ddof=1)) + regularization * np.eye(K)
    Sigma_T = np.atleast_2d(np.cov(Z_tgt, rowvar=False, ddof=1)) + regularization * np.eye(K)

    mu_T_alpha    = (1.0 - alpha) * mu_S    + alpha * mu_T
    Sigma_T_alpha = (1.0 - alpha) * Sigma_S + alpha * Sigma_T

    # ── 5. Sample synthetic latent factors ──────────────────────────────────
    Z_src_syn = rng.multivariate_normal(mu_S, Sigma_S, size=n_output_samps_source)
    Z_tgt_syn = rng.multivariate_normal(mu_T_alpha, Sigma_T_alpha, size=n_output_samps_target)

    # ── 6. Reconstruct in standardized concatenated space ───────────────────
    src_rec_scaled = Z_src_syn @ V_k.T + pca.mean_   # (n_out_src, total_p)
    tgt_rec_scaled = Z_tgt_syn @ V_k.T + pca.mean_   # (n_out_tgt, total_p)

    # ── 7. Split per omic, rescale, add layer-specific and SNR noise ────────
    synth_source: dict[str, pd.DataFrame] = {}
    synth_target: dict[str, pd.DataFrame] = {}

    for k in omic_names:
        start, end = col_slices[k]
        mu_k, sd_k = src_stats[k]

        # Back to original feature scale (shared component only)
        X_src_shared = src_rec_scaled[:, start:end] * sd_k + mu_k
        X_tgt_shared = tgt_rec_scaled[:, start:end] * sd_k + mu_k

        # Layer-specific noise: calibrated so Var(shared)/Var(total) = shared_var_ratio
        sig_var = np.var(X_src_shared, axis=0, ddof=1).clip(min=1e-10)
        ratio   = float(np.clip(shared_var_ratio, 1e-6, 1.0 - 1e-6))
        layer_noise_sd = np.sqrt(sig_var * (1.0 - ratio) / ratio)

        X_src = X_src_shared + rng.normal(0.0, layer_noise_sd, size=X_src_shared.shape)
        X_tgt = X_tgt_shared + rng.normal(0.0, layer_noise_sd, size=X_tgt_shared.shape)

        # Additional feature-level SNR noise (same convention as generate_synth_data_pca)
        if snr is not None:
            total_sig_var = np.var(X_src, axis=0, ddof=1).clip(min=1e-10)
            snr_noise_sd  = np.sqrt(total_sig_var / float(snr))
            X_src += rng.normal(0.0, snr_noise_sd, size=X_src.shape)
            X_tgt += rng.normal(0.0, snr_noise_sd, size=X_tgt.shape)

        synth_source[k] = pd.DataFrame(X_src, columns=common_cols_per_omic[k])
        synth_target[k] = pd.DataFrame(X_tgt, columns=common_cols_per_omic[k])

    return synth_source, synth_target, Z_src_syn, Z_tgt_syn


def generate_synth_data_multiomics_pca(
    source_omics: dict[str, pd.DataFrame],
    target_omics: dict[str, pd.DataFrame],
    n_output_samps_source: int,
    n_output_samps_target: int,
    n_output_features: dict[str, int] | None = None,
    n_components: int | None = None,
    regularization: float = 1e-6,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Generate paired synthetic multi-omics source and target datasets via PCA.

    Each omic layer is standardized to unit variance per feature before
    concatenation so no single omic dominates the shared PC space.  Cross-omic
    correlations are preserved because all layers enter the PCA together.
    Output is split back into per-omic DataFrames at their original scales.

    Args:
        source_omics: Dict mapping omic name to feature-only DataFrame (source domain).
        target_omics: Dict mapping omic name to feature-only DataFrame (target domain).
            Must have the same keys as source_omics.
        n_output_samps_source: Number of synthetic source samples to generate.
        n_output_samps_target: Number of synthetic target samples to generate.
        n_output_features: Optional dict mapping omic name to desired output
            feature count (e.g. {"prot": 100, "rna": 200}).  Values <= p
            subsample randomly; values > p use variance-weighted LC mixing.
            Omics not listed keep all their features.  Default None (keep all).
        n_components: PCA components to retain per domain. Default None (max rank).
        regularization: Ridge term added to PC-space covariance. Default 1e-6.

    Returns:
        (synth_source, synth_target) — each is a dict mapping omic name to a
        feature-only DataFrame of synthetic samples.
    """
    ro.r["Sys.setenv"](OMICSTL_PKG_ROOT=_PKG_ROOT())
    ro.r["source"](os.path.join(_PKG_ROOT(), "r", "requirements.R"))
    ro.r["source"](os.path.join(_PKG_ROOT(), "r", "data_simulation_utils.R"))

    source_r = ro.ListVector({
        k: pd2df(df.select_dtypes(include="number"))
        for k, df in source_omics.items()
    })
    target_r = ro.ListVector({
        k: pd2df(df.select_dtypes(include="number"))
        for k, df in target_omics.items()
    })

    kwargs: dict = dict(
        source_omics=source_r,
        target_omics=target_r,
        n_output_samps_source=int(n_output_samps_source),
        n_output_samps_target=int(n_output_samps_target),
        regularization=float(regularization),
    )
    if n_output_features is not None:
        kwargs["n_output_features"] = ro.ListVector(
            {k: ro.IntVector([v]) for k, v in n_output_features.items()}
        )
    if n_components is not None:
        kwargs["n_components"] = int(n_components)

    result = ro.r["dat_generator_multiomics_pca"](**kwargs)

    synth_source_r = result.rx2("synth_source")
    synth_target_r = result.rx2("synth_target")
    omic_names = list(synth_source_r.names)

    synth_source = {
        name: pd.DataFrame(df2pd(synth_source_r.rx2(name)))
        for name in omic_names
    }
    synth_target = {
        name: pd.DataFrame(df2pd(synth_target_r.rx2(name)))
        for name in omic_names
    }
    return synth_source, synth_target

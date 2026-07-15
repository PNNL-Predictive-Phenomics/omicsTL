import os
import typing
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.rinterface_lib.sexp import NULLType
import pandas as pd
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
    regularization: float = 1e-6,
    snr: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate paired synthetic source and target data via PCA-based simulation.

    Both domains are simulated together using a shared joint PCA so their
    correlation structures are comparable.  Unlike generate_synth_data, no
    response column is added; the returned DataFrames contain features only.

    Args:
        source_data: Feature-only DataFrame for the source domain.
        target_data: Feature-only DataFrame for the target domain.
        n_output_samps_source: Number of synthetic source samples to generate.
        n_output_samps_target: Number of synthetic target samples to generate.
        n_output_features: Number of output features.  If None all p common
            features are returned.  If <= p a random subset is returned with
            exact statistics preserved.  If > p structured LC mixing is used.
        n_components: PCA components to retain.  Defaults to min(n_source-1, p).
        regularization: Ridge term added to PC-space covariance for
            positive-definiteness.  Default 1e-6.
        snr: Signal-to-noise ratio.  If provided, additive Gaussian noise is
            injected into each feature so that noise_var_j = signal_var_j / snr.
            None (default) adds no extra noise.

    Returns:
        (synth_source, synth_target) feature-only DataFrames.
    """
    ro.r["Sys.setenv"](OMICSTL_PKG_ROOT=_PKG_ROOT())
    ro.r["source"](os.path.join(_PKG_ROOT(), "r", "requirements.R"))
    ro.r["source"](os.path.join(_PKG_ROOT(), "r", "data_simulation_utils.R"))

    # Keep only numeric columns before passing to R; non-numeric columns
    # (object/category dtype from CSV row-labels, protein IDs, etc.) become
    # character vectors in R and cause colMeans / svd to fail.
    source_data = source_data.select_dtypes(include="number")
    target_data = target_data.select_dtypes(include="number")

    kwargs: dict = dict(
        source_data=pd2df(source_data),
        target_data=pd2df(target_data),
        n_output_samps_source=int(n_output_samps_source),
        n_output_samps_target=int(n_output_samps_target),
        regularization=float(regularization),
    )
    if n_output_features is not None:
        kwargs["n_output_features"] = int(n_output_features)
    if n_components is not None:
        kwargs["n_components"] = int(n_components)
    if snr is not None:
        kwargs["snr"] = float(snr)

    result = ro.r["dat_generator_pca"](**kwargs)
    synth_source = pd.DataFrame(df2pd(result.rx2("synth_source")))
    synth_target = pd.DataFrame(df2pd(result.rx2("synth_target")))
    return synth_source, synth_target


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

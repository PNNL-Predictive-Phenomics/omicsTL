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
        )
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
        )

    return pd.DataFrame(df2pd(result_data.rx2("data"))), df2pd(result_data.rx2("lc_info")), result_data.rx2("cut_point")


def generate_synth_data_pca(
        source_data: pd.DataFrame,
        target_data: pd.DataFrame,
        num_samples_source: int,
        num_samples_target: int,
        response_fn: ro._reval,
        response_parameters: dict | None = None,
        snr: float = 1,
        n_components: int | None = None,
        regularization: float = 1e-6,
        num_features: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, typing.Union[ro.FloatVector, NULLType]]:
    """Generate paired synthetic source and target data using joint PCA.

    Fits PCA on the combined (source + target) real data so both datasets share
    the same loading matrix, then samples new observations from domain-specific
    Gaussian distributions in PC space and projects back to the original feature
    space.  This preserves the inter-feature covariance structure of the real
    data while allowing the two domains to differ in their means and variances.

    The same response function is applied to both domains.  For categorical
    responses, the cut points derived from the synthetic source data are reused
    for the target to keep the two datasets linked.

    Args:
        source_data: DataFrame of real source data (samples × features).
        target_data: DataFrame of real target data (samples × features).
            Must share the same column names as source_data.
        num_samples_source: Number of synthetic source samples to generate.
        num_samples_target: Number of synthetic target samples to generate.
        response_fn: Response function created by response_function().
        response_parameters: (optional) Dict with "ncats" (int) and "quantile"
            ("quantile", "random", or list of cut points) for a categorical
            response.  Pass None for a continuous response.  Default None.
        snr: Signal-to-noise ratio applied when generating the response column.
            Default 1 (equal parts signal and noise).
        n_components: Number of PCs to retain.  Defaults to
            min(n_source - 1, n_target - 1, p), which guarantees non-singular
            per-domain covariance matrices without regularisation.  Increase
            alongside regularization to capture more variance.
        regularization: Small ridge constant added to each per-domain PC-space
            covariance matrix before sampling.  Default 1e-6.
        num_features: Number of synthetic features to produce.  When None
            (default) the output has the same number of features as the real
            data.  When set, a shared random convex-combination matrix is
            applied to the PC scores so the output covariance reflects the
            real data's covariance structure regardless of the requested
            feature count.

    Returns:
        A tuple of (source_synth_data, target_synth_data, source_cut_points,
        target_cut_points) where source_synth_data and target_synth_data are
        DataFrames with the response prepended as the first column.
        source_cut_points and target_cut_points are domain-specific vectors of
        category boundaries (NULL for a continuous response).
    """

    ro.r["Sys.setenv"](OMICSTL_PKG_ROOT=_PKG_ROOT())  # type: ignore
    ro.r["source"](os.path.join(_PKG_ROOT(), "r", "requirements.R"))  # type: ignore
    ro.r["source"](os.path.join(_PKG_ROOT(), "r", "data_simulation_utils.R"))  # type: ignore

    response_parameters_r = None
    if response_parameters is not None:
        ncats = response_parameters.get("ncats", None)
        quantile = response_parameters.get("quantile", None)
        if type(quantile) is list:
            quantile = ro.FloatVector(quantile)
        if ncats is not None:
            response_parameters_r = ro.ListVector({"ncats": ncats, "quantile": quantile})

    result = ro.r["data_generator_wrapper_pca"](
        source_data=pd2df(source_data),
        target_data=pd2df(target_data),
        n_output_samps_source=num_samples_source,
        n_output_samps_target=num_samples_target,
        response_fn=response_fn,
        response_parameters=ro.NULL if response_parameters is None else response_parameters_r,
        snr=snr,
        n_components=ro.NULL if n_components is None else n_components,
        regularization=regularization,
        n_output_features=ro.NULL if num_features is None else num_features,
    )

    source_synth      = pd.DataFrame(df2pd(result.rx2("source_data")))
    target_synth      = pd.DataFrame(df2pd(result.rx2("target_data")))
    cut_point         = result.rx2("cut_point")
    target_cut_point  = result.rx2("target_cut_point")

    return source_synth, target_synth, cut_point, target_cut_point
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

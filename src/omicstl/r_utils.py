"""Utilities for interacting with R objects from python."""
import pandas as pd
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.vectors import DataFrame

def df2pd(df: DataFrame) -> pd.DataFrame:
    """Convert R dataframe to pandas dataframe."""
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(df)

def pd2df(pd: pd.DataFrame) -> DataFrame:
    """Convert pandas dataframe to R dataframe."""
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(pd)
    
def set_seed(seed: int) -> None:
    ro.r["set.seed"](seed)

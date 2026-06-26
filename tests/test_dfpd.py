import pandas as pd
from omicstl.r_utils import df2pd, pd2df
from pandas.testing import assert_frame_equal

def test():
    demo_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["7", "8", "9"]})
    conv_df = pd2df(demo_pd)
    conv_dfpd = df2pd(conv_df)
    print(demo_pd)
    print(conv_dfpd)
    demo_pd = demo_pd.reset_index(drop=True)
    conv_dfpd = conv_dfpd.reset_index(drop=True)
    assert_frame_equal(demo_pd, conv_dfpd, check_index_type=False, check_dtype=False)
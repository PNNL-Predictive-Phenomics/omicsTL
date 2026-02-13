import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from omicstl.simulation_utils.model_utils import split_dataframe_indices

def test():
    np.random.seed(42)
    test_data = pd.DataFrame(np.random.randint(0, 100, (100, 50)))

    test_indices = test_data.index.tolist()

    np.random.seed(42)
    np.random.shuffle(test_indices)

    expect_left = test_indices[:90]
    expect_right = test_indices[90:]

    actual_left, actual_right = split_dataframe_indices(test_data, 0.9, 42)

    assert actual_left == expect_left
    assert actual_right == expect_right

    




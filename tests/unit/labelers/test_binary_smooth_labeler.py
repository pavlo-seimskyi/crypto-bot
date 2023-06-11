import numpy as np
import pandas as pd
import pytest

from src.labelers import BinarySmoothLabeler


@pytest.mark.unit
def test_binary_labeler():
    labeler = BinarySmoothLabeler(price_col="price", period=3)

    data = pd.DataFrame(
        {
            "price": [
                1,  # next avg. 3.00 -> 1
                2,  # next avg. 2.66 -> 1
                5,  # next avg. 1.66 -> 0
                2,  # next avg. 1.33 -> 0
                1,  # nan
                2,  # nan
                1,  # nan
            ]
        }
    )

    expected = [1.0, 1.0, 0.0, 0.0, np.nan, np.nan, np.nan]

    output = labeler.transform(data)

    assert output == expected

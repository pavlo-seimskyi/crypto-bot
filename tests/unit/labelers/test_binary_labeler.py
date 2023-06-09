import numpy as np
import pandas as pd
import pytest

from src.labelers import BinaryLabeler


@pytest.mark.unit
def test_binary_labeler():
    labeler = BinaryLabeler(price_col="price", period=3)

    data = pd.DataFrame(
        {
            "price": [
                1,
                2,
                3,
                4,  # above 1
                5,  # above 2
                1,  # below 3
                4,  # equal to 4
            ]
        }
    )

    expected = [1.0, 1.0, 0.0, 0.0, np.nan, np.nan, np.nan]

    output = labeler.transform(data)

    assert output == expected

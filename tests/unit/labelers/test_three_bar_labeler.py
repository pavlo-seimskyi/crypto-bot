import numpy as np
import pandas as pd
import pytest

from src.labelers import ThreeBarLabeler


@pytest.mark.unit
def test_three_bar_labeler():
    labeler = ThreeBarLabeler(
        price_col="price", period=3, threshold_pct=10.0  # 3 steps ahead  # 10%
    )

    data = pd.DataFrame(
        {
            "price": [
                1.0,
                2.0,
                4.0,
                4.0,  # +300% (label 2)
                5.0,  # +150% (label 2)
                1.0,  # -75% (label 0)
                4.25,  # +6.25% (label 1)
            ]
        }
    )

    expected = [2.0, 2.0, 0.0, 1.0, np.nan, np.nan, np.nan]

    output = labeler.transform(data)

    assert output == expected

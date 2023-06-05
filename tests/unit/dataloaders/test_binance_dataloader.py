import pandas as pd
import pytest
from binance import enums

from clients import BinanceClient
from src.dataloaders import BinanceDataLoader


@pytest.fixture
def binance_dataloader():
    return BinanceDataLoader(
        interval=enums.KLINE_INTERVAL_1MINUTE,
        assets=["BTC", "ETH"],
        fiat="USDT",
        exchange_client=BinanceClient(api_key=None, api_secret=None),
        datetime_fmt="%Y-%m-%d %H:%M:%S",
    )


@pytest.fixture
def raw_input_data():
    return [
        [
            1614729660000,
            "48306.80000000",
            "48409.33000000",
            "48304.61000000",
            "48341.15000000",
            "47.55695200",
            1614729719999,
            "2299649.94896855",
            1207,
            "22.07072500",
            "1067139.41449928",
            "0",
        ],
        [
            1614729720000,
            "48341.14000000",
            "48409.39000000",
            "48306.81000000",
            "48325.84000000",
            "77.64910500",
            1614729779999,
            "3754492.14707568",
            1184,
            "37.20116100",
            "1799172.95863799",
            "0",
        ],
    ]


@pytest.mark.unit
def test_read_raw_input_data(binance_dataloader, raw_input_data):
    data = binance_dataloader.read_raw_input_data(raw_input_data)
    assert type(data) == pd.DataFrame
    assert not data.empty
    assert data.shape == (2, 11)


@pytest.mark.unit
def test_pivot_price_data(binance_dataloader, monkeypatch):
    # Patch dtypes
    dtypes = {
        "open_timestamp": int,
        "open": float,
        "close": float,
    }
    monkeypatch.setattr(binance_dataloader.exchange_client, "dtypes", dtypes)
    input_dict = {
        "open_timestamp": [1614729660000, 1614729720000],
        "open": [48306.8, 48341.14],
        "close": [48341.15, 48325.84],
        "symbol": ["BTC", "BTC"],
    }
    input_data = pd.DataFrame(input_dict)
    output = binance_dataloader.pivot_price_data(input_data)
    # Symbol column is removed after pivot
    _ = input_dict.pop("symbol")
    assert output.T.values.tolist() == list(input_dict.values())
    assert output.columns.tolist() == ["open_timestamp", "BTC_open", "BTC_close"]


@pytest.mark.unit
def test_process_missing_intervals(binance_dataloader):
    input_data = pd.DataFrame(
        {
            "open_timestamp": [1614729660000, 1614729840000],
            "BTC_open": [48306.8, 48281.94],
            "BTC_close": [48341.15, 48240.63],
        }
    )

    output = binance_dataloader.process_missing_intervals(input_data)

    expected_output = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2021-03-03 00:01:00"),
                pd.Timestamp("2021-03-03 00:02:00"),  # Extended
                pd.Timestamp("2021-03-03 00:03:00"),  # Extended
                pd.Timestamp("2021-03-03 00:04:00"),
            ],
            "open_timestamp": [
                1614729660000.0,
                1614729660000.0,  # Extended
                1614729660000.0,  # Extended
                1614729840000.0,
            ],
            "BTC_open": [48306.8, 48306.8, 48306.8, 48281.94],
            "BTC_close": [48341.15, 48341.15, 48341.15, 48240.63],
            "service_down": [False, True, True, False],
        }
    )

    pd.testing.assert_frame_equal(output, expected_output)


@pytest.mark.unit
def test_validate_invalid_symbols(binance_dataloader, monkeypatch):
    with pytest.raises(AssertionError):
        monkeypatch.setattr(binance_dataloader, "assets", [])
        binance_dataloader.validate()


@pytest.mark.unit
def test_validate_invalid_interval(binance_dataloader, monkeypatch):
    with pytest.raises(AssertionError):
        monkeypatch.setattr(binance_dataloader, "interval", "1-minutito")
        binance_dataloader.validate()

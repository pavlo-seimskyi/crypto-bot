import pathlib
from binance import enums

# Paths
BASE_PATH = str(pathlib.Path(__file__).parent)

# Assets
ASSET_TO_TRADE = "ETH"
PREDICTOR_ASSETS = ["BTC", "LTC", "XRP", "ADA", "BNB", "XRP"]
FIAT_TO_TRADE = "USDT"

#
BINANCE_INTERVAL = enums.KLINE_INTERVAL_1MINUTE

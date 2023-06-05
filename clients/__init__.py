import os

from dotenv import load_dotenv

from clients.exchange.binance import BinanceClient
from constants import BASE_PATH

load_dotenv(os.path.join(BASE_PATH, ".env"))

ENV = os.environ.copy()

__all__ = ["ENV", "BinanceClient"]

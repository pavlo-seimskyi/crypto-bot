# Crypto bot
Let's achieve financial freedom.

# Project parts
- [ ] Clients for APIs.
- [ ] Data service. Uses different DataLoaders to load each of the features.
- [ ] ML model. Predicts price changes.
    - Backtest & continuous evaluation just for the model.
- [ ] Portfolio manager. Tracks assets, ROI, commissions.
- [ ] Order executer. Interprets the predictions, taking into account the portfolio situation, to place buy/hold/sell.
    - Loss-preventive logic. Automated checks to avoid ruin.
- [ ] Strategy Evaluator. To backtest after each implemented change.
    - Compares the strategy to baselines:
        - Moving Average Crossover
        - Holding BTC
        - Not investing, keeping cash
- [ ] Logging & monitoring.
    - Continuously evaluate recent performance in prod
    - Notification system
- [ ] Error handling.
    - Killswitch
    - Downtime management
- [ ] (optional) User interface


# Plan
- [ ] R&D: Prove that ML can be consistently 10% better than naive baseline. Prototypes for:
    - Binance API
    - Data service
    - ML model
    - Model backtester
- [ ] Rest.

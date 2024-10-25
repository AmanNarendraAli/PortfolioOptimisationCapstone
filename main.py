import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from Model import Model  # Your existing Model class
import schedule
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas_market_calendars as mcal


class MarketDataHandler:
    def __init__(self, tickers, window_size=51):
        self.tickers = tickers
        self.window_size = window_size
        self.data = pd.DataFrame()
        self.rolling_window = None

    def fetch_historical_data(self, start_date, end_date):
        """Fetches historical data for all tickers with proper adjustment"""
        data_frames = []
        for ticker in self.tickers:
            try:
                ticker_data = yf.download(ticker, start=start_date, end=end_date)
                data_frames.append(ticker_data["Adj Close"].rename(ticker))
            except Exception as e:
                logging.error(f"Error fetching data for {ticker}: {e}")
                raise

        self.data = pd.concat(data_frames, axis=1)
        self.data.fillna(method="ffill", inplace=True)
        self.data.dropna(inplace=True)  # Make sure no missing data remains

        self.data = self.data.iloc[::-1]
        # Ensure index is timezone-naive
        self.data.index = self.data.index.tz_localize(None)
        return self.data

    def update_rolling_window(self, current_date):
        """Updates rolling window of data up to current_date"""
        try:
            # Get data up to and including current_date
            mask = self.data.index <= current_date
            recent_data = self.data[mask].tail(self.window_size)
            
            if len(recent_data) == self.window_size:
                self.rolling_window = recent_data
                return True
            return False
        except Exception as e:
            logging.error(f"Error in update_rolling_window: {e}")
            return False


class PortfolioManager:
    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}
        self.portfolio_values = pd.Series()  # Track daily portfolio values
        self.transaction_history = []

    def initialize_tickers(self, tickers):
        """Initialize position tracking for given tickers"""
        self.positions = {ticker: 0 for ticker in tickers}

    def rebalance_portfolio(self, allocations, current_prices, date):
        """Executes portfolio rebalancing with proper value calculation"""
        # Calculate current portfolio value
        self.portfolio_value = self.calculate_portfolio_value(current_prices)

        # If this is the first trade, use initial capital
        if self.portfolio_value == 0:
            self.portfolio_value = self.initial_capital

        # Calculate target position values
        target_values = {
            ticker: self.portfolio_value * alloc
            for ticker, alloc in zip(self.positions.keys(), allocations)
        }

        # Record portfolio value
        self.portfolio_values[date] = self.portfolio_value

        # Execute trades
        for ticker in self.positions.keys():
            current_value = self.positions[ticker] * current_prices[ticker]
            target_value = target_values[ticker]

            # Calculate shares to trade (avoid division by zero)
            if current_prices[ticker] > 0:
                shares_to_trade = (target_value - current_value) / current_prices[
                    ticker
                ]
            else:
                logging.warning(f"Zero price detected for {ticker}")
                continue

            # Execute trade if significant
            if abs(shares_to_trade) > 1e-10:
                self.positions[ticker] += shares_to_trade
                self.transaction_history.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "shares": shares_to_trade,
                        "price": current_prices[ticker],
                        "value": shares_to_trade * current_prices[ticker],
                        "allocation": target_values[ticker] / self.portfolio_value,
                    }
                )

    def update_daily_portfolio_value(self, current_prices, date):
        portfolio_value = self.calculate_portfolio_value(current_prices)
        self.portfolio_values[date] = portfolio_value

    def calculate_portfolio_value(self, current_prices):
        return sum(
            self.positions[ticker] * current_prices[ticker] for ticker in self.positions
        )

    def get_portfolio_stats(self):
        """Calculate portfolio performance statistics"""
        if len(self.portfolio_values) == 0:
            return {"total_return": 0, "sharpe_ratio": 0, "num_trades": 0}

        # Calculate returns and metrics
        returns = self.portfolio_values.pct_change().dropna()
        total_return = (self.portfolio_values.iloc[-1] / self.initial_capital - 1) * 100

        stats = {
            "total_return": total_return,
            "sharpe_ratio": (
                (returns.mean() / returns.std()) * np.sqrt(252)
                if len(returns) > 1
                else 0
            ),
            "num_trades": len(self.transaction_history),
            "final_portfolio_value": self.portfolio_values.iloc[-1],
        }
        return stats


class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.market_data = MarketDataHandler(config["tickers"], config["window_size"])
        self.portfolio = PortfolioManager(config["initial_capital"])
        self.model = Model()
        
        # Get NYSE calendar
        nyse = mcal.get_calendar("NYSE")
        self.trading_days = nyse.schedule(
            start_date=config["start_date"],
            end_date=config["end_date"]
        )
        
        # Convert to list of dates
        self.trading_days = pd.to_datetime(self.trading_days.index).tz_localize(None)
        
        # Set initial date as first trading day
        self.current_date = self.trading_days[0]
        self.end_date = self.trading_days[-1]

        # Initialize portfolio with tickers
        self.portfolio.initialize_tickers(config["tickers"])
        
        self.training_data = []
        logging.basicConfig(
            level=logging.INFO, 
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def run_backtest(self):
        """Main backtesting loop"""
        try:
            # Fetch all historical data first
            self.market_data.fetch_historical_data(
                self.config["start_date"], 
                self.config["end_date"]
            )

            # Iterate through trading days only
            for trade_date in self.trading_days:
                self.current_date = trade_date
                
                # Update rolling window
                if self.market_data.update_rolling_window(self.current_date):
                    # Update daily portfolio value
                    try:
                        current_prices = self.market_data.data.loc[self.current_date]
                        self.portfolio.update_daily_portfolio_value(
                            current_prices, 
                            self.current_date
                        )

                        # Rebalance on first trading day of each month
                        if trade_date == self.trading_days[0] or (
                            trade_date.month != self.trading_days[
                                self.trading_days.get_loc(trade_date) - 1
                            ].month
                        ):
                            self._execute_rebalance()
                    except KeyError as e:
                        logging.warning(f"No data available for date {self.current_date}")
                        continue

            return self.portfolio.get_portfolio_stats()
            
        except Exception as e:
            logging.error(f"Error in run_backtest: {e}")
            raise

    def _plot_asset_prices(self):
        """Plot individual asset prices"""
        n_assets = len(self.config["tickers"])
        n_cols = 2
        n_rows = (n_assets + 1) // 2

        fig = plt.figure(figsize=(15, 5 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig)

        data = self.market_data.data
        for idx, ticker in enumerate(self.config["tickers"]):
            ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
            data[ticker].plot(ax=ax)
            ax.set_title(f"{ticker} Price History")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(True)

        plt.tight_layout()
        return fig

    def _plot_portfolio_value(self):
        """Plot portfolio value over time"""
        if len(self.portfolio.portfolio_values) == 0:
            return None

        fig, ax = plt.subplots(figsize=(15, 7))
        self.portfolio.portfolio_values.plot(ax=ax)
        ax.set_title("Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True)

        # Add annotations
        final_value = self.portfolio.portfolio_values.iloc[-1]
        total_return = (final_value / self.portfolio.initial_capital - 1) * 100
        ax.text(
            0.02,
            0.98,
            f"Total Return: {total_return:.2f}%\n"
            f"Initial Value: ${self.portfolio.initial_capital:,.2f}\n"
            f"Final Value: ${final_value:,.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        return fig

    def _execute_rebalance(self):
        try:
            data = self.market_data.rolling_window

            if data is None or not isinstance(data, pd.DataFrame):
                logging.warning("No valid rolling window data available for rebalancing")
                return

            # Get model allocations
            allocations = self.model.get_allocations(data)

            # Get current prices
            current_prices = data.iloc[-1]

            # Execute rebalance
            self.portfolio.rebalance_portfolio(
                allocations, 
                current_prices, 
                self.current_date
            )

            logging.info(f"Rebalance executed on {self.current_date}")
            logging.info(
                f"New allocations: {dict(zip(self.config['tickers'], allocations))}"
            )
            
        except Exception as e:
            logging.error(f"Error during rebalancing: {e}")

    def generate_reports(self):
        """Generate backtest reports with improved error handling"""
        try:
            # Get the portfolio stats (e.g., total return, Sharpe ratio)
            stats = self.portfolio.get_portfolio_stats()

            # Access the transaction history from the portfolio manager
            trades_df = (
                pd.DataFrame(self.portfolio.transaction_history)
                if self.portfolio.transaction_history
                else pd.DataFrame()
            )

            # Create visualizations
            fig_assets = self._plot_asset_prices()
            fig_portfolio = self._plot_portfolio_value()
            fig_allocations = self._plot_allocation_history()

            return {
                "stats": stats,
                "trades": trades_df,
                "final_positions": self.portfolio.positions,
                "figures": {
                    "assets": fig_assets,
                    "portfolio": fig_portfolio,
                    "allocations": fig_allocations,
                },
            }
        except Exception as e:
            logging.error(f"Error generating reports: {e}")
            raise

    def _plot_allocation_history(self):
        """Plot allocation history over time"""
        if not self.portfolio.transaction_history:
            return None

        df = pd.DataFrame(self.portfolio.transaction_history)
        pivot_df = df.pivot(index="date", columns="ticker", values="allocation")

        fig, ax = plt.subplots(figsize=(15, 7))
        pivot_df.plot(kind="area", stacked=True, ax=ax)
        ax.set_title("Portfolio Allocation Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.grid(True)
        ax.legend(title="Assets")

        return fig


def main():
    # Configuration
    config = {
        "tickers": ["VTI", "AGG", "DBC", "VIXY"],
        "start_date": "2015-10-01",
        "end_date": "2020-10-01",
        "initial_capital": 100000,
        "window_size": 51,  # Match QuantConnect's default window size
    }

    # Set up logging with more detail
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize and run backtest
    backtest = BacktestEngine(config)

    try:
        # Run backtest
        stats = backtest.run_backtest()

        # Generate reports
        reports = backtest.generate_reports()

        print("\nBacktest Results:")
        print(f"Total Return: {stats['total_return']:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"Number of Trades: {stats['num_trades']}")
        print(f"Final Portfolio Value: ${stats['final_portfolio_value']:,.2f}")

        print("\nFinal Positions:")
        for ticker, shares in reports["final_positions"].items():
            print(f"{ticker}: {shares:.2f} shares")

        # Show plots
        plt.show()

    except Exception as e:
        logging.error(f"Error during backtest: {str(e)}")
        raise


if __name__ == "__main__":
    main()

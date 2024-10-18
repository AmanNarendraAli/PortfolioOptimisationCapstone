import yfinance as yf
import pandas as pd
import numpy as np
import logging
from Model import Model  # Import your model class
import matplotlib.pyplot as plt
import math
# Initialize logging
logging.basicConfig(level=logging.INFO)

class OptimizedUncoupledAutosequencers:
    def __init__(self, start_date, end_date, tickers, initial_cash=100000, n_periods=51):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.initial_cash = initial_cash
        self.n_periods = n_periods  # Look-back period for historical data
        self.data = None  # Placeholder for historical data
        self.model = Model()  # Model instance
        self.portfolio = {ticker: 0 for ticker in tickers}  # Initialize portfolio holdings
        self.portfolio['total_value'] = initial_cash  # Start with initial cash
        self.historical_data = {}  # Store fetched data for analysis

    def plot_data(self):
        '''
        Plot the historical closing prices for each ticker in its own subplot
        '''
        if self.historical_data is None or self.historical_data.empty:
            print("No data to plot")
            return
        
        num_tickers = len(self.tickers)
        cols = 2
        rows = math.ceil(num_tickers / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        axes = axes.flatten()

        for i, ticker in enumerate(self.tickers):
            print(f"Plotting {ticker} on subplot {i}")
            print(self.historical_data[ticker].head())  # Print first few rows of the data
            axes[i].plot(self.historical_data.index, self.historical_data[ticker])
            axes[i].set_title(f"Price of {ticker}")
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel("Normalized Price")
            axes[i].grid(True)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


    def load_data(self):
        '''
        Fetch historical data for all tickers and store it in self.data
        '''
        logging.info("Loading historical data for tickers: {}".format(self.tickers))
        self.historical_data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        self.historical_data.dropna(inplace=True)  # Drop any missing values
        #self.historical_data = self.normalize_data(self.historical_data)
        logging.info("Data loaded successfully")

    def normalize_data(self, data):
        return (data - data.mean()) / data.std()

    def rebalance(self):
        '''
        Rebalance the portfolio based on model allocations
        '''
        # Check if enough data has been accumulated to rebalance
        if len(self.historical_data) < self.n_periods:
            logging.info("Not enough data for rebalance")
            return

        # Get recent data (last n_periods)
        recent_data = self.historical_data.tail(self.n_periods)

        # Get allocations from the model
        allocations = self.model.get_allocations(recent_data)

        logging.info(f'Portfolio Allocations: {allocations}')

        # Update the portfolio based on the new allocations
        self.update_portfolio(allocations)

    def update_portfolio(self, allocations):
        '''
        Updates the portfolio with new allocations and calculates portfolio value
        '''
        for ticker, allocation in zip(self.tickers, allocations):
            self.portfolio[ticker] = allocation * self.portfolio['total_value'] / self.historical_data[ticker].iloc[-1]

        # Update the total portfolio value
        self.portfolio['total_value'] = sum(
            self.portfolio[ticker] * self.historical_data[ticker].iloc[-1] for ticker in self.tickers
        )
        
        logging.info(f"Updated Portfolio: {self.portfolio}")

    def backtest(self):
        '''
        Main backtesting loop that runs over the historical data
        '''
        logging.info("Starting backtest...")

        prev_week = None
        for current_date, _ in self.historical_data.iterrows():
            # Rebalance at the start of each week (ISO week 1 = Monday)
            current_week = current_date.isocalendar()[1]  # Extract week number

            if current_week != prev_week:
                logging.info(f"Rebalancing on {current_date} (Week {current_week})")
                self.rebalance()
                logging.info(f"Portfolio value on {current_date}: {self.portfolio['total_value']}")
            
            prev_week = current_week  # Update the previous week for comparison

        logging.info(f"Final portfolio value: {self.portfolio['total_value']}")
        logging.info("Backtest completed successfully")


if __name__ == "__main__":
    # Setup tickers, start and end date for backtesting
    tickers = ['GOOG','SPY']
    start_date = '2023-09-01'
    end_date = '2024-09-01'
    initial_cash = 100000

    # Initialize the strategy
    strategy = OptimizedUncoupledAutosequencers(start_date, end_date, tickers, initial_cash)

    # Load historical data
    strategy.load_data()
    # Plot the data
    strategy.plot_data()
    # Run the backtest
    strategy.backtest()

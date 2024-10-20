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
                # Explicitly request adjusted close prices
                ticker_data = yf.download(ticker, start=start_date, end=end_date)
                data_frames.append(ticker_data['Adj Close'].rename(ticker))
            except Exception as e:
                logging.error(f"Error fetching data for {ticker}: {e}")
                raise
        
        self.data = pd.concat(data_frames, axis=1)
        self.data.fillna(method='ffill', inplace=True)
        return self.data
    
    def update_rolling_window(self, current_date):
        """Updates rolling window of data up to current_date"""
        mask = self.data.index <= current_date
        recent_data = self.data[mask].tail(self.window_size)
        if len(recent_data) == self.window_size:
            self.rolling_window = recent_data
            return True
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
        target_values = {ticker: self.portfolio_value * alloc 
                        for ticker, alloc in zip(self.positions.keys(), allocations)}
        
        # Record portfolio value
        self.portfolio_values[date] = self.portfolio_value
        
        # Execute trades
        for ticker in self.positions.keys():
            current_value = self.positions[ticker] * current_prices[ticker]
            target_value = target_values[ticker]
            
            # Calculate shares to trade (avoid division by zero)
            if current_prices[ticker] > 0:
                shares_to_trade = (target_value - current_value) / current_prices[ticker]
            else:
                logging.warning(f"Zero price detected for {ticker}")
                continue
            
            # Execute trade if significant
            if abs(shares_to_trade) > 1e-6:
                self.positions[ticker] += shares_to_trade
                self.transaction_history.append({
                    'date': date,
                    'ticker': ticker,
                    'shares': shares_to_trade,
                    'price': current_prices[ticker],
                    'value': shares_to_trade * current_prices[ticker],
                    'allocation': target_values[ticker] / self.portfolio_value
                })

    def calculate_portfolio_value(self, current_prices):
        """Calculates current portfolio value"""
        return sum(shares * current_prices[ticker] 
                  for ticker, shares in self.positions.items())
    
    def get_portfolio_stats(self):
        """Calculate portfolio performance statistics"""
        if len(self.portfolio_values) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'num_trades': 0
            }
        
        # Calculate returns and metrics
        returns = self.portfolio_values.pct_change().dropna()
        total_return = (self.portfolio_values.iloc[-1] / self.initial_capital - 1) * 100
        
        stats = {
            'total_return': total_return,
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 else 0,
            'num_trades': len(self.transaction_history),
            'final_portfolio_value': self.portfolio_values.iloc[-1]
        }
        return stats


class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.market_data = MarketDataHandler(config['tickers'], config['window_size'])
        self.portfolio = PortfolioManager(config['initial_capital'])
        self.model = Model()
        self.current_date = pd.Timestamp(config['start_date'])
        self.end_date = pd.Timestamp(config['end_date'])
        
        # Initialize portfolio with tickers
        self.portfolio.initialize_tickers(config['tickers'])
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
    
    def run_backtest(self):
        """Main backtesting loop"""
        # Fetch all historical data first
        self.market_data.fetch_historical_data(
            self.config['start_date'], 
            self.config['end_date']
        )
        
        while self.current_date <= self.end_date:
            # Update rolling window
            if self.market_data.update_rolling_window(self.current_date):
                # Check if it's time to rebalance (weekly)
                if self.current_date.weekday() == 0:  # Monday
                    self._execute_rebalance()
            
            self.current_date += timedelta(days=1)
        
        return self.portfolio.get_portfolio_stats()
    
    def _plot_asset_prices(self):
        """Plot individual asset prices"""
        n_assets = len(self.config['tickers'])
        n_cols = 2
        n_rows = (n_assets + 1) // 2
        
        fig = plt.figure(figsize=(15, 5*n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        data = self.market_data.data
        for idx, ticker in enumerate(self.config['tickers']):
            ax = fig.add_subplot(gs[idx//n_cols, idx%n_cols])
            data[ticker].plot(ax=ax)
            ax.set_title(f'{ticker} Price History')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def _plot_portfolio_value(self):
        """Plot portfolio value over time"""
        if len(self.portfolio.portfolio_values) == 0:
            return None
            
        fig, ax = plt.subplots(figsize=(15, 7))
        self.portfolio.portfolio_values.plot(ax=ax)
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True)
        
        # Add annotations
        final_value = self.portfolio.portfolio_values.iloc[-1]
        total_return = (final_value / self.portfolio.initial_capital - 1) * 100
        ax.text(0.02, 0.98, 
                f'Total Return: {total_return:.2f}%\n'
                f'Initial Value: ${self.portfolio.initial_capital:,.2f}\n'
                f'Final Value: ${final_value:,.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return fig
    
    def _execute_rebalance(self):
        """Execute rebalancing logic with necessary checks and improvements"""
        try:
            # Get current window of data
            data = self.market_data.rolling_window
            
            # Check if data is valid
            if data is None or len(data) < self.config['window_size']:
                logging.warning(f"Insufficient data points: {len(data) if data is not None else 0}")
                return
                
            # Calculate returns (excluding the first row which will be NaN)
            returns = data.pct_change().dropna()  # Drop any NaNs
            logging.info(f"Returns data cleaned. Shape: {returns.shape}")
            
            # Log returns for debugging purposes
            logging.info(f"Returns before combining:\n{returns}")
            if returns.isna().any().any():
                logging.error("NaN detected in returns.")
                return
            if np.isinf(returns.values).any():
                logging.error("Infinity detected in returns.")
                return
            
            # Get prices (excluding the first row to match returns)
            prices = data.iloc[1:]  # Align prices with returns
            
            # Concatenate prices and returns horizontally
            combined_data = pd.concat([prices, returns], axis=1)

            # Log combined data to verify correctness
            logging.info(f"Combined data being fed to model:\n{combined_data.head()}")
            
            # Get model allocations
            allocations = self.model.get_allocations(combined_data)
            logging.info(f"Allocations from model: {allocations}")
            
            # Check for NaNs in allocations
            if np.any(np.isnan(allocations)):
                logging.error(f"NaN allocations detected: {allocations}")
                return
            
            # Normalize allocations if they don't sum to 1 (with additional safeguards)
            allocation_sum = sum(allocations)
            if abs(allocation_sum - 1) > 0.01:
                if allocation_sum > 1e-6:
                    logging.warning(f"Allocations sum to {allocation_sum}, normalizing")
                    allocations = allocations / allocation_sum
                else:
                    logging.error(f"Allocations sum too small to normalize: {allocation_sum}. Returning zeros.")
                    allocations = np.zeros_like(allocations)  # Return zeros if normalization isn't possible
            
            # Get current prices for rebalancing
            current_prices = data.iloc[-1]

            # Check if current_prices are valid
            if current_prices.isna().any():
                logging.error("NaN detected in current_prices.")
                return
            
            # Execute rebalance
            logging.info(f"Executing rebalance on {self.current_date} with allocations: {allocations}")
            self.portfolio.rebalance_portfolio(allocations, current_prices, self.current_date)
            
            logging.info(f"Rebalance executed on {self.current_date}")
            logging.info(f"New allocations: {dict(zip(self.config['tickers'], allocations))}")
            logging.info(f"Current portfolio value: ${self.portfolio.calculate_portfolio_value(current_prices):,.2f}")
            
        except Exception as e:
            logging.error(f"Error during rebalancing: {str(e)}")
            raise



    def generate_reports(self):
        """Generate backtest reports with improved error handling"""
        try:
            stats = self.portfolio.get_portfolio_stats()
            trades_df = pd.DataFrame(self.transaction_history) if self.portfolio.transaction_history else pd.DataFrame()
            
            # Create visualizations
            fig_assets = self._plot_asset_prices()
            fig_portfolio = self._plot_portfolio_value()
            fig_allocations = self._plot_allocation_history()
            
            return {
                'stats': stats,
                'trades': trades_df,
                'final_positions': self.portfolio.positions,
                'figures': {
                    'assets': fig_assets,
                    'portfolio': fig_portfolio,
                    'allocations': fig_allocations
                }
            }
        except Exception as e:
            logging.error(f"Error generating reports: {e}")
            raise

    def _plot_allocation_history(self):
        """Plot allocation history over time"""
        if not self.portfolio.transaction_history:
            return None
            
        df = pd.DataFrame(self.portfolio.transaction_history)
        pivot_df = df.pivot(index='date', columns='ticker', values='allocation')
        
        fig, ax = plt.subplots(figsize=(15, 7))
        pivot_df.plot(kind='area', stacked=True, ax=ax)
        ax.set_title('Portfolio Allocation Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Allocation')
        ax.grid(True)
        ax.legend(title='Assets')
        
        return fig

def main():
    # Configuration
    config = {
        'tickers': ['VTI', 'AGG', 'DBC', 'VIXY'],
        'start_date': '2015-10-01',
        'end_date': '2016-10-01',
        'initial_capital': 100000,
        'window_size': 51  # Match QuantConnect's default window size
    }
    
    # Set up logging with more detail
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
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
        for ticker, shares in reports['final_positions'].items():
            print(f"{ticker}: {shares:.2f} shares")
        
        # Show plots
        plt.show()
            
    except Exception as e:
        logging.error(f"Error during backtest: {str(e)}")
        raise

if __name__ == '__main__':
    main()
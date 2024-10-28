# deep-learning-portfolio-optimization

[Paper Source](https://arxiv.org/pdf/2005.13665.pdf)
  

[QuantConnect Code](https://www.quantconnect.com/terminal/processCache/?request=embedded_backtest_4ebbe01bfea8c5ae6f98fcda38a50b1c.html)  
  
# LSTM-Based Portfolio Optimization

This project uses an LSTM model to dynamically optimize asset allocations in a long-only portfolio, rebalancing every two years to adapt to market patterns. The objective is to maximize risk-adjusted returns by leveraging deep learning and financial data.

## Table of Contents
- [Project Objective](#project-objective)
- [Data Preprocessing](#data-preprocessing)
- [Portfolio Handling and Rebalancing Setup](#portfolio-handling-and-rebalancing-setup)
- [Model Training and Rebalancing](#model-training-and-rebalancing)
- [Holding Period Performance Evaluation](#holding-period-performance-evaluation)
- [Performance Metrics Calculation](#performance-metrics-calculation)
- [Visualization and Comparison](#visualization-and-comparison)

---

## Project Objective
The goal of this project is to create an optimized portfolio that maximizes risk-adjusted returns. Using an LSTM model, portfolio weights are adjusted every two years based on cumulative historical data. The project evaluates portfolio performance over time, tracking cumulative returns and comparing results with baseline strategies.

---

## Data Preprocessing
**Objective**: Prepare historical price data for input to the LSTM model.
  
### Steps:
1. **Fetch Historical Data**: Retrieve adjusted close prices for selected assets.
2. **Calculate Daily Returns**: Compute percentage changes in price between consecutive days.
3. **Normalize Data**: Normalize prices so each asset’s time series starts at 1, ensuring consistency in return calculations.

---

## Portfolio Handling and Rebalancing Setup
**Objective**: Create a framework to manage portfolio weights, calculate returns, and track portfolio value over time.

### Steps:
1. **Initialize Portfolio**: Start with a specified cash value.
2. **Apply Rebalanced Weights**: At the end of each training period, apply optimized weights from the model.
3. **Compound Returns Daily**: During each holding period, apply fixed weights to each asset’s daily returns, updating the portfolio value based on compounded returns.

This framework enables tracking of cumulative returns and portfolio growth without a separate testing period.

---

## Model Training and Rebalancing
**Objective**: Optimize portfolio weights at the end of each training period (every two years).

### Steps:
1. **Initialize Model**: Load the LSTM model with data from the start up to each rebalancing point.
2. **Train the Model**: At the end of each two-year interval, fit the model to determine optimal weights.
3. **Save Weights**: Store optimized weights to use during the subsequent holding period until the next rebalancing.

---

## Holding Period Performance Evaluation
**Objective**: Track portfolio performance between rebalancing points by applying optimized weights.

### Steps:
1. **Daily Portfolio Returns**: For each day in the holding period, calculate portfolio returns by applying the last set of optimized weights to each asset’s daily returns.
2. **Update Portfolio Value**: Compound portfolio returns daily to reflect growth or decline based on fixed weights.

This simulates real-time portfolio performance during holding periods between rebalancing.

---

## Performance Metrics Calculation
**Objective**: Measure the risk-adjusted performance of the portfolio over the entire evaluation period.

### Steps:
1. **Cumulative Portfolio Value**: Calculate and track cumulative portfolio value across all rebalancing and holding periods.
2. **Risk-Adjusted Metrics**: Compute key performance metrics, including:
   - **Sharpe Ratio**: Measures risk-adjusted returns.
   - **Sortino Ratio**: Focuses on downside risk-adjusted returns.
   - **Maximum Drawdown**: Assesses the largest portfolio drop from peak to trough.

---

## Visualization and Comparison
**Objective**: Visualize portfolio growth and compare the LSTM-optimized strategy with baseline methods.

### Steps:
1. **Cumulative Portfolio Growth**: Plot cumulative portfolio value to observe growth over time and through each rebalancing period.
2. **Baseline Comparison**: Optionally, implement baseline strategies (e.g., equal-weighted or mean-variance optimized portfolios) to track and compare alongside the LSTM-based portfolio.
3. **Performance Comparison**: Use risk-adjusted performance metrics to evaluate the LSTM approach relative to traditional strategies.

---


"""
Created on Sun Apr 28 16:23:15 2024

@author: Ryan Milgrim, CFA
"""
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_data(file_path='data.csv'):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df = df.set_index(['Date', 'Crypto']).unstack()
    return df

def calculate_returns(data):
    return data.Close.pct_change(fill_method=None).dropna()

def minimize_risk(returns, target_return=None):
    mean_returns = returns.mean().to_numpy()
    cov_matrix = returns.cov().to_numpy()
    n_assets = len(mean_returns)
    weights = cp.Variable(n_assets)
    
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    constraints = [cp.sum(weights) == 1, weights >= 0]
    
    if target_return is not None:
        expected_return = mean_returns @ weights
        constraints.append(expected_return >= target_return)
    
    objective = cp.Minimize(portfolio_variance)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return pd.Series(weights.value, returns.columns)

def optimal_portfolios(returns, num_portfolios=100):

    min_risk_portfolio = minimize_risk(returns)
    min_return = simple_backtest(returns, min_risk_portfolio)[2]
    max_return = returns.mean().max()
    
    portfolios = dict()
    target_returns = np.linspace(min_return, max_return, num_portfolios)   
    for r in target_returns:
        weights = minimize_risk(returns, target_return=r)
        stats = simple_backtest(returns, weights)
        stats = (stats[2], stats[3], stats[2] / stats[3])
        portfolios[stats] = weights

    portfolios = pd.DataFrame(portfolios).transpose()
    portfolios.index.names = ['Return', 'Risk', 'Sharpe Ratio']
    return portfolios.reset_index()

def simple_backtest(returns, portfolio):
    portfolio_returns = returns @ portfolio
    portfolio_price = (1 + portfolio_returns).cumprod()
    portfolio_mean = portfolio_returns.mean()
    portfolio_risk = portfolio_returns.std()
    return portfolio_returns, portfolio_price, portfolio_mean, portfolio_risk

def plot_efficient_frontier(returns, portfolios):
    
    # Find the portfolio with the highest Sharpe Ratio
    max_sharpe_idx = portfolios['Sharpe Ratio'].idxmax()
    max_sharpe_portfolio = portfolios.iloc[max_sharpe_idx]
    
    # Plotting the efficient frontier
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        portfolios['Risk'],
        portfolios['Return'],
        c=portfolios['Sharpe Ratio']
    )
    plt.colorbar(scatter, label='Sharpe Ratio')
    ax.scatter(
        max_sharpe_portfolio['Risk'],
        max_sharpe_portfolio['Return'],
        color='red', marker='*',
        label='Maximum Sharpe Ratio'
    )

    # Format the axis to show percentage
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Risk (Volatility)')
    ax.set_ylabel('Return')
    ax.legend()
    plt.show()

    # Plotting the pie chart for the portfolio with the highest Sharpe ratio
    asset_weights = max_sharpe_portfolio.drop(['Return', 'Risk', 'Sharpe Ratio'])
    asset_weights = asset_weights[asset_weights > 0]  # filter out zero weights
    labels = asset_weights.index
    sizes = asset_weights.values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Benchmark Allocation')
    plt.axis('equal')    
    plt.show()

    # Plotting historical price performance of the benchmark portfolio
    returns, prices, _, _ = simple_backtest(
        returns, max_sharpe_portfolio.drop(['Return', 'Risk', 'Sharpe Ratio']))

    fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

    # Plotting returns
    axs[0].plot(returns.index, returns, label='Daily Returns', color='blue')
    axs[0].set_title('Historical Daily Returns')
    axs[0].set_ylabel('Returns')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()

    # Plotting prices
    axs[1].plot(prices.index, prices, label='Price', color='green')
    axs[1].set_title('Price Performance')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    return max_sharpe_portfolio


if __name__ == '__main__':
    data = load_data()
    returns = calculate_returns(data)
    portfolios = optimal_portfolios(returns)
    best_portfolio = plot_efficient_frontier(returns, portfolios)

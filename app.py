import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Funções auxiliares
def download_data(tickers):
    data = yf.download(tickers, period='5y')
    return data['Adj Close']

def calculate_annualized_returns(data):
    daily_returns = data.pct_change().dropna()
    return daily_returns.mean() * 252

def calculate_annualized_covariance_matrix(data):
    daily_returns = data.pct_change().dropna()
    return daily_returns.cov() * 252

def markowitz_optimization(returns, cov_matrix):
    num_assets = len(returns)
    args = (returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

# Streamlit Interface
st.title('BDR Portfolio Optimizer')

budget = st.number_input('Investment Budget', min_value=0, value=10000)
sector_filter = st.selectbox('Sector', options=['All', 'Tech', 'Finance', 'Healthcare'])

# Carregar dados dos ativos
ativos_df = pd.read_csv('https://raw.githubusercontent.com/richardrt13/bdrrecommendation/main/bdrs.csv')
tickers = ativos_df['Ticker'].tolist()
sectors = ativos_df.set_index('Ticker')['Sector'].to_dict()

data = download_data(tickers)
returns = calculate_annualized_returns(data)
cov_matrix = calculate_annualized_covariance_matrix(data)

filtered_tickers = tickers
if sector_filter != 'All':
    filtered_tickers = [ticker for ticker in tickers if sectors[ticker] == sector_filter]
    data = data[filtered_tickers]
    returns = returns[filtered_tickers]
    cov_matrix = calculate_annualized_covariance_matrix(data)

optimal_weights = markowitz_optimization(returns, cov_matrix)
portfolio = pd.DataFrame({
    'Ticker': filtered_tickers,
    'Weight': optimal_weights,
    'Investment': optimal_weights * budget
})

st.write('Optimized Portfolio Allocation')
st.dataframe(portfolio)

st.write('Portfolio Summary')
expected_return = np.sum(returns * optimal_weights)
portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
sharpe_ratio = (expected_return - 0.02) / portfolio_volatility

st.write(f"Total Investment: {portfolio['Investment'].sum()}")
st.write(f"Expected Annual Return: {expected_return:.2%}")
st.write(f"Portfolio Volatility: {portfolio_volatility:.2%}")
st.write(f"Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")

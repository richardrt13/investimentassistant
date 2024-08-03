import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize

# Funções auxiliares
@st.cache_data
def download_data(tickers):
    data = yf.download(tickers, period='5y')['Adj Close']
    return data

def calculate_annualized_returns(data):
    daily_returns = data.pct_change().dropna()
    return daily_returns.mean() * 252

def calculate_annualized_covariance_matrix(data):
    daily_returns = data.pct_change().dropna()
    return daily_returns.cov() * 252

def optimize_portfolio(returns, cov_matrix, risk_free_rate=0.02):
    num_assets = len(returns)
    def portfolio_performance(weights):
        port_return = np.sum(weights * returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return port_volatility, port_return
    
    def objective_function(weights):
        port_volatility, port_return = portfolio_performance(weights)
        return -((port_return - risk_free_rate) / port_volatility)  # Maximiza o Sharpe Ratio

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]
    result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x, portfolio_performance(result.x)

# Streamlit Interface
st.title('Investment Portfolio Recommender')

# Definições de entrada
tickers = st.text_input('Enter Tickers (comma-separated)', 'AAPL,MSFT,GOOGL').split(',')
budget = st.number_input('Investment Budget', min_value=1000, value=10000, step=1000)

# Carregar e processar dados
data = download_data(tickers)
returns = calculate_annualized_returns(data)
cov_matrix = calculate_annualized_covariance_matrix(data)

# Otimização da carteira
weights, (volatility, return_) = optimize_portfolio(returns, cov_matrix)

# Resultados
portfolio = pd.DataFrame({
    'Ticker': tickers,
    'Weight': weights,
    'Investment': weights * budget
})

st.subheader('Optimized Portfolio Allocation')
st.dataframe(portfolio.style.format({'Weight': '{:.2%}', 'Investment': '${:.2f}'}))

fig_allocation = px.pie(portfolio, names='Ticker', values='Investment', title='Portfolio Allocation')
st.plotly_chart(fig_allocation)

st.subheader('Portfolio Summary')
st.metric("Total Investment", f"${portfolio['Investment'].sum():.2f}")
st.metric("Expected Annual Return", f"{return_:.2%}")
st.metric("Portfolio Volatility", f"{volatility:.2%}")
st.metric("Portfolio Sharpe Ratio", f"{(return_ - 0.02) / volatility:.2f}")
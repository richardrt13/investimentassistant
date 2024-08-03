import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px

# Funções auxiliares
def download_data(tickers):
    try:
        data = yf.download(tickers, period='5y')
        return data['Adj Close']
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return None

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
    initial_guess = num_assets * [1. / num_assets,]
    result = minimize(negative_sharpe_ratio, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

# Streamlit Interface
st.title('BDR Portfolio Optimizer')

budget = st.number_input('Investment Budget', min_value=1, value=10000)

# Carregar dados dos ativos
ativos_df = pd.read_csv('https://raw.githubusercontent.com/richardrt13/bdrrecommendation/main/bdrs.csv')
ativos_df = ativos_df[ativos_df['Sector'].isin(['Tecnologia', 'Financeiro', 'Farmacêutico'])]
ativos_df = ativos_df[ativos_df['Ticker'].str.contains('34')]
tickers = ativos_df['Ticker'].apply(lambda x: x + '.SA').tolist()
sectors = ativos_df.set_index('Ticker')['Sector'].to_dict()
sectors_list = ativos_df['Sector'].tolist()
sectors_list = list(set(sectors_list))
sectors_list.insert(0, 'All')
sector_filter = st.selectbox('Sector', options=sectors_list)

if st.button('Montar Recomendação'):
    data = download_data(tickers)
    if data is not None:
        returns = calculate_annualized_returns(data)
        cov_matrix = calculate_annualized_covariance_matrix(data)

        filtered_tickers = tickers
        if sector_filter != 'All':
            filtered_tickers = [ticker for ticker in tickers if sectors[ticker[:-3]] == sector_filter]
            data = data[filtered_tickers]
            returns = calculate_annualized_returns(data)
            cov_matrix = calculate_annualized_covariance_matrix(data)

        optimal_weights = markowitz_optimization(returns, cov_matrix)
        portfolio = pd.DataFrame({
            'Ticker': filtered_tickers,
            'Weight': optimal_weights,
            'Investment': optimal_weights * budget
        })

        st.write('Optimized Portfolio Allocation')
        st.dataframe(portfolio)

        # Gráfico de alocação
        fig_allocation = px.pie(portfolio, names='Ticker', values='Investment', title='Portfolio Allocation')
        st.plotly_chart(fig_allocation)

        st.write('Portfolio Summary')
        expected_return = np.sum(returns * optimal_weights)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (expected_return - 0.02) / portfolio_volatility

        st.write(f"Total Investment: {portfolio['Investment'].sum()}")
        st.write(f"Expected Annual Return: {expected_return:.2%}")
        st.write(f"Portfolio Volatility: {portfolio_volatility:.2%}")
        st.write(f"Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")

        # Gráfico de retorno esperado vs. volatilidade
        fig_risk_return = px.scatter(
            x=[portfolio_volatility], 
            y=[expected_return], 
            labels={'x': 'Volatility', 'y': 'Expected Return'}, 
            title='Risk vs Return'
        )
        fig_risk_return.update_traces(marker=dict(size=10))
        st.plotly_chart(fig_risk_return)
else:
    st.write("Clique no botão para gerar a recomendação.")
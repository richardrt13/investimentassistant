import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

@st.cache_data
def load_assets():
    return pd.read_csv('https://raw.githubusercontent.com/richardrt13/bdrrecommendation/main/bdrs.csv')

def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_returns(prices):
    return prices.pct_change().dropna()

def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    p_return, p_volatility = portfolio_performance(weights, returns)
    return -(p_return - risk_free_rate) / p_volatility

def optimize_portfolio(returns, risk_free_rate):
    num_assets = returns.shape[1]
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets*[1./num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def main():
    st.title('BDR Recommendation and Portfolio Optimization')

    ativos_df = load_assets()
    setores = sorted(set(ativos_df['Sector']))
    setores.insert(0, 'Todos')

    sector_filter = st.selectbox('Selecione o Setor', options=setores)

    if sector_filter != 'Todos':
        ativos_df = ativos_df[ativos_df['Sector'] == sector_filter]

    # Filtrar ativos com informações necessárias
    ativos_df = ativos_df.dropna(subset=['P/L', 'P/VP', 'ROE', 'Liquidez Média Diária'])

    # Análise fundamentalista e de liquidez
    ativos_df['Score'] = (
        ativos_df['ROE'] / ativos_df['P/L'] +
        1 / ativos_df['P/VP'] +
        np.log(ativos_df['Liquidez Média Diária'])
    )

    # Selecionar os top 10 ativos com base no score
    top_ativos = ativos_df.nlargest(10, 'Score')

    st.subheader('Top 10 BDRs Recomendados')
    st.dataframe(top_ativos[['Ticker', 'Nome', 'Sector', 'P/L', 'P/VP', 'ROE', 'Liquidez Média Diária', 'Score']])

    # Otimização de portfólio
    tickers = top_ativos['Ticker'].apply(lambda x: x + '.SA').tolist()

    start_date = '2020-01-01'
    end_date = '2023-12-31'

    stock_data = get_stock_data(tickers, start_date, end_date)
    returns = calculate_returns(stock_data)

    risk_free_rate = 0.05  # 5% como exemplo, ajuste conforme necessário

    optimal_weights = optimize_portfolio(returns, risk_free_rate)

    st.subheader('Alocação Ótima do Portfólio')
    for ticker, weight in zip(tickers, optimal_weights):
        st.write(f"{ticker}: {weight:.2%}")

    portfolio_return, portfolio_volatility = portfolio_performance(optimal_weights, returns)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    st.subheader('Métricas do Portólio')
    st.write(f"Retorno Anual Esperado: {portfolio_return:.2%}")
    st.write(f"Volatilidade Anual: {portfolio_volatility:.2%}")
    st.write(f"Índice de Sharpe: {sharpe_ratio:.2f}")

if __name__ == "__main__":
    main()

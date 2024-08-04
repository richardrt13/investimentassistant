import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta
from requests.exceptions import ConnectionError
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier
import time

# Função para carregar os ativos do CSV
@st.cache_data
def load_assets():
    return pd.read_csv('https://raw.githubusercontent.com/richardrt13/bdrrecommendation/main/bdrs.csv')

# Função para obter dados fundamentais de um ativo
@st.cache_data
def get_fundamental_data(ticker, max_retries=3):
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'P/L': info.get('trailingPE', np.nan),
                'P/VP': info.get('priceToBook', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'Volume': info.get('averageVolume', np.nan),
                'Price': info.get('currentPrice', np.nan)
            }
        except ConnectionError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                st.warning(f"Não foi possível obter dados para {ticker}. Erro: {e}")
                return {
                    'P/L': np.nan,
                    'P/VP': np.nan,
                    'ROE': np.nan,
                    'Volume': np.nan,
                    'Price': np.nan
                }

# Função para obter dados históricos de preços com tratamento de erro
@st.cache_data
def get_stock_data(tickers, years=5, max_retries=3):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            return data
        except ConnectionError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                st.error(f"Erro ao obter dados históricos. Possível limite de requisição atingido. Erro: {e}")
                return pd.DataFrame()

# Função para calcular o retorno acumulado
@st.cache_data
def get_cumulative_return(ticker):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    hist = stock.history(start=start_date, end=end_date)
    if len(hist) > 0:
        cumulative_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
    else:
        cumulative_return = None
    return cumulative_return

def calculate_returns(prices):
    if prices.empty:
        return pd.DataFrame()
    returns = prices.pct_change().dropna()
    # Remove infinitos e NaNs
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns

# Função para calcular o desempenho do portfólio
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

# Função para calcular o índice de Sharpe negativo (para otimização)
def negative_sharpe_ratio(weights, returns, risk_free_rate):
    p_return, p_volatility = portfolio_performance(weights, returns)
    return -(p_return - risk_free_rate) / p_volatility

# Função para otimizar o portfólio
def optimize_portfolio(returns, risk_free_rate):
    num_assets = returns.shape[1]
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets*[1./num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Função para gerar portfólios aleatórios
def generate_random_portfolios(returns, num_portfolios=5000):
    results = []
    n_assets = returns.shape[1]
    for _ in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        p_return, p_volatility = portfolio_performance(weights, returns)
        results.append({
            'Return': p_return,
            'Volatility': p_volatility,
            'Sharpe': (p_return - risk_free_rate) / p_volatility,
            'Weights': weights
        })
    return pd.DataFrame(results)

# Função para plotar a fronteira eficiente
def plot_efficient_frontier(returns, optimal_portfolio):
    portfolios = generate_random_portfolios(returns)

    fig = go.Figure()

    # Plotar portfólios aleatórios
    fig.add_trace(go.Scatter(
        x=portfolios['Volatility'],
        y=portfolios['Return'],
        mode='markers',
        marker=dict(
            size=5,
            color=portfolios['Sharpe'],
            colorscale='Viridis',
            colorbar=dict(title='Índice de Sharpe'),
            showscale=True
        ),
        text=portfolios['Sharpe'].apply(lambda x: f'Sharpe: {x:.3f}'),
        hoverinfo='text+x+y',
        name='Portfólios'
    ))

    # Plotar portfólio ótimo
    opt_return, opt_volatility = portfolio_performance(optimal_portfolio, returns)
    opt_sharpe = (opt_return - risk_free_rate) / opt_volatility

    fig.add_trace(go.Scatter(
        x=[opt_volatility],
        y=[opt_return],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='star'
        ),
        text=[f'Portfólio Ótimo<br>Sharpe: {opt_sharpe:.3f}'],
        hoverinfo='text+x+y',
        name='Portfólio Ótimo'
    ))

    fig.update_layout(
        title='Fronteira Eficiente',
        xaxis_title='Volatilidade Anual',
        yaxis_title='Retorno Anual Esperado',
        showlegend=True,
        hovermode='closest'
    )

    return fig

def calculate_risk_parity_weights(returns):
    # Calcular a matriz de covariância
    cov_matrix = risk_models.sample_cov(returns)

    # Criar o objeto EfficientFrontier
    # Criar o objeto EfficientFrontier
    ef = EfficientFrontier(None, cov_matrix, weight_bounds=(0, 1))

    # Calcular os pesos de paridade de risco
    weights = ef.min_volatility()

    # Normalizar os pesos para garantir que somem 1
    weights = ef.clean_weights()

    return weights

# Função principal para rodar o aplicativo Streamlit
def main():
    st.title("Recomendação de Investimentos em BDRs")

    # Carregar a lista de ativos
    assets = load_assets()
    tickers = assets['Ticker'].tolist()

    # Sidebar para seleção do valor a ser investido
    st.sidebar.header("Parâmetros de Investimento")
    investment_amount = st.sidebar.number_input("Valor a ser investido (R$)", min_value=0.0, value=10000.0, step=1000.0)

    # Sidebar para seleção do método de otimização
    st.sidebar.header("Método de Otimização")
    optimization_method = st.sidebar.selectbox("Selecione o método de otimização", ['Índice de Sharpe', 'Paridade de Risco'])

    # Barra de progresso
    progress_bar = st.progress(0)

    # Coletar dados fundamentalistas
    st.header("Dados Fundamentalistas")
    fundamental_data = {}
    for i, ticker in enumerate(tickers):
        data = get_fundamental_data(ticker)
        fundamental_data[ticker] = data
        progress_bar.progress((i + 1) / len(tickers))

    fundamental_df = pd.DataFrame(fundamental_data).T
    st.write(fundamental_df)

    # Coletar dados históricos de preços
    st.header("Dados Históricos de Preços")
    prices = get_stock_data(tickers)
    st.write(prices)

    # Calcular retornos
    returns = calculate_returns(prices)
    st.write(returns)

    # Definir a taxa livre de risco
    global risk_free_rate
    risk_free_rate = 0.05

    # Otimizar o portfólio com base no método selecionado
    st.header("Otimização do Portfólio")
    if optimization_method == 'Índice de Sharpe':
        optimal_weights = optimize_portfolio(returns, risk_free_rate)
    elif optimization_method == 'Paridade de Risco':
        optimal_weights = calculate_risk_parity_weights(returns)

    optimal_portfolio = dict(zip(tickers, optimal_weights))
    st.write("Pesos do Portfólio Ótimo:")
    st.write(optimal_portfolio)

    # Plotar a fronteira eficiente
    st.header("Fronteira Eficiente")
    fig = plot_efficient_frontier(returns, optimal_weights)
    st.plotly_chart(fig)

    # Exibir o resumo do portfólio
    st.header("Resumo do Portfólio")
    summary = {
        'Ticker': list(optimal_portfolio.keys()),
        'Peso (%)': [round(weight * 100, 2) for weight in optimal_weights],
        'Preço Atual (R$)': [fundamental_data[ticker]['Price'] for ticker in optimal_portfolio.keys()],
        'Volume Médio': [fundamental_data[ticker]['Volume'] for ticker in optimal_portfolio.keys()],
        'P/L': [fundamental_data[ticker]['P/L'] for ticker in optimal_portfolio.keys()],
        'P/VP': [fundamental_data[ticker]['P/VP'] for ticker in optimal_portfolio.keys()],
        'ROE (%)': [fundamental_data[ticker]['ROE'] for ticker in optimal_portfolio.keys()],
        'Retorno Acumulado (%)': [round(get_cumulative_return(ticker) * 100, 2) for ticker in optimal_portfolio.keys()]
    }
    summary_df = pd.DataFrame(summary)
    st.write(summary_df)

# Função para exibir o resumo do portfólio
def display_summary():
    st.header("Resumo do Portfólio")
    st.write(summary_df)

if __name__ == "__main__":
    main()
    display_summary()
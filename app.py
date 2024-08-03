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
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]
    result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x, portfolio_performance(result.x)

# Streamlit Interface
st.title('Investment Portfolio Recommender')

# Carregar dados dos ativos
@st.cache_data
def load_assets():
    return pd.read_csv('https://raw.githubusercontent.com/richardrt13/bdrrecommendation/main/bdrs.csv')

ativos_df = load_assets()
setores = sorted(set(ativos_df['Sector']))
setores.insert(0, 'Todos')

# Seleção do setor
sector_filter = st.selectbox('Selecione o Setor', options=setores)

# Filtrar ativos conforme o setor selecionado
if sector_filter != 'Todos':
    ativos_df = ativos_df[ativos_df['Sector'] == sector_filter]

# Obter tickers
tickers = ativos_df['Ticker'].apply(lambda x: x + '.SA').tolist()
sector_mapping = ativos_df.set_index('Ticker')['Sector'].to_dict()

# Definições de entrada
budget = st.number_input('Orçamento para Investimento', min_value=1000, value=10000, step=1000)
max_assets = st.slider('Número Máximo de Ativos na Carteira', min_value=1, max_value=20, value=10)

# Processar dados e otimizar
if st.button('Montar Recomendação'):
    with st.spinner('Baixando dados e otimizando portfólio...'):
        data = download_data(tickers)

        if data is not None and not data.empty:
            returns = calculate_annualized_returns(data)
            cov_matrix = calculate_annualized_covariance_matrix(data)

            # Verificar se há ativos suficientes após a filtragem
            if len(returns) < 2:
                st.error("Número insuficiente de ativos para calcular a matriz de covariância.")
                st.stop()

            try:
                weights, (volatility, return_) = optimize_portfolio(returns, cov_matrix)
            except Exception as e:
                st.error(f"Erro durante a otimização da carteira: {e}")
                st.stop()

            portfolio = pd.DataFrame({
                'Ticker': tickers,
                'Weight': weights,
                'Investment': weights * budget
            })

            st.subheader('Alocação de Carteira Otimizada')
            st.dataframe(portfolio.style.format({'Weight': '{:.2%}', 'Investment': '${:.2f}'}))

            fig_allocation = px.pie(portfolio, names='Ticker', values='Investment', title='Alocação da Carteira')
            st.plotly_chart(fig_allocation)

            st.subheader('Resumo da Carteira')
            st.metric("Investimento Total", f"${portfolio['Investment'].sum():.2f}")
            st.metric("Retorno Anual Esperado", f"{return_:.2%}")
            st.metric("Volatilidade da Carteira", f"{volatility:.2%}")
            st.metric("Índice de Sharpe da Carteira", f"{(return_ - 0.02) / volatility:.2f}")

            fig_risk_return = px.scatter(
                x=[volatility], 
                y=[return_], 
                labels={'x': 'Volatilidade', 'y': 'Retorno Esperado'}, 
                title='Risco vs Retorno'
            )
            fig_risk_return.update_traces(marker=dict(size=10))
            st.plotly_chart(fig_risk_return)
        else:
            st.error("Não foi possível baixar os dados. Por favor, tente novamente mais tarde.")
else:
    st.info("Clique no botão para gerar a recomendação.")
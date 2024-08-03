import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px

# Funções auxiliares
@st.cache_data
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

@st.cache_data
def get_magic_formula_rankings(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data[ticker] = {
                'ROC': stock.info.get('returnOnEquity', np.nan),
                'EY': stock.info.get('earningsYield', np.nan)
            }
        except Exception as e:
            st.error(f"Erro ao obter dados para {ticker}: {e}")

    df = pd.DataFrame(data).T.dropna()
    df['ROC_Rank'] = df['ROC'].rank(ascending=False)
    df['EY_Rank'] = df['EY'].rank(ascending=False)
    df['MagicFormula_Rank'] = df['ROC_Rank'] + df['EY_Rank']
    return df.sort_values(by='MagicFormula_Rank')

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    if portfolio_volatility == 0:
        return 0  # Retorna 0 se a volatilidade for zero para evitar divisão por zero
    
    return -(portfolio_return - risk_free_rate) / portfolio_volatility

def markowitz_optimization(returns, cov_matrix, max_assets):
    num_assets = len(returns)
    
    if num_assets == 0:
        st.error("Não há ativos disponíveis para otimização.")
        return [], []
    
    args = (returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    def objective(weights, returns, cov_matrix):
        return negative_sharpe_ratio(weights, returns, cov_matrix)
    
    initial_weights = num_assets * [1. / num_assets]
    
    try:
        result = minimize(objective, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    except Exception as e:
        st.error(f"Erro durante a otimização: {e}")
        return [], []

    if not result.success:
        st.warning(f"A otimização não convergiu: {result.message}")
        return [], []

    weights = result.x
    selected_assets = np.argsort(weights)[-max_assets:]
    selected_weights = weights[selected_assets]
    
    # Verifica se a soma dos pesos é zero
    if np.sum(selected_weights) == 0:
        st.error("A otimização resultou em pesos zero para todos os ativos.")
        return [], []
    
    selected_weights /= np.sum(selected_weights)  # Renormaliza os pesos

    return selected_assets, selected_weights

# Streamlit Interface
st.title('BDR Portfolio Optimizer')

budget = st.number_input('Investment Budget', min_value=1000, value=10000, step=1000)

# Carregar dados dos ativos
@st.cache_data
def load_assets():
    return pd.read_csv('https://raw.githubusercontent.com/richardrt13/bdrrecommendation/main/bdrs.csv')

ativos_df = load_assets()
ativos_df = ativos_df[ativos_df['Sector'].isin(['Tecnologia', 'Financeiro', 'Farmacêutico'])]
ativos_df = ativos_df[ativos_df['Ticker'].str.contains('34')]
tickers = ativos_df['Ticker'].apply(lambda x: x + '.SA').tolist()
sectors = ativos_df.set_index('Ticker')['Sector'].to_dict()
sectors_list = sorted(set(ativos_df['Sector']))
sectors_list.insert(0, 'All')
sector_filter = st.selectbox('Sector', options=sectors_list)

max_assets = st.slider('Number of Assets in Portfolio', min_value=1, max_value=20, value=10)

if st.button('Montar Recomendação'):
    with st.spinner('Baixando dados e otimizando portfólio...'):
        data = download_data(tickers)
        
        if data is not None and not data.empty:
            returns = calculate_annualized_returns(data)
            cov_matrix = calculate_annualized_covariance_matrix(data)

            filtered_tickers = tickers
            if sector_filter != 'All':
                filtered_tickers = [ticker for ticker in tickers if sectors[ticker[:-3]] == sector_filter]
                data = data[filtered_tickers]
                returns = calculate_annualized_returns(data)
                cov_matrix = calculate_annualized_covariance_matrix(data)

            if len(returns) == 0:
                st.error("Não há dados suficientes para realizar a otimização.")
                st.stop()

            # Verificar se há ativos suficientes após a filtragem
            if len(returns) < max_assets:
                max_assets = len(returns)
                st.warning(f"Não há ativos suficientes após a filtragem. O número máximo de ativos foi ajustado para {max_assets}.")

            magic_formula_df = get_magic_formula_rankings(filtered_tickers)
            selected_tickers = magic_formula_df.head(max_assets).index.tolist()

            selected_returns = returns[selected_tickers]
            selected_cov_matrix = cov_matrix.loc[selected_tickers, selected_tickers]

            selected_assets, optimal_weights = markowitz_optimization(selected_returns, selected_cov_matrix, max_assets)

            if len(selected_assets) == 0 or len(optimal_weights) == 0:
                st.error("A otimização falhou. Por favor, tente com diferentes parâmetros ou ativos.")
                st.stop()

            final_tickers = [selected_tickers[i] for i in selected_assets]

            portfolio = pd.DataFrame({
                'Ticker': final_tickers,
                'Weight': optimal_weights,
                'Investment': optimal_weights * budget
            })

            st.subheader('Optimized Portfolio Allocation')
            st.dataframe(portfolio.style.format({'Weight': '{:.2%}', 'Investment': '${:.2f}'}))

            # Gráfico de alocação
            fig_allocation = px.pie(portfolio, names='Ticker', values='Investment', title='Portfolio Allocation')
            st.plotly_chart(fig_allocation)

            st.subheader('Portfolio Summary')
            expected_return = np.sum(selected_returns[selected_assets] * optimal_weights)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(selected_cov_matrix.values, optimal_weights)))
            sharpe_ratio = (expected_return - 0.02) / portfolio_volatility

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Investment", f"${portfolio['Investment'].sum():.2f}")
                st.metric("Expected Annual Return", f"{expected_return:.2%}")
            with col2:
                st.metric("Portfolio Volatility", f"{portfolio_volatility:.2%}")
                st.metric("Portfolio Sharpe Ratio", f"{sharpe_ratio:.2f}")

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
            st.error("Não foi possível baixar os dados. Por favor, tente novamente mais tarde.")
else:
    st.info("Clique no botão para gerar a recomendação.")
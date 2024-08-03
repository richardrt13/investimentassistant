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

def markowitz_optimization(returns, cov_matrix, max_assets):
    num_assets = len(returns)
    args = (returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    def objective(weights):
        return negative_sharpe_ratio(weights, returns, cov_matrix)
    
    result = minimize(objective, num_assets * [1. / num_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    weights = result.x
    # Manter apenas os melhores ativos com pesos positivos
    selected_assets = np.argsort(weights)[-max_assets:]
    selected_weights = weights[selected_assets]

    return selected_assets, selected_weights

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
ativos_df = ativos_df[~ativos_df['Sector'].isin(['-'])]
ativos_df = ativos_df[ativos_df['Ticker'].str.contains('34')]
tickers = ativos_df['Ticker'].apply(lambda x: x + '.SA').tolist()
sectors = ativos_df.set_index('Ticker')['Sector'].to_dict()
sectors_list = ativos_df['Sector'].tolist()
sectors_list = list(set(sectors_list))
sectors_list.insert(0, 'All')
sector_filter = st.selectbox('Sector', options=sectors_list)

max_assets = st.slider('Number of Assets in Portfolio', min_value=1, max_value=20, value=10)

if st.button('Montar Recomendação'):
    data = download_data(tickers)
    if data is not None:
        returns = calculate_annualized_returns(data)
        cov_matrix = calculate_annualized_covariance_matrix(data)
        
        st.write("Retornos anualizados:")
        st.write(returns)
        st.write("Matriz de covariância anualizada:")
        st.write(cov_matrix)

        filtered_tickers = tickers
        if sector_filter != 'All':
            filtered_tickers = [ticker for ticker in tickers if sectors[ticker[:-3]] == sector_filter]
            data = data[filtered_tickers]
            returns = calculate_annualized_returns(data)
            cov_matrix = calculate_annualized_covariance_matrix(data)
            
            st.write("Retornos anualizados após filtragem:")
            st.write(returns)
            st.write("Matriz de covariância anualizada após filtragem:")
            st.write(cov_matrix)

        # Verificar se há ativos suficientes após a filtragem
        if len(returns) < max_assets:
            max_assets = len(returns)
            st.warning(f"Não há ativos suficientes após a filtragem. O número máximo de ativos foi ajustado para {max_assets}.")

        try:
            selected_assets, optimal_weights = markowitz_optimization(returns, cov_matrix, max_assets)
        except Exception as e:
            st.error(f"Erro na otimização: {e}")
            st.stop()

        selected_tickers = [filtered_tickers[i] for i in selected_assets]

        portfolio = pd.DataFrame({
            'Ticker': selected_tickers,
            'Weight': optimal_weights,
            'Investment': optimal_weights * budget
        })

        st.write('Optimized Portfolio Allocation')
        st.dataframe(portfolio)

        # Gráfico de alocação
        fig_allocation = px.pie(portfolio, names='Ticker', values='Investment', title='Portfolio Allocation')
        st.plotly_chart(fig_allocation)

        st.write('Portfolio Summary')
        expected_return = np.sum(returns[selected_assets] * optimal_weights)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix[selected_assets][:, selected_assets], optimal_weights)))
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
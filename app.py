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
    df = df.sort_values(by='MagicFormula_Rank')
    
    return df

def markowitz_optimization(returns, cov_matrix, prices, budget, max_assets):
    num_assets = len(returns)
    args = (returns, cov_matrix, prices, budget)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: budget - np.sum(x * prices)}
    )
    bounds = tuple((0, 1) for asset in range(num_assets))

    def objective(weights, returns, cov_matrix):
        return negative_sharpe_ratio(weights, returns, cov_matrix)
    
    result = minimize(objective, num_assets * [1. / num_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    weights = result.x
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
ativos_df = ativos_df[ativos_df['Sector'].isin(['Tecnologia', 'Financeiro', 'Farmacêutico'])]
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

        filtered_tickers = tickers
        if sector_filter != 'All':
            filtered_tickers = [ticker for ticker in tickers if sectors[ticker[:-3]] == sector_filter]
            data = data[filtered_tickers]
            returns = calculate_annualized_returns(data)
            cov_matrix = calculate_annualized_covariance_matrix(data)

        # Verificar se há ativos suficientes após a filtragem
        if len(returns) < max_assets:
            max_assets = len(returns)
            st.warning(f"Não há ativos suficientes após a filtragem. O número máximo de ativos foi ajustado para {max_assets}.")

        magic_formula_df = get_magic_formula_rankings(filtered_tickers)
        selected_tickers = magic_formula_df.head(max_assets).index.tolist()

        selected_returns = returns[selected_tickers]
        selected_cov_matrix = cov_matrix.loc[selected_tickers, selected_tickers]

        try:
            prices = yf.download(selected_tickers, period='1d')['Adj Close'].iloc[-1].values
            selected_assets, optimal_weights = markowitz_optimization(selected_returns, selected_cov_matrix, prices, budget, max_assets)
        except Exception as e:
            st.error(f"Erro na otimização: {e}")
            st.stop()

        final_tickers = [selected_tickers[i] for i in selected_assets]

        portfolio = pd.DataFrame({
            'Ticker': final_tickers,
            'Weight': optimal_weights,
            'Investment': optimal_weights * budget
        })
# Exibir resultados
        st.write('Optimized Portfolio Allocation')
        st.dataframe(portfolio)

        st.write('Portfolio Summary')
        expected_return = np.sum(selected_returns.loc[final_tickers].mean() * optimal_weights)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(selected_cov_matrix, optimal_weights)))
        sharpe_ratio = (expected_return - 0.02) / portfolio_volatility

        st.write(f"Total Investment: R$ {portfolio['Investment'].sum():,.2f}")
        st.write(f"Expected Annual Return: {expected_return:.2%}")
        st.write(f"Portfolio Volatility: {portfolio_volatility:.2%}")
        st.write(f"Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")

        # Visualização dos ativos e suas alocações
        fig = px.bar(portfolio, x='Ticker', y='Investment', title='Investment Allocation by Asset')
        st.plotly_chart(fig)

        # Visualização do retorno e volatilidade
        st.write("Portfolio Statistics")
        st.write(f"Expected Return: {expected_return:.2%}")
        st.write(f"Volatility: {portfolio_volatility:.2%}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Gráfico de fronteira eficiente
        def efficient_frontier(returns, cov_matrix, num_portfolios=10000):
            results = np.zeros((num_portfolios, 3))
            for i in range(num_portfolios):
                weights = np.random.random(len(returns))
                weights /= np.sum(weights)
                portfolio_return = np.sum(returns * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
                results[i] = [portfolio_return, portfolio_volatility, sharpe_ratio]
            return results

        frontier_results = efficient_frontier(selected_returns, selected_cov_matrix)
        frontier_df = pd.DataFrame(frontier_results, columns=['Return', 'Volatility', 'Sharpe Ratio'])
        fig_frontier = px.scatter(frontier_df, x='Volatility', y='Return', color='Sharpe Ratio', title='Efficient Frontier')
        st.plotly_chart(fig_frontier)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pymongo import MongoClient
import time

# Função para conectar ao MongoDB
def connect_to_mongo(uri, db_name, collection_name):
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        return collection
    except Exception as e:
        st.error(f"Erro ao conectar com o MongoDB: {e}")
        return None

# Inicializar o banco de dados
@st.cache_resource
def init_db():
    mongo_uri = "mongodb+srv://richardrt13:QtZ9CnSP6dv93hlh@stockidea.isx8swk.mongodb.net/?retryWrites=true&w=majority&appName=StockIdea"
    return connect_to_mongo(mongo_uri, 'StockIdea', 'transactions')

# Função para carregar os ativos do CSV
@st.cache_data
def load_assets():
    try:
        return pd.read_csv('https://raw.githubusercontent.com/richardrt13/bdrrecommendation/main/bdrs.csv')
    except Exception as e:
        st.error(f"Erro ao carregar os ativos: {e}")
        return pd.DataFrame()

# Função para obter dados fundamentais de um ativo
@st.cache_data(ttl=3600)
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
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                st.warning(f"Não foi possível obter dados para {ticker}. Erro: {e}")
                return {key: np.nan for key in ['P/L', 'P/VP', 'ROE', 'Volume', 'Price']}

# Função para obter dados históricos de preços com tratamento de erro
@st.cache_data(ttl=3600)
def get_stock_data(tickers, years=2, max_retries=3):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    all_data = pd.DataFrame()
    
    for ticker in tickers:
        for attempt in range(max_retries):
            try:
                data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
                if not data.empty:
                    all_data[ticker] = data
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Não foi possível obter dados para {ticker}. Erro: {e}")
                else:
                    time.sleep(2 ** attempt)
    
    if all_data.empty:
        st.error("Não foi possível obter dados para nenhum dos tickers.")
    return all_data

# Função para calcular o retorno acumulado
@st.cache_data
def get_cumulative_return(ticker):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    hist = stock.history(start=start_date, end=end_date)
    return (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1 if len(hist) > 0 else None

def calculate_returns(prices):
    if prices.empty:
        return pd.DataFrame()
    returns = prices.pct_change().dropna()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()

# Função para calcular o desempenho do portfólio
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

# Função para calcular o índice de Sharpe negativo (para otimização)
def negative_sharpe_ratio(weights, returns, risk_free_rate):
    p_return, p_volatility = portfolio_performance(weights, returns)
    return -(p_return - risk_free_rate) / p_volatility

def risk_parity_optimization(returns):
    if returns.empty:
        st.error("Não há dados de retorno para otimizar.")
        return np.array([])

    def risk_parity_objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        asset_risk_contribution = (weights * np.dot(returns.cov() * 252, weights)) / np.sqrt(portfolio_variance)
        return np.sum((asset_risk_contribution - (1.0 / len(weights))) ** 2)

    num_assets = returns.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    try:
        result = minimize(risk_parity_objective, num_assets * [1. / num_assets], bounds=bounds, constraints=constraints)
        return result.x
    except Exception as e:
        st.error(f"Erro na otimização: {e}")
        return np.array([])

# Função para gerar portfólios aleatórios
@st.cache_data
def generate_random_portfolios(returns, num_portfolios=1000, risk_free_rate=0.05):
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
def plot_efficient_frontier(returns, optimal_portfolio, risk_free_rate=0.05):
    portfolios = generate_random_portfolios(returns, risk_free_rate=risk_free_rate)
    fig = go.Figure()
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

def get_current_positions(collection):
    pipeline = [
        {'$group': {
            '_id': '$Ticker',
            'quantity': {'$sum': {'$cond': [{'$eq': ['$Action', 'BUY']}, '$Quantity', {'$multiply': ['$Quantity', -1]}]}},
            'average_price': {'$avg': '$Price'}
        }}
    ]
    results = list(collection.aggregate(pipeline))
    positions = pd.DataFrame(results).rename(columns={'_id': 'Ticker'})
    return positions[positions['quantity'] > 0]  # Filtra ativos com quantidade positiva

def main():
    st.title('BDR Recommendation and Portfolio Optimization')
    
    collection = init_db()
    if collection is None:
        st.error("Não foi possível conectar ao banco de dados. Por favor, tente novamente mais tarde.")
        return

    ativos_df = load_assets()
    if ativos_df.empty:
        st.error("Não foi possível carregar os ativos. Por favor, tente novamente mais tarde.")
        return

    ativos_df["Sector"] = ativos_df["Sector"].replace("-", "Outros")
    setores = sorted(set(ativos_df['Sector']))
    setores.insert(0, 'Todos')
    sector_filter = st.multiselect('Selecione o Setor', options=setores, default=['Todos'])
    
    if 'Todos' not in sector_filter:
        ativos_df = ativos_df[ativos_df['Sector'].isin(sector_filter)]

    invest_value = st.number_input('Valor a ser investido (R$)', min_value=100.0, value=10000.0, step=100.0)

    if st.button('Gerar Recomendação'):
        with st.spinner('Processando dados...'):
            fundamental_data = []
            for ticker in ativos_df['Ticker']:
                data = get_fundamental_data(ticker)
                data['Ticker'] = ticker
                fundamental_data.append(data)
            
            fundamental_df = pd.DataFrame(fundamental_data)
            ativos_df = pd.merge(ativos_df, fundamental_df, on='Ticker').dropna()

            tickers = ativos_df['Ticker'].tolist()
            stock_data = get_stock_data(tickers)
            stock_returns = calculate_returns(stock_data)

            positions = get_current_positions(collection)
            merged_positions = pd.merge(positions, ativos_df, on='Ticker', how='right')

            initial_weights = np.zeros(stock_returns.shape[1])
            for i, ticker in enumerate(stock_returns.columns):
                if ticker in merged_positions['Ticker'].values:
                    qty = merged_positions.loc[merged_positions['Ticker'] == ticker, 'quantity'].values[0]
                    price = merged_positions.loc[merged_positions['Ticker'] == ticker, 'average_price'].values[0]
                    initial_weights[i] = qty * price / invest_value

            initial_weights = initial_weights / np.sum(initial_weights)

            if not stock_returns.empty:
                optimal_portfolio = risk_parity_optimization(stock_returns)
                ativos_df['Initial Weight'] = initial_weights
                ativos_df['Optimized Weight'] = optimal_portfolio

                st.subheader('Resultados da Recomendação de Portfólio')
                st.write('Valores em porcentagem')
                st.dataframe(ativos_df[['Ticker', 'Company', 'Sector', 'Initial Weight', 'Optimized Weight']])

                fig = plot_efficient_frontier(stock_returns, optimal_portfolio)
                st.plotly_chart(fig)
            else:
                st.error("Não foi possível obter dados de retorno para os ativos selecionados. Por favor, tente novamente mais tarde.")

            st.subheader('Posições Atuais da Carteira')
            st.dataframe(positions[['Ticker', 'quantity', 'average_price']])

if __name__ == "__main__":
    main()

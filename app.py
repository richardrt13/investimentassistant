import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta

@st.cache_data
def load_assets():
    return pd.read_csv('https://raw.githubusercontent.com/richardrt13/bdrrecommendation/main/bdrs.csv')

@st.cache_data
def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'P/L': info.get('trailingPE', np.nan),
        'P/VP': info.get('priceToBook', np.nan),
        'ROE': info.get('returnOnEquity', np.nan),
        'Volume': info.get('averageVolume', np.nan),
        'Price': info.get('currentPrice', np.nan)
    }

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


def get_stock_data(tickers, years=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def get_cumulative_return(tickers, years=5):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    hist = stock.history(start=start_date, end=end_date)
    if len(hist) > 0:
        cumulative_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
    else:
        cumulative_return = None
    return cumulative_return

def calculate_returns(prices):
    return prices.pct_change().dropna()

def calculate_cumulative_returns(prices):
    return (prices.pct_change() + 1).prod() - 1

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

def main():
    st.title('BDR Recommendation and Portfolio Optimization')

    ativos_df = load_assets()
    setores = sorted(set(ativos_df['Sector']))
    setores.insert(0, 'Todos')

    sector_filter = st.multiselect('Selecione o Setor', options=setores)

    if sector_filter != 'Todos':
        ativos_df = ativos_df[ativos_df['Sector'].isin(sector_filter)]

    invest_value = st.number_input('Valor a ser investido (R$)', min_value=100.0, value=10000.0, step=100.0)

    if st.button('Gerar Recomendação'):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Obter dados fundamentalistas
        fundamental_data = []
        for i, ticker in enumerate(ativos_df['Ticker']):
            status_text.text(f'Carregando dados para {ticker}...')
            progress_bar.progress((i + 1) / len(ativos_df))
            data = get_fundamental_data(ticker + '.SA')
            data['Ticker'] = ticker
            fundamental_data.append(data)

        fundamental_df = pd.DataFrame(fundamental_data)
        ativos_df = ativos_df.merge(fundamental_df, on='Ticker')

        # Filtrar ativos com informações necessárias
        ativos_df = ativos_df.dropna(subset=['P/L', 'P/VP', 'ROE', 'Volume', 'Price'])

        # Análise fundamentalista e de liquidez
        ativos_df['Score'] = (
            ativos_df['ROE'] / ativos_df['P/L'] +
            1 / ativos_df['P/VP'] +
            np.log(ativos_df['Volume'])
        )

        # Selecionar os top 10 ativos com base no score
        top_ativos = ativos_df.nlargest(10, 'Score')

        # Obter dados históricos dos últimos 5 anos
        tickers = top_ativos['Ticker'].apply(lambda x: x + '.SA').tolist()
        status_text.text('Obtendo dados históricos...')
        stock_data = get_stock_data(tickers)

        # Calcular rentabilidade acumulada
        cumulative_returns = calculate_cumulative_returns(stock_data)
        top_ativos['Rentabilidade Acumulada (5 anos)'] = get_cumulative_return(ticker)

        st.subheader('Top 10 BDRs Recomendados')
        st.dataframe(top_ativos[['Ticker', 'Sector', 'P/L', 'P/VP', 'ROE', 'Volume', 'Price', 'Score', 'Rentabilidade Acumulada (5 anos)']])

        # Otimização de portfólio
        returns = calculate_returns(stock_data)

        global risk_free_rate
        risk_free_rate = 0.05  # 5% como exemplo, ajuste conforme necessário

        status_text.text('Otimizando portfólio...')
        optimal_weights = optimize_portfolio(returns, risk_free_rate)

        st.subheader('Alocação Ótima do Portfólio')
        allocation_data = []
        for ticker, weight in zip(tickers, optimal_weights):
            price = top_ativos.loc[top_ativos['Ticker'] == ticker[:-3], 'Price'].values[0]
            allocated_value = weight * invest_value
            shares = allocated_value / price
            cumulative_return = top_ativos.loc[top_ativos['Ticker'] == ticker[:-3], 'Rentabilidade Acumulada (5 anos)'].values[0]
            allocation_data.append({
                'Ticker': ticker,
                'Peso': f"{weight:.2%}",
                'Valor Alocado': f"R$ {allocated_value:.2f}",
                'Quantidade de Ações': f"{shares:.2f}",
                'Rentabilidade Acumulada (5 anos)': f"{cumulative_return:.2%}"
            })
        
        allocation_df = pd.DataFrame(allocation_data)
        st.table(allocation_df)

        portfolio_return, portfolio_volatility = portfolio_performance(optimal_weights, returns)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        st.subheader('Métricas do Portfólio')
        st.write(f"Retorno Anual Esperado: {portfolio_return:.2%}")
        st.write(f"Volatilidade Anual: {portfolio_volatility:.2%}")
        st.write(f"Índice de Sharpe: {sharpe_ratio:.2f}")

        # Gerar e exibir o gráfico de dispersão
        status_text.text('Gerando gráfico da fronteira eficiente...')
        fig = plot_efficient_frontier(returns, optimal_weights)
        st.plotly_chart(fig)

        status_text.text('Análise concluída!')
        progress_bar.progress(100)

if __name__ == "__main__":
    main()

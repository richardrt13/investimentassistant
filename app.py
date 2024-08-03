import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.cluster.hierarchy as sch

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

def hrp_portfolio(cov_matrix, max_assets):
    if cov_matrix.empty or cov_matrix.shape[0] < 2:
        st.error("Não há dados suficientes para realizar a otimização HRP.")
        return [], []

    corr = cov_matrix.corr()
    distance = np.sqrt(0.5 * (1 - corr))

    # Verifica se a matriz de distância tem valores válidos
    if np.isnan(distance).all() or np.isinf(distance).all():
        st.error("A matriz de distância não contém valores válidos.")
        return [], []

    linkage = sch.linkage(distance, 'complete')
    sort_ix = sch.leaves_list(linkage)

    clusters = [sort_ix]

    while len(clusters) > max_assets:
        clusters = [cluster[:len(cluster)//2] for cluster in clusters] + [cluster[len(cluster)//2:] for cluster in clusters]
        if len(clusters) > max_assets:
            clusters = clusters[:max_assets]

    selected_assets = clusters[0] if clusters else []
    weights = np.ones(len(selected_assets)) / len(selected_assets)
    return selected_assets, weights

# Streamlit Interface
st.title('BDR Portfolio Optimizer (HRP)')

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

            if len(returns) < max_assets:
                max_assets = len(returns)
                st.warning(f"Não há ativos suficientes após a filtragem. O número máximo de ativos foi ajustado para {max_assets}.")

            magic_formula_df = get_magic_formula_rankings(filtered_tickers)
            selected_tickers = magic_formula_df.head(max_assets).index.tolist()

            selected_returns = returns[selected_tickers]
            selected_cov_matrix = cov_matrix.loc[selected_tickers, selected_tickers]

            if selected_cov_matrix.isnull().values.any() or selected_cov_matrix.shape[0] < 2:
                st.error("A matriz de covariância contém valores inválidos ou insuficientes.")
                st.stop()

            try:
                selected_assets, optimal_weights = hrp_portfolio(selected_cov_matrix, max_assets)
            except Exception as e:
                st.error(f"Erro durante a otimização HRP: {e}")
                st.stop()

            if len(selected_assets) == 0 or len(optimal_weights) == 0:
                st.error("A otimização falhou. Por favor, tente com diferentes parâmetros ou ativos.")
                st.stop()

            final_tickers = [selected_tickers[i] for i in selected_assets]

            portfolio = pd.DataFrame({
                'Ticker': final_tickers,
                'Weight': optimal_weights,
                'Investment': optimal_weights * budget
            })

            st.subheader('Optimized Portfolio Allocation (HRP)')
            st.dataframe(portfolio.style.format({'Weight': '{:.2%}', 'Investment': '${:.2f}'}))

            fig_allocation = px.pie(portfolio, names='Ticker', values='Investment', title='Portfolio Allocation (HRP)')
            st.plotly_chart(fig_allocation)

            st.subheader('Portfolio Summary')
            expected_return = np.sum(selected_returns * optimal_weights)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(selected_cov_matrix.values, optimal_weights)))
            sharpe_ratio = (expected_return - 0.02) / portfolio_volatility

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Investment", f"${portfolio['Investment'].sum():.2f}")
                st.metric("Expected Annual Return", f"{expected_return:.2%}")
            with col2:
                st.metric("Portfolio Volatility", f"{portfolio_volatility:.2%}")
                st.metric("Portfolio Sharpe Ratio", f"{sharpe_ratio:.2f}")

            fig_risk_return = px.scatter(
                x=[portfolio_volatility], 
                y=[expected_return], 
                labels={'x': 'Volatility', 'y': 'Expected Return'}, 
                title='Risk vs Return (HRP)'
            )
            fig_risk_return.update_traces(marker=dict(size=10))
            st.plotly_chart(fig_risk_return)
        else:
            st.error("Não foi possível baixar os dados. Por favor, tente novamente mais tarde.")
else:
    st.info("Clique no botão para gerar a recomendação.")
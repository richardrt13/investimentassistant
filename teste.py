import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta
from requests.exceptions import ConnectionError
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

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
#@st.cache_data
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

# Novas funções para detecção de anomalias e cálculo de indicadores

def detect_price_anomalies(prices, window=20, threshold=2):
    returns = prices.pct_change().dropna()
    model = ARIMA(returns, order=(1,1,1))
    results = model.fit()
    residuals = results.resid
    mean = residuals.rolling(window=window).mean()
    std = residuals.rolling(window=window).std()
    z_scores = (residuals - mean) / std
    return abs(z_scores) > threshold

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_adjusted_score(row):
    base_score = (
        row['ROE'] / row['P/L'] +
        1 / row['P/VP'] +
        np.log(row['Volume']) +
        row['revenue_growth'] * 10 +  # Multiplicador para dar mais peso
        row['income_growth'] * 10 +   # Multiplicador para dar mais peso
        row['debt_stability'] * 5      # Multiplicador para dar mais peso
    )
    anomaly_penalty = sum([row[col] for col in ['price_anomaly', 'rsi_anomaly']])
    return base_score * (1 - 0.1 * anomaly_penalty)

def adjust_weights_for_growth_and_anomalies(weights, returns, growth_data):
    anomaly_scores = calculate_anomaly_scores(returns)
    growth_scores = growth_data.mean(axis=1)  # Média dos fatores de crescimento
    
    # Normalizar os scores
    growth_scores = (growth_scores - growth_scores.min()) / (growth_scores.max() - growth_scores.min())
    
    # Ajustar pesos
    adjusted_weights = weights * (1 - 0.5 * anomaly_scores + 0.5 * growth_scores)
    return adjusted_weights / adjusted_weights.sum()

def adjust_weights_for_anomalies(weights, anomaly_scores):
    adjusted_weights = weights * (1 - anomaly_scores)
    return adjusted_weights / adjusted_weights.sum()

def calculate_anomaly_scores(returns):
    anomaly_scores = returns.apply(lambda x: detect_price_anomalies(x).mean())
    return anomaly_scores

def get_financial_growth_data(ticker, years=5):
    stock = yf.Ticker(ticker)
    
    # Obter dados financeiros anuais
    try:
        financials = stock.financials
        balance_sheet = stock.balance_sheet
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    
    if financials.empty or balance_sheet.empty:
        print(f"No financial data available for {ticker}.")
        return None
    
    try:
        # Verificar se há dados financeiros suficientes
        if 'Total Revenue' not in financials.index or 'Net Income' not in financials.index:
            print(f"Necessary financial metrics not available for {ticker}.")
            return None
        
        # Calcular crescimento da receita
        revenues = financials.loc['Total Revenue'].dropna().sort_index()
        if len(revenues) > 1:
            available_years = min(len(revenues) - 1, years)
            revenue_growth = round((revenues.iloc[-1] / revenues.iloc[-(available_years+1)]) ** (1/available_years) - 1,2)
        else:
            revenue_growth = None
        
        # Calcular crescimento do lucro
        net_income = financials.loc['Net Income'].dropna().sort_index()
        if len(net_income) > 1 and net_income.iloc[0] > 0:
            available_years = min(len(net_income) - 1, years)
            income_growth = round((net_income.iloc[-1] / net_income.iloc[-(available_years+1)]) ** (1/available_years) - 1,2)
        else:
            income_growth = None
        
        # Verificar se há dados de balanço suficientes
        if 'Total Debt' not in balance_sheet.index:
            print(f"Necessary balance sheet metrics not available for {ticker}.")
            return None
        
        # Calcular estabilidade da dívida
        total_debt = balance_sheet.loc['Total Debt'].dropna().sort_index()
        if len(total_debt) > 1:
            available_years = min(len(total_debt) - 1, years)
            debt_stability = round(-((total_debt.iloc[-1] / total_debt.iloc[-(available_years+1)]) ** (1/available_years) - 1),2)
        else:
            debt_stability = None
    except Exception as e:
        print(f"Error calculating growth data for {ticker}: {e}")
        return None
    
    return {
        'revenue_growth': revenue_growth,
        'income_growth': income_growth,
        'debt_stability': debt_stability
    }


def main():
    st.title('BDR Recommendation and Portfolio Optimization')

    ativos_df = load_assets()

    # Substituir "-" por "Outros" na coluna "Sector"
    ativos_df["Sector"] = ativos_df["Sector"].replace("-", "Outros")

    setores = sorted(set(ativos_df['Sector']))
    setores.insert(0, 'Todos')

    sector_filter = st.multiselect('Selecione o Setor', options=setores)

    if 'Todos' not in sector_filter:
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
            growth_data = get_financial_growth_data(ticker + '.SA')
            if growth_data:
                data.update(growth_data)
            data['Ticker'] = ticker
            fundamental_data.append(data)

        fundamental_df = pd.DataFrame(fundamental_data)
        ativos_df = ativos_df.merge(fundamental_df, on='Ticker')

        # Filtrar ativos com informações necessárias
        ativos_df = ativos_df.dropna(subset=['P/L', 'P/VP', 'ROE', 'Volume', 'Price', 'revenue_growth', 'income_growth', 'debt_stability'])

        # Verificar se há ativos suficientes para continuar
        if len(ativos_df) < 10:
            st.error("Não há ativos suficientes com dados completos para realizar a análise. Por favor, tente novamente mais tarde.")
            return

        # Análise fundamentalista e de liquidez
        ativos_df['Score'] = (
            ativos_df['ROE'] / ativos_df['P/L'] +
            1 / ativos_df['P/VP'] +
            np.log(ativos_df['Volume'])
        )

        tickers_raw = ativos_df['Ticker'].apply(lambda x: x + '.SA').tolist()
        stock_data_raw = get_stock_data(tickers_raw)

        # Detecção de anomalias e cálculo de RSI
        for ticker in tickers_raw:
            price_anomalies = detect_price_anomalies(stock_data_raw[ticker])
            rsi = calculate_rsi(stock_data_raw[ticker])
            ativos_df.loc[ativos_df['Ticker'] == ticker[:-3], 'price_anomaly'] = price_anomalies.mean()
            ativos_df.loc[ativos_df['Ticker'] == ticker[:-3], 'rsi_anomaly'] = (rsi > 70).mean() + (rsi < 30).mean()

        # Calcular score ajustado
        ativos_df['Adjusted_Score'] = ativos_df.apply(calculate_adjusted_score, axis=1)
        ativos_df

        # Selecionar os top 10 ativos com base no score
        top_ativos = ativos_df.nlargest(10, 'Adjusted_Score')
        

        tickers = top_ativos['Ticker'].apply(lambda x: x + '.SA').tolist()
        status_text.text('Obtendo dados históricos...')
        stock_data = get_stock_data(tickers)

        # Verificar se os dados históricos foram obtidos com sucesso
        if stock_data.empty:
            st.error("Não foi possível obter dados históricos. Por favor, tente novamente mais tarde.")
            return

        # Calcular rentabilidade acumulada
        cumulative_returns = [get_cumulative_return(ticker) for ticker in tickers]
        top_ativos['Rentabilidade Acumulada (5 anos)'] = cumulative_returns

        st.subheader('Top 10 BDRs Recomendados')
        st.dataframe(top_ativos[['Ticker', 'Sector', 'P/L', 'P/VP', 'ROE', 'Volume', 'Price', 'Score', 'Adjusted_Score','revenue_growth','income_growth','debt_stability','Rentabilidade Acumulada (5 anos)']])

        # Otimização de portfólio
        returns = calculate_returns(stock_data)

        # Verificar se há retornos válidos para continuar
        if returns.empty:
            st.error("Não foi possível calcular os retornos dos ativos. Por favor, tente novamente mais tarde.")
            return

        # Calcular rentabilidade acumulada
        cumulative_returns = [get_cumulative_return(ticker) for ticker in tickers]
        top_ativos['Rentabilidade Acumulada (5 anos)'] = cumulative_returns

        # Otimização de portfólio
        returns = calculate_returns(stock_data)

        # Verificar se há retornos válidos para continuar
        if returns.empty:
            st.error("Não foi possível calcular os retornos dos ativos. Por favor, tente novamente mais tarde.")
            return

        global risk_free_rate
        risk_free_rate = 0.05  # 5% como exemplo, ajuste conforme necessário

        status_text.text('Otimizando portfólio...')
        try:
            optimal_weights = optimize_portfolio(returns, risk_free_rate)
            # Ajustar pesos com base nas anomalias
            anomaly_scores = calculate_anomaly_scores(returns)
            adjusted_weights = adjust_weights_for_anomalies(optimal_weights, anomaly_scores)
        except Exception as e:
            st.error(f"Erro ao otimizar o portfólio: {e}")
            return

        st.subheader('Alocação Ótima do Portfólio')
        allocation_data = []
        for ticker, weight in zip(tickers, adjusted_weights):
            price = top_ativos.loc[top_ativos['Ticker'] == ticker[:-3], 'Price'].values[0]
            allocated_value = weight * invest_value
            shares = allocated_value / price
            cumulative_return = top_ativos.loc[top_ativos['Ticker'] == ticker[:-3], 'Rentabilidade Acumulada (5 anos)'].values[0]
            allocation_data.append({
                'Ticker': ticker,
                'Peso': f"{weight:.2%}",
                'Valor Alocado': f"R$ {allocated_value:.2f}",
                'Quantidade de Ações': f"{shares:.2f}",
                'Rentabilidade Acumulada (5 anos)': f"{cumulative_return:.2%}" if cumulative_return is not None else "N/A"
            })

        allocation_df = pd.DataFrame(allocation_data)
        st.table(allocation_df)

        portfolio_return, portfolio_volatility = portfolio_performance(adjusted_weights, returns)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        st.subheader('Métricas do Portfólio')
        st.write(f"Retorno Anual Esperado: {portfolio_return:.2%}")
        st.write(f"Volatilidade Anual: {portfolio_volatility:.2%}")
        st.write(f"Índice de Sharpe: {sharpe_ratio:.2f}")

        # Gerar e exibir o gráfico de dispersão
        status_text.text('Gerando gráfico da fronteira eficiente...')
        fig = plot_efficient_frontier(returns, adjusted_weights)
        st.plotly_chart(fig)

        # Exibir informações sobre anomalias detectadas
        st.subheader('Análise de Anomalias')
        anomaly_data = []
        for ticker in tickers:
            price_anomalies = detect_price_anomalies(stock_data[ticker])
            rsi = calculate_rsi(stock_data[ticker])
            rsi_anomalies = (rsi > 70) | (rsi < 30)
            anomaly_data.append({
                'Ticker': ticker[:-3],
                'Anomalias de Preço (%)': f"{price_anomalies.mean()*100:.2f}%",
                'Anomalias de RSI (%)': f"{rsi_anomalies.mean()*100:.2f}%"
            })
        
        anomaly_df = pd.DataFrame(anomaly_data)
        st.table(anomaly_df)

        st.write("As anomalias de preço indicam movimentos incomuns nos preços dos ativos, enquanto as anomalias de RSI indicam períodos de sobrecompra ou sobrevenda.")

        status_text.text('Análise concluída!')
        progress_bar.progress(100)

def display_summary():
    st.header("Lógica")

    resumo = """
    ***Racional do Código para Seleção de Ativos e Alocação de Investimentos***
    
    ***Objetivo:***
    O código foi desenvolvido para ajudá-lo a escolher os melhores ativos BDRs (Brazilian Depositary Receipts) e alocar seus investimentos de forma eficiente, maximizando o retorno e minimizando o risco.
    
    ***Importação de Dados:***
    Primeiramente, o código importa dados históricos dos preços dos ativos. Esses dados são essenciais para calcular retornos e volatilidade, entre outros indicadores financeiros.
    
    ***Cálculo de Indicadores Financeiros:***
    Retorno Médio: Calcula-se a média dos retornos diários ou mensais dos ativos ao longo de um período de tempo. O retorno médio é uma medida de desempenho histórico do ativo.
    Retorno Médio = (1/N) * Σ(Ri), onde Ri é o retorno no período i e N é o número total de períodos.
    
    ***Volatilidade:*** Mede a dispersão dos retornos dos ativos. A volatilidade é calculada como o desvio padrão dos retornos.
    Volatilidade = sqrt((1/(N-1)) * Σ(Ri - Retorno Médio)^2)
    
    ***Índice de Sharpe:*** Avalia a relação entre o retorno esperado e a volatilidade do ativo. Um índice de Sharpe mais alto indica uma melhor relação risco-retorno.
    Índice de Sharpe = (Retorno Médio - Rf) / Volatilidade, onde Rf é a taxa livre de risco.
    
    Valuation (P/L - Preço/Lucro): É uma métrica que relaciona o preço da ação com o lucro por ação. Um P/L mais baixo pode indicar que a ação está subvalorizada.
    P/L = Preço da Ação / Lucro por Ação
    
    ***Liquidez:*** Mede a facilidade de compra e venda do ativo sem afetar seu preço. A liquidez é importante para garantir que você possa entrar e sair de posições facilmente.
    
    ***Seleção dos Melhores Ativos:***
    O código classifica os ativos com base nos indicadores calculados, priorizando aqueles com maior índice de Sharpe, boa valuation (P/L), e alta liquidez.
    
    ***Simulação de Alocação de Capital:***
    Utilizando os ativos selecionados, o código aplica métodos de otimização de carteira, como a Fronteira Eficiente de Markowitz. Esse método busca encontrar a combinação de ativos que oferece o maior retorno esperado para um dado nível de risco.
    Minimizar σp^2 = Σ(wi * wj * σij), Sujeito a Σ(wi) = 1, onde wi é a proporção do capital alocada no ativo i e σij é a covariância entre os retornos dos ativos i e j.
    
    ***Recomendação de Investimento:***
    O código gera uma recomendação detalhada de como alocar seu capital nos ativos selecionados, mostrando a quantidade a ser investida em cada ativo. Ele também apresenta gráficos para visualizar o retorno esperado, o risco, e a alocação de capital.
    """
    st.markdown(resumo)

if __name__ == "__main__":
    main()
    display_summary()


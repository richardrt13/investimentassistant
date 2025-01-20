import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from requests.exceptions import ConnectionError
from statsmodels.tsa.arima.model import ARIMA
import warnings
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
warnings.filterwarnings('ignore')
from pymongo import MongoClient
import logging
import time
import google.generativeai as genai
from typing import Optional, Dict, List
import yaml
import re
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import bcrypt
import time
import logging
from streamlit_cookies_manager import EncryptedCookieManager
import uuid
from data_handling import get_fundamental_data, get_stock_data, get_historical_prices, get_financial_growth_data
from ai_features import PortfolioAnalyzer
from portfolio_calculation import FinancialAnalysis

financial_analyzer = FinancialAnalysis()

# MongoDB Atlas connection
mongo_uri = st.secrets["mongo_uri"]
client = MongoClient(mongo_uri)
db = client['StockIdea']
collection = db['transactions']
prices_collection = db['historical_prices']
stocks_collection = db['stocks']
users_collection = db['users']


# Função para hashear senhas
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Função para verificar senha
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Página de login
cookies = EncryptedCookieManager(prefix="app_", password="heroaiinvestment")  
if not cookies.ready():
    st.stop()
def login_page():
    st.title("Login")

    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        user = users_collection.find_one({"username": username})
        if user and check_password(password, user["password"]):
            cookies["authenticated"] = "true"
            cookies["user_name"] = user["name"]
            cookies["user_id"] = user["user_id"]  
            cookies.save()  # Salva alterações nos cookies
            st.rerun()  # Recarrega a página para atualizar o estado
        else:
            st.error("Usuário ou senha incorretos.")
    return False, None  # Retorna False e None por padrão
    
# Página de registro
def register_page():
    st.title("Registro")

    name = st.text_input("Nome Completo")
    username = st.text_input("Nome de Usuário")
    email = st.text_input("Email")
    password = st.text_input("Senha de Acesso", type="password")
    password_confirm = st.text_input("Confirme a Senha", type="password")

    if st.button("Registrar"):
        if password != password_confirm:
            st.error("As senhas não coincidem.")
        elif users_collection.find_one({"username": username}):
            st.error("Usuário já existe.")
        else:
            # Gera um user_id único
            user_id = str(uuid.uuid4())
            
            # Hash da senha
            hashed_password = hash_password(password)
            
            # Cria o documento do usuário
            user = {
                "user_id": user_id,  # Adiciona o user_id
                "name": name,
                "username": username,
                "email": email,
                "password": hashed_password,
            }
            
            # Insere o usuário no banco de dados
            users_collection.insert_one(user)
            st.success("Usuário registrado com sucesso! Faça login para continuar.")

@st.cache_data(ttl=3600)
def load_assets():
    assets = pd.DataFrame(list(stocks_collection.find()))
    if '_id' in assets.columns:
        assets = assets.drop('_id', axis=1)
    return assets

# Function to initialize the database
def init_db():
    if 'transactions' not in db.list_collection_names():
        collection.create_index([('Date', 1), ('Ticker', 1), ('Action', 1), ('Quantity', 1), ('Price', 1)])

# Function to log transactions
def log_transaction(date, ticker, action, quantity, price, user_id):
    transaction = {
        'Date': date,
        'Ticker': ticker,
        'Action': action,
        'Quantity': quantity,
        'Price': price,
        'user_id': user_id  # Adiciona user_id na transação
    }
    collection.insert_one(transaction)
    st.success('Transação registrada com sucesso!')

# Modificar as funções de compra e venda
def buy_stock(date, ticker, quantity, price, user_id):
    log_transaction(date, ticker, 'BUY', quantity, price, user_id)

def sell_stock(date, ticker, quantity, price, user_id):
    log_transaction(date, ticker, 'SELL', quantity, price, user_id)


def get_portfolio_performance(user_id):
    # Fetch transactions for specific user
    transactions = pd.DataFrame(list(collection.find({'user_id': user_id})))
    #transactions = pd.DataFrame(list(collection.find())) 
    transactions
    
    if transactions.empty:
        return pd.DataFrame(), pd.Series()
    
    # Convert date and sort
    transactions['Date'] = pd.to_datetime(transactions['Date'])
    transactions = transactions.sort_values('Date')
    
    # Group transactions by ticker to calculate final positions
    portfolio_summary = transactions.groupby('Ticker').apply(
        lambda group: pd.Series({
            'Total_Quantity': group[group['Action'] == 'BUY']['Quantity'].sum() - 
                              group[group['Action'] == 'SELL']['Quantity'].sum(),
            'Total_Invested': (group[group['Action'] == 'BUY']['Quantity'] * group[group['Action'] == 'BUY']['Price']).sum() - 
                              (group[group['Action'] == 'SELL']['Quantity'] * group[group['Action'] == 'SELL']['Price']).sum()
        })
    ).reset_index()
    
    # Filter out stocks with zero quantity
    active_portfolio = portfolio_summary[portfolio_summary['Total_Quantity'] > 0]
    
    # Fetch current prices for active stocks
    end_date_raw = datetime.now()
    start_date_raw = transactions['Date'].min()
    end_date = end_date_raw.strftime('%Y-%m-%d')
    start_date = start_date_raw.strftime('%Y-%m-%d')
    
    # Create a DataFrame to store daily portfolio values
    daily_values = pd.DataFrame()
    
    for _, stock in active_portfolio.iterrows():
        ticker = stock['Ticker']
        quantity = stock['Total_Quantity']
        
        try:
            ticker_prices = get_historical_prices(ticker, start_date, end_date)
            ticker_prices = ticker_prices.set_index('date')['Close']
            ticker_prices = ticker_prices.dropna()
            
            daily_values[ticker] = ticker_prices * quantity
        except Exception as e:
            print(f"Could not fetch prices for {ticker}: {e}")
    
    invested_values = active_portfolio.set_index('Ticker')['Total_Invested']
    
    return daily_values, invested_values

@st.cache_data(ttl=3600)
def get_ibovespa_data(start_date, end_date):
    """
    Fetch Ibovespa historical data from MongoDB instead of yfinance
    
    Parameters:
    start_date (datetime): Start date for historical data
    end_date (datetime): End date for historical data
    
    Returns:
    pandas.Series: Series with Ibovespa returns
    """
    # Convert dates to string format
    start_date_str = start_date
    end_date_str = end_date
    
    # Query MongoDB for Ibovespa data
    query = {
        'ticker': '^BVSP',
        'date': {
            '$gte': start_date_str,
            '$lte': end_date_str
        }
    }
    
    # Fetch data from MongoDB
    cursor = prices_collection.find(
        query,
        {'_id': 0, 'date': 1, 'Close': 1}
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        return pd.Series()
        
    # Convert date string to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    
    # Calculate return percentage
    ibov_return = (df.set_index('date')['Close'] / df.set_index('date')['Close'].iloc[0] - 1) * 100
    
    return ibov_return

def calculate_portfolio_metrics(portfolio_data, invested_value):
    total_invested = invested_value.sum()
    current_value = portfolio_data.iloc[-1].sum()
    total_return = ((current_value - total_invested) / total_invested) * 100
    return total_invested, current_value, total_return
            

def calculate_optimal_contribution_with_genai(portfolio_data, invested_value, contribution_amount):
    """
    Uses GenAI to analyze portfolio and recommend optimal contribution allocation
    """
    try:
        genai.configure(api_key=st.secrets["api_key"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare portfolio summary
        portfolio_summary = {
            "total_invested": invested_value.sum(),
            "current_value": portfolio_data.iloc[-1].sum(),
            "assets": []
        }

        for ticker in portfolio_data.columns:
            # Get current stock data
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            
            # Get fundamental data
            fundamental_data = get_fundamental_data(ticker)
            growth_data = get_financial_growth_data(ticker)
            
            asset_data = {
                "ticker": ticker,
                "invested": invested_value[ticker],
                "current_value": portfolio_data[ticker].iloc[-1],
                "return": ((portfolio_data[ticker].iloc[-1] / invested_value[ticker]) - 1) * 100,
                "weight": portfolio_data[ticker].iloc[-1] / portfolio_data.iloc[-1].sum() * 100,
                "price": current_price,
                "fundamentals": fundamental_data,
                "growth": growth_data
            }
            portfolio_summary["assets"].append(asset_data)

        # Create prompt for GenAI
        prompt = f"""Analise a seguinte carteira de investimentos e com o objetivo de manter a carteira balenceada e otimizar os retornos futuros 
        recomende a melhor forma de alocar um aporte de R$ {contribution_amount:.2f}:

        Resumo da Carteira:
        Total Investido: R$ {portfolio_summary['total_invested']:.2f}
        Valor Atual: R$ {portfolio_summary['current_value']:.2f}
        
        Composição Atual:
        """
        for asset in portfolio_summary["assets"]:
            prompt += f"""
            {asset['ticker']}:
            - Valor Investido: R$ {asset['invested']:.2f}
            - Valor Atual: R$ {asset['current_value']:.2f}
            - Retorno: {asset['return']:.2f}%
            - Peso Atual: {asset['weight']:.2f}%
            - Preço Atual: R$ {asset['price']:.2f}
            - P/L: {asset['fundamentals'].get('P/L', 'N/A')}
            - ROE: {asset['fundamentals'].get('ROE', 'N/A')}
            - Crescimento Receita: {asset['growth'].get('revenue_growth', 'N/A') if asset['growth'] else 'N/A'}
            """

        prompt += """
        Por favor, forneça:
        1. Uma recomendação detalhada de como alocar o aporte entre os ativos existentes
        2. Justificativa para cada alocação sugerida
        3. Número específico de ações a comprar de cada ativo
        4. Considerações sobre balanceamento da carteira
        5. Análise de risco e retorno esperado
        
        Responda em português e no seguinte formato:
        ALOCAÇÃO RECOMENDADA:
        - Ticker: [número de ações] ações a R$ [preço atual] = R$ [valor total]
        ...
        
        JUSTIFICATIVA:
        [explicação detalhada]
        
        CONSIDERAÇÕES:
        [considerações adicionais]"""

        # Generate recommendation
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Erro ao gerar recomendação: {e}"


def allocate_portfolio_integer_shares(invest_value, prices, weights):
    allocation = {}
    remaining_value = invest_value
    
    # Ordenar os ativos por peso, do maior para o menor
    sorted_assets = sorted(zip(weights, prices.index), reverse=True)
    
    for weight, ticker in sorted_assets:
        price = prices[ticker]
        target_value = invest_value * weight
        shares = int(target_value / price)  # Arredonda para baixo para obter um número inteiro de ações
        
        if shares > 0 and price * shares <= remaining_value:
            allocation[ticker] = shares
            remaining_value -= price * shares
    
    # Tenta alocar o valor restante em mais ações, se possível
    for weight, ticker in sorted_assets:
        price = prices[ticker]
        if price <= remaining_value:
            additional_shares = int(remaining_value / price)
            if additional_shares > 0:
                allocation[ticker] = allocation.get(ticker, 0) + additional_shares
                remaining_value -= price * additional_shares
    
    return allocation, remaining_value

# Configuração básica para logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def get_asset_recommendations(top_ativos, tickers, stock_data, returns, risk_free_rate, portfolio_return, portfolio_volatility, anomaly_df, invest_value):
    try:
        start_time = time.time()
        logging.info("Iniciando a execução da função.")

        # Inicialização do modelo
        model_start = time.time()
        genai.configure(api_key=st.secrets["api_key"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        logging.info(f"Modelo inicializado em {time.time() - model_start:.2f} segundos.")

        # Preparação de dados do portfólio
        metrics_start = time.time()
        portfolio_metrics = {
            "return": portfolio_return * 100,
            "volatility": portfolio_volatility * 100,
            "sharpe": (portfolio_return - risk_free_rate) / portfolio_volatility
        }
        logging.info(f"Métricas do portfólio calculadas em {time.time() - metrics_start:.2f} segundos.")

        # Criação de resumos de ativos
        assets_start = time.time()
        assets = []
        for ticker in tickers:
            ticker_start = time.time()
            base_ticker = ticker.replace('.SA', '')
            asset_data = top_ativos[top_ativos['symbol'] == base_ticker].iloc[0]
            anomaly_data = anomaly_df[anomaly_df['symbol'] == ticker].iloc[0]
            
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            
            assets.append({
                "ticker": base_ticker,
                "sector": asset_data['sector'],
                "industry": asset_data['industry'],
                "preco_atual": current_price,
                "fundamentals": {
                    "pe_ratio": asset_data['P/L'],
                    "pb_ratio": asset_data['P/VP'],
                    "roe": asset_data['ROE'],
                    "roic": asset_data['ROIC'],
                    "dividend_yield": asset_data['Dividend Yield']
                },
                "growth": {
                    "revenue": asset_data['revenue_growth'],
                    "income": asset_data['income_growth'],
                    "debt_stability": asset_data['debt_stability']
                },
                "risk": {
                    "price_anomalies": anomaly_data['price_anomaly'],
                    "rsi_anomalies": anomaly_data['rsi_anomaly'],
                    "returns_volatility": returns[ticker].std() * np.sqrt(252)
                }
            })
            logging.info(f"Ticker {ticker} processado em {time.time() - ticker_start:.2f} segundos.")

        logging.info(f"Resumo de ativos criado em {time.time() - assets_start:.2f} segundos.")

        # Preparação do prompt
        prompt_start = time.time()
        prompt = f"""Analise os seguintes ativos e selecione os melhores para investir pensando em retornos futuros:

        Ativos Selecionados:
        {str(assets)}

        Valor que quero investir:
        {str(invest_value)}

        Por favor, forneça:
        1. Uma tabela com os ativos selecionados, quantidade sugerida de compra, valor para investir e justificativa do investimento. 
        Se certifique de que a soma total do investimento não supere o valor que eu quero investir
        Responda em português e de forma estruturada."""
        logging.info(f"Prompt preparado em {time.time() - prompt_start:.2f} segundos.")

        # Geração de conteúdo
        generation_start = time.time()
        response = model.generate_content(prompt)
        logging.info(f"Resposta gerada em {time.time() - generation_start:.2f} segundos.")

        total_time = time.time() - start_time
        logging.info(f"Execução total da função concluída em {total_time:.2f} segundos.")
        return response.text

    except Exception as e:
        logging.error(f"Erro ao gerar recomendações: {e}")
        return f"Erro ao gerar recomendações: {e}"

def mask_monetary_value(value, show_values):
    """
    Masks or shows monetary values based on user preference
    
    Parameters:
    value (float): The monetary value to mask/show
    show_values (bool): Whether to show actual values
    
    Returns:
    str: Formatted value string or masked string
    """
    if show_values:
        return f"R$ {value:.2f}"
    return "R$ ****,**"

def portfolio_tracking(user_id):
    st.title('Acompanhamento da Carteira')

    show_values = st.sidebar.toggle('Mostrar Valores', value=True)
    
    # Initialize database
    init_db()
    analyzer = PortfolioAnalyzer()

    # Get all assets
    assets_df = load_assets()
    tickers = assets_df.apply(lambda row: row['symbol'] + '.SA' if row['country'].lower() == 'brazil' else row['symbol'], axis=1).tolist()
    
    # Transaction input
    st.subheader('Registrar Transação')
    col1, col2, col3 = st.columns(3)
    with col1:
        transaction_date = st.date_input('Data da Transação', value=datetime.now().date())
    with col2:
        transaction_ticker = st.selectbox('Ticker', options=tickers)
    with col3:
        transaction_action = st.selectbox('Ação', options=['Compra', 'Venda'])
    col4, col5 = st.columns(2)
    with col4:
        transaction_quantity = st.number_input('Quantidade', min_value=1, value=1, step=1)
    with col5:
        transaction_price = st.number_input('Preço', min_value=0.01, value=1.00, step=0.01)

    if st.button('Registrar Transação'):
        transaction_date_str = transaction_date.strftime('%Y-%m-%d %H:%M:%S')
        if transaction_action == 'Compra':
            buy_stock(transaction_date_str, transaction_ticker, transaction_quantity, transaction_price, user_id)
        else:
            sell_stock(transaction_date_str, transaction_ticker, transaction_quantity, transaction_price, user_id)

    # Display portfolio performance
    st.subheader('Desempenho da Carteira')
    portfolio_data, invested_value = get_portfolio_performance(user_id)
    
    if not portfolio_data.empty:
        total_invested, current_value, total_return = calculate_portfolio_metrics(portfolio_data, invested_value)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Valor Total Investido", mask_monetary_value(total_invested, show_values))
        col2.metric("Valor Atual da Carteira", mask_monetary_value(current_value, show_values))
        col3.metric("Retorno Total", f"{total_return:.2f}%")

        # Calculate returns for each asset
        asset_returns = {}
        for ticker in portfolio_data.columns:
            initial_value = invested_value[ticker]
            current_value = portfolio_data[ticker].iloc[-1]
            if initial_value > 0:
                asset_return = ((current_value - initial_value) / initial_value) * 100
                asset_returns[ticker] = {
                    'return': asset_return,
                    'current_value': current_value
                }

        # Sort assets by return
        sorted_assets = sorted(asset_returns.items(), key=lambda x: x[1]['return'], reverse=True)

        # Create bar chart for asset returns
        fig_asset_returns = go.Figure()
        tickers = []
        returns = []
        current_values = []
        for ticker, data in sorted_assets:
            tickers.append(ticker)
            returns.append(data['return'])
            current_values.append(data['current_value'])

        # Modify the hover text based on show_values setting
        hover_text = [
            f"{r:.2f}%<br>{mask_monetary_value(v, show_values)}" 
            for r, v in zip(returns, current_values)
        ]

        fig_asset_returns.add_trace(go.Bar(
            x=tickers,
            y=returns,
            text=hover_text,
            textposition='auto',
            name='Retorno Acumulado'
        ))

        fig_asset_returns.update_layout(
            title='Retorno Acumulado por Ativo',
            xaxis_title='Ativo',
            yaxis_title='Retorno Acumulado (%)',
            yaxis_tickformat = '.2f%'
        )

        st.plotly_chart(fig_asset_returns)

        # Calculate daily portfolio value
        daily_portfolio_value = portfolio_data.sum(axis=1)

        # Calculate daily returns
        daily_returns = daily_portfolio_value.pct_change()

        # Calculate cumulative returns
        portfolio_cumulative_returns = (1 + daily_returns).cumprod() - 1
        portfolio_cumulative_returns = portfolio_cumulative_returns * 100

        # Ensure the final return matches the total return
        portfolio_cumulative_returns = portfolio_cumulative_returns * (total_return / portfolio_cumulative_returns.iloc[-1])

        # Get Ibovespa data
        ibovespa_start_date = portfolio_data.index[0].strftime('%Y-%m-%d')
        ibovespa_end_date = portfolio_data.index[-1].strftime('%Y-%m-%d')
        ibov_return = get_ibovespa_data(ibovespa_start_date, ibovespa_end_date)

        # Create figure for cumulative returns comparison with conditional hover template
        fig_returns = go.Figure()
        
        # Modify hover template based on show_values setting
        if show_values:
            portfolio_hover = 'Data: %{x}<br>Valor: R$ %{customdata:.2f}<br>Retorno: %{y:.2f}%'
        else:
            portfolio_hover = 'Data: %{x}<br>Valor: R$ ****,**<br>Retorno: %{y:.2f}%'

        fig_returns.add_trace(go.Scatter(
            x=portfolio_cumulative_returns.index, 
            y=portfolio_cumulative_returns.values,
            customdata=daily_portfolio_value.values,
            mode='lines', 
            name='Carteira',
            hovertemplate=portfolio_hover
        ))

        fig_returns.add_trace(go.Scatter(
            x=ibov_return.index, 
            y=ibov_return.values,
            mode='lines', 
            name='Ibovespa',
            hovertemplate='Data: %{x}<br>Retorno Ibovespa: %{y:.2f}%'
        ))

        fig_returns.update_layout(
            title='Comparação de Retorno Percentual Acumulado: Carteira vs Ibovespa',
            xaxis_title='Data',
            yaxis_title='Retorno Acumulado (%)',
            yaxis_tickformat = '.2f%',
            hovermode='x unified'
        )
        st.plotly_chart(fig_returns)

    else:
        st.write("Não há transações registradas ainda.")

    st.subheader('Aporte Inteligente na Carteira')
    contribution_amount = st.number_input('Valor do Aporte (R$)', min_value=0.01, value=1000.00, step=0.01)

    if st.button('Calcular Distribuição Ótima do Aporte'):
        portfolio_data, invested_value = get_portfolio_performance(user_id)
        if not portfolio_data.empty:
            with st.spinner('Gerando recomendação personalizada...'):
                recommendation = calculate_optimal_contribution_with_genai(portfolio_data, invested_value, contribution_amount)
                # Mask values in recommendation if needed
                if not show_values:
                    recommendation = re.sub(r'R\$ \d+[.,]\d+', 'R$ ****,**', recommendation)
                st.markdown(recommendation)
        else:
            st.write("Não há dados suficientes para calcular a distribuição do aporte.")
    
    portfolio_data, invested_value = get_portfolio_performance(user_id)
    
    if not portfolio_data.empty:
        # Botão para gerar análise
        if st.button('Gerar Análise da Carteira'):
            with st.spinner('Gerando análise detalhada...'):
                analysis = analyzer.analyze_portfolio(portfolio_data, invested_value)
                if analysis:
                    # Mask values in analysis if needed
                    if not show_values:
                        analysis = re.sub(r'R\$ \d+[.,]\d+', 'R$ ****,**', analysis)
                    st.markdown(analysis)
                    
        # Botão para sugestões de otimização
        if st.button('Gerar Sugestões de Otimização'):
            with st.spinner('Gerando sugestões de otimização...'):
                market_data = {
                    'ibovespa_return': get_ibovespa_ytd_return(),
                    'selic': 11.25
                }
                suggestions = analyzer.get_optimization_suggestions(portfolio_data, market_data)
                if suggestions:
                    # Mask values in suggestions if needed
                    if not show_values:
                        suggestions = re.sub(r'R\$ \d+[.,]\d+', 'R$ ****,**', suggestions)
                    st.markdown(suggestions)
    else:
        st.write("Não há dados suficientes para análise. Registre suas transações primeiro.")

def get_ibovespa_ytd_return():
    """
    Calcula o retorno do Ibovespa no ano
    """
    try:
        start_date = datetime(datetime.now().year, 1, 1)
        ibov_data = yf.download('^BVSP', start=start_date)['Adj Close']
        ytd_return = ((ibov_data[-1] / ibov_data[0]) - 1) * 100
        return round(ytd_return, 2)
    except:
        return None


def main():
    if "authenticated" in cookies and cookies["authenticated"] == "true":
        st.sidebar.success(f"Bem-vindo(a), {cookies['user_name']}!")
        
        # Get user_id from cookies or session state
        user_id = cookies.get("user_id", "")
        
        
        if st.sidebar.button("Logout"):
            cookies["authenticated"] = "false"
            cookies["user_name"] = ""
            cookies["user_id"] = ""
            cookies.save()
            st.rerun()

        page = st.sidebar.radio('Selecione uma página', 
                              ['Recomendação de Ativos', 'Acompanhamento da Carteira'])
    
        if page == 'Recomendação de Ativos':
    
            ativos_df = load_assets()
            
            ativos_df['sector'] = ativos_df['sector'].fillna('Não especificado')
            ativos_df['industry'] = ativos_df['industry'].fillna('Não especificado')
            ativos_df['country'] = ativos_df['country'].fillna('Não especificado')
            ativos_df['type'] = ativos_df['type'].fillna('Não especificado')
    
            # Valores iniciais dos filtros
            todos_opcao = 'Todos'
    
            # Criação dos filtros mostrando todas as opções disponíveis do DataFrame original
            country_filter = st.selectbox(
                'Selecione o País', 
                options=[todos_opcao] + sorted(ativos_df['country'].unique())
            )
    
            type_filter = st.selectbox(
                'Selecione a Categoria', 
                options=[todos_opcao] + sorted(ativos_df['type'].unique())
            )
    
            sector_filter = st.selectbox(
                'Selecione o Setor', 
                options=[todos_opcao] + sorted(ativos_df['sector'].unique())
            )
    
            industry_filter = st.selectbox(
                'Selecione a Indústria', 
                options=[todos_opcao] + sorted(ativos_df['industry'].unique())
            )
    
            # Aplicar todos os filtros de uma vez
            filtered_df = ativos_df.copy()
    
            if country_filter != todos_opcao:
                filtered_df = filtered_df[filtered_df['country'] == country_filter]
        
            if type_filter != todos_opcao:
                filtered_df = filtered_df[filtered_df['type'] == type_filter]
        
            if sector_filter != todos_opcao:
                filtered_df = filtered_df[filtered_df['sector'] == sector_filter]
        
            if industry_filter != todos_opcao:
                filtered_df = filtered_df[filtered_df['industry'] == industry_filter]
        
    
            ativos_df = filtered_df
    
           
            invest_value = st.number_input('Valor a ser investido (R$)', min_value=100.0, value=10000.0, step=100.0)
        
            if st.button('Gerar Recomendação'):
                progress_bar = st.progress(0)
                status_text = st.empty()
        
                # Obter dados fundamentalistas
                fundamental_data = []
                for i, ticker in enumerate(ativos_df['symbol']):
                    status_text.text(f'Carregando dados para {ticker}...')
                    progress_bar.progress((i + 1) / len(ativos_df))
                    
                    # Adiciona o sufixo .SA apenas se o país for Brazil
                    ticker_symbol = ticker + '.SA' if ativos_df.iloc[i]['country'].lower() == 'brazil' else ticker
                    
                    data = get_fundamental_data(ticker_symbol)
                    growth_data = get_financial_growth_data(ticker_symbol)
                    
                    if growth_data:
                        data.update(growth_data)
                    
                    data['symbol'] = ticker  # Mantém o ticker original sem o sufixo
                    fundamental_data.append(data)
    
        
                fundamental_df = pd.DataFrame(fundamental_data)
                ativos_df = ativos_df.merge(fundamental_df, on='symbol')
        
                # Filtrar ativos com informações necessárias
                ativos_df = ativos_df.dropna(subset=['P/L', 'P/VP', 'ROE', 'ROIC', 'Dividend Yield','Volume', 'Price', 'revenue_growth', 'income_growth', 'debt_stability'])
          
                
             
                #Filtrar ativos com boa liquidez
                #ativos_df = ativos_df[ativos_df.Volume > ativos_df.Volume.quantile(.25)]
        
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
        
                tickers_raw = ativos_df.apply(lambda row: row['symbol'] + '.SA' if row['country'].lower() == 'brazil' else row['symbol'], axis=1).tolist()
            
                
                stock_data_raw = get_stock_data(tickers_raw)
        
                # Detecção de anomalias e cálculo de RSI
                for ticker in tickers_raw:
                    price_anomalies = financial_analyzer.detect_price_anomalies(stock_data_raw[ticker])
                    rsi = financial_analyzer.calculate_rsi(stock_data_raw[ticker])
                    ativos_df.loc[ativos_df['symbol'] == ticker[:-3], 'price_anomaly'] = price_anomalies.mean()
                    ativos_df.loc[ativos_df['symbol'] == ticker[:-3], 'rsi_anomaly'] = (rsi > 70).mean() + (rsi < 30).mean()
        
                # Calcular score ajustado
                cumulative_returns_raw = [financial_analyzer.get_cumulative_return(ticker) for ticker in tickers_raw]
                ativos_df['Rentabilidade Acumulada (5 anos)'] = cumulative_returns_raw
                optimized_weights = financial_analyzer.optimize_weights(ativos_df)
                ativos_df['Adjusted_Score'] = ativos_df.apply(lambda row: financial_analyzer.calculate_adjusted_score(row, optimized_weights), axis=1)
        
                # Selecionar os top 10 ativos com base no score
                top_ativos = ativos_df.nlargest(10, 'Adjusted_Score')
                growth_data = top_ativos[['revenue_growth', 'income_growth']].mean(axis=1).values
                quality_data = top_ativos['ROIC'].values
    
        
                tickers = top_ativos.apply(lambda row: row['symbol'] + '.SA' if row['country'].lower() == 'brazil' else row['symbol'], axis=1).tolist()
                status_text.text('Obtendo dados históricos...')
                stock_data = get_stock_data(tickers)
        
                # Verificar se os dados históricos foram obtidos com sucesso
                if stock_data.empty:
                    st.error("Não foi possível obter dados históricos. Por favor, tente novamente mais tarde.")
                    return
        
        
                st.subheader('Top 10 BDRs Recomendados')
                st.dataframe(top_ativos[['symbol', 'sector','industry', 'P/L', 'P/VP', 'ROE', 'ROIC', 'Dividend Yield','Volume', 'Price', 'Score', 'Adjusted_Score','revenue_growth','income_growth','debt_stability','Rentabilidade Acumulada (5 anos)']])
        
                # Otimização de portfólio
                returns = financial_analyzer.calculate_returns(stock_data)
        
                # Verificar se há retornos válidos para continuar
                if returns.empty:
                    st.error("Não foi possível calcular os retornos dos ativos. Por favor, tente novamente mais tarde.")
                    return
        
                # Calcular rentabilidade acumulada
                cumulative_returns = [financial_analyzer.get_cumulative_return(ticker) for ticker in tickers]
                top_ativos['Rentabilidade Acumulada (5 anos)'] = cumulative_returns
        
                # Otimização de portfólio
                returns = financial_analyzer.calculate_returns(stock_data)
        
                # Verificar se há retornos válidos para continuar
                if returns.empty:
                    st.error("Não foi possível calcular os retornos dos ativos. Por favor, tente novamente mais tarde.")
                    return
        
                global risk_free_rate
                risk_free_rate = 0.1
        
                status_text.text('Otimizando portfólio...')
                try:
                    optimal_weights = financial_analyzer.optimize_portfolio(returns)
                    # Ajustar pesos com base nas anomalias
                    anomaly_scores = financial_analyzer.calculate_anomaly_scores(returns)
                    adjusted_weights = financial_analyzer.adjust_weights_for_anomalies(optimal_weights, anomaly_scores)
                except Exception as e:
                    st.error(f"Erro ao otimizar o portfólio: {e}")
                    return
    
    
                # Exibir informações sobre anomalias detectadas
                #st.subheader('Análise de Anomalias')
                anomaly_data = []
                for ticker in tickers:
                    price_anomalies = financial_analyzer.detect_price_anomalies(stock_data[ticker])
                    rsi = financial_analyzer.calculate_rsi(stock_data[ticker])
                    rsi_anomalies = (rsi > 70) | (rsi < 30)
                    anomaly_data.append({
                        'symbol': ticker,
                        'price_anomaly': round(price_anomalies.mean(),2),
                        'rsi_anomaly': round(rsi_anomalies.mean(),2)
                    })
    
                anomaly_df = pd.DataFrame(anomaly_data)
               
                portfolio_return, portfolio_volatility = financial_analyzer.portfolio_performance(adjusted_weights, returns)
                portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    
                prices = top_ativos.set_index('symbol')['Price']
                allocation, remaining_value = allocate_portfolio_integer_shares(invest_value, prices, adjusted_weights)
                
    
                st.subheader('Análise Detalhada da Recomendação')
                with st.spinner('Gerando análise detalhada...'):
                    recommendation = get_asset_recommendations(
                        top_ativos,
                        tickers,
                        stock_data,
                        returns,
                        risk_free_rate,
                        portfolio_return,
                        portfolio_volatility,
                        anomaly_df,
                        invest_value
                    )
                    st.markdown(recommendation)
            pass
        
        elif page == 'Acompanhamento da Carteira':
            portfolio_tracking(user_id)
    else:
        # Usuário não autenticado: exibe as abas de login e registro
        tab1, tab2 = st.tabs(["Login", "Registrar"])

        # Aba de Login
        with tab1:
            is_logged_in, user = login_page()
            if is_logged_in:
                st.rerun()  # Recarrega a página para atualizar o estado

        # Aba de Registrar
        with tab2:
            register_page()

if __name__ == "__main__":
    main()

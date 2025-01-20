import streamlit as st
import time
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pymongo import MongoClient

@st.cache_data(ttl=3600)
def get_fundamental_data(ticker, max_retries=3):
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Obter dados do balanço patrimonial e demonstração financeira
            balance_sheet = stock.balance_sheet
            financials = stock.financials

            # Calcular o ROIC
            if not balance_sheet.empty and not financials.empty:
                net_income = financials.loc['Net Income'].iloc[0]  # Último ano fiscal
                total_assets = balance_sheet.loc['Total Assets'].iloc[0]  # Último ano fiscal
                total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]  # Último ano fiscal
                cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]  # Último ano fiscal

                invested_capital = total_assets - total_liabilities - cash
                if invested_capital != 0:
                    roic = (net_income / invested_capital) * 100  # em percentagem
                else:
                    roic = np.nan
            else:
                roic = np.nan

            return {
                'P/L': info.get('trailingPE', np.nan),
                'P/VP': info.get('priceToBook', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'Volume': info.get('averageVolume', np.nan),
                'Price': info.get('currentPrice', np.nan),
                'ROIC': roic,
                'Dividend Yield': info.get('trailingAnnualDividendYield', np.nan),
                'Debt to Equity': info.get('debtToEquity', np.nan)
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                st.warning(f"Não foi possível obter dados para {ticker}. Erro: {e}")
                return {
                    'P/L': np.nan,
                    'P/VP': np.nan,
                    'ROE': np.nan,
                    'Volume': np.nan,
                    'Price': np.nan,
                    'ROIC': np.nan,
                    'Dividend Yield': np.nan,
                    'Debt to Equity': np.nan
                }
                
@st.cache_data(ttl=3600)
def get_stock_data(tickers, years=5, max_retries=3):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    

    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            return data
        except ConnectionError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                st.error(f"Erro ao obter dados históricos. Possível limite de requisição atingido. Erro: {e}")
                return pd.DataFrame()
                
def get_historical_prices(ticker, start_date, end_date):
    """
    Fetch historical price data from MongoDB instead of yfinance
    
    Parameters:
    ticker (str): Stock ticker symbol
    start_date (datetime): Start date for historical data
    end_date (datetime): End date for historical data
    
    Returns:
    pandas.DataFrame: DataFrame with date and adjusted close prices
    """
    # Convert dates to string format matching MongoDB
    start_date_str = start_date
    end_date_str = end_date
    
    # Query MongoDB for historical prices
    query = {
        'ticker': ticker,
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
        return df
        
    # Convert date string to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    
    return df

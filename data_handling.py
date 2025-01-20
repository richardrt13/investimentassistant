import streamlit as st
import time
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pymongo import MongoClient
import pandas as pd

mongo_uri = st.secrets["mongo_uri"]
client = MongoClient(mongo_uri)
db = client['StockIdea']
prices_collection = db['historical_prices']

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
    Fetch historical price data from MongoDB first, falling back to yfinance if needed
    
    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date for historical data in YYYY-MM-DD format
    end_date (str): End date for historical data in YYYY-MM-DD format
    
    Returns:
    pandas.DataFrame: DataFrame with date and adjusted close prices
    """
    try:
        # Convert dates to datetime for comparison
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        current_dt = pd.to_datetime(datetime.now().date())
        
        # Query MongoDB for historical prices
        query = {
            'ticker': ticker,
            'date': {
                '$gte': start_date,
                '$lte': end_date
            }
        }
        
        # Fetch data from MongoDB
        cursor = prices_collection.find(
            query,
            {'_id': 0, 'date': 1, 'Close': 1}
        )
        
        df = pd.DataFrame(list(cursor))
        
        # Check if MongoDB data exists and is up to date
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            latest_date = df['date'].max()
            latest_dt = pd.to_datetime(latest_date)
            
            # If data is current (updated today) and complete, return MongoDB data
            if latest_dt.date() == current_dt.date() and len(df) == len(pd.date_range(start_dt, end_dt, freq='B')):
                return df

        # If MongoDB data is missing or outdated, fetch from yfinance
        yf_data = yf.download(ticker, start=start_date, end=end_date)
        
        if yf_data.empty:
            return pd.DataFrame()
            
        # Format yfinance data to match MongoDB structure
        df_yf = pd.DataFrame({
            'date': yf_data.index,
            'Close': yf_data['Close']
        })
        
        # Store new data in MongoDB for future use
        new_records = []
        for _, row in df_yf.iterrows():
            record = {
                'ticker': ticker,
                'date': row['date'].strftime('%Y-%m-%d'),
                'Close': float(row['Close'])
            }
            new_records.append(record)
            
        if new_records:
            try:
                prices_collection.insert_many(new_records, ordered=False)
            except Exception as e:
                # Ignore duplicate key errors
                pass
                
        return df_yf
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    
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

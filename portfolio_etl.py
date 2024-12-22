import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING
from typing import List, Dict
import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

def setup_indexes():
        """Configura índices necessários no MongoDB"""
        prices_collection.create_index([
            ('ticker', ASCENDING),
            ('date', ASCENDING)
        ], unique=True)

setup_indexes()

full = False

end_date = datetime.now()
start_date_full = end_date - timedelta(days=3*365)
start_date_append = end_date - timedelta(days=1)

mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['StockIdea']
prices_collection = db['historical_prices']

transactions = db['transactions'].find()
transactions_list =  list({transaction['Ticker'] for transaction in transactions})
transactions_list.append('^BVSP')

#daily update

for ativo in transactions_list:
      print(f"Baixando dados para {ativo}...")
      dados = yf.download(ativo, start=start_date_append.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
      
      # Verificar se há dados
      if dados.empty:
          print(f"Nenhum dado encontrado para {ativo}.")
          continue
      
      # Converter o índice para coluna (MongoDB não aceita índices como datetime diretamente)
      dados.reset_index(inplace=True)

      dados.columns = dados.columns.get_level_values(0)

      dados['date'] = dados['Date'].dt.strftime("%Y-%m-%d")

      dados = dados.drop('Date', axis=1)
      
      # Adicionar campo de identificação do ativo
      dados["ticker"] = ativo
      
      # Converter para dicionários e salvar no MongoDB
      registros = dados.to_dict("records")
      
      # Use update_one with upsert to handle existing documents
      for registro in registros:
          prices_collection.update_one(
              {"ticker": registro["ticker"], "date": registro["date"]},  # Filter for existing document
              {"$set": registro},  # Update the document if found
              upsert=True  # Insert a new document if not found
          )
      print(f"Dados de {ativo} salvos no MongoDB.")

#historical update of new tickers

historical_prices_db = db['historical_prices'].find()
historical_prices_db_list = list({(prices['ticker'], prices['date']) for prices in historical_prices_db})

# Extract all tickers from historical_prices_db_list
all_tickers = [item[0] for item in historical_prices_db_list]

# Count the occurrences of each ticker
ticker_counts = Counter(all_tickers)

# Create the two lists
old_tickers = [ticker for ticker, count in ticker_counts.items() if count > 2]
new_tickers = [ticker for ticker, count in ticker_counts.items() if count < 2]

for ativo in new_tickers:
      print(f"Baixando dados para {ativo}...")
      dados = yf.download(ativo, start=start_date_full.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
      
      # Verificar se há dados
      if dados.empty:
          print(f"Nenhum dado encontrado para {ativo}.")
          continue
      
      # Converter o índice para coluna (MongoDB não aceita índices como datetime diretamente)
      dados.reset_index(inplace=True)

      dados.columns = dados.columns.get_level_values(0)

      dados['date'] = dados['Date'].dt.strftime("%Y-%m-%d")

      dados = dados.drop('Date', axis=1)
      
      # Adicionar campo de identificação do ativo
      dados["ticker"] = ativo
      
      # Converter para dicionários e salvar no MongoDB
      registros = dados.to_dict("records")
      
      # Use update_one with upsert to handle existing documents
      for registro in registros:
          prices_collection.update_one(
              {"ticker": registro["ticker"], "date": registro["date"]},  # Filter for existing document
              {"$set": registro},  # Update the document if found
              upsert=True  # Insert a new document if not found
          )
      print(f"Dados de {ativo} salvos no MongoDB.")

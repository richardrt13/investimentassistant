import requests
import pandas as pd
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime
import logging
import time
import json
from requests.exceptions import RequestException
import investpy
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

def connect_mongodb(database_name="financial_data", collection_name="stocks"):
    """
    Conecta ao MongoDB e retorna a collection
    """
    mongo_uri = os.getenv('MONGO_URI')
    client = MongoClient(mongo_uri)
    db = client['StockIdea']
    collection = db['stocks']
    return collection

def get_data_from_mongodb_to_df():
    collection = connect_mongodb()
    data = list(collection.find())
    df = pd.DataFrame(data)
    return df

def get_all_bdrs():
    base_url = "https://maisretorno.com/lista-bdr"
    all_stocks = []
    page = 1
    max_retries = 3

    while True:
        url = f"{base_url}/page/{page}"
        tries = 0

        while tries < max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()
                break
            except RequestException as e:
                tries += 1
                if tries == max_retries or '502' in str(e):
                    print(f"Finalizado na página {page-1} devido a: {str(e)}")
                    df = pd.DataFrame(all_stocks)
                    df['type'] = 'BDR'
                    df['country'] = 'brazil'
                    return df
                time.sleep(3)

        soup = BeautifulSoup(response.content, 'html.parser')
        script_tag = soup.find('script', type='application/json')

        if not script_tag:
            break

        json_data = json.loads(script_tag.string)
        stocks_list = json_data.get('props', {}).get('pageProps', {}).get('list', [])

        if not stocks_list:
            break

        for stock in stocks_list:
            main_info = stock.get('mainInfo', {})
            all_stocks.append({
                'symbol': main_info.get('name', ''),
            })

        page += 1
        time.sleep(1)

    df = pd.DataFrame(all_stocks)
    df['type'] = 'BDR'
    df['country'] = 'brazil'
    return df

def get_all_acoes():
    base_url = "https://maisretorno.com/lista-acoes"
    all_stocks = []
    page = 1
    max_retries = 3

    while True:
        url = f"{base_url}/page/{page}"
        tries = 0

        while tries < max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()
                break
            except RequestException as e:
                tries += 1
                if tries == max_retries or '502' in str(e):
                    print(f"Finalizado na página {page-1} devido a: {str(e)}")
                    df = pd.DataFrame(all_stocks)
                    df['type'] = 'acao'
                    df['country'] = 'brazil'
                    return df
                time.sleep(3)

        soup = BeautifulSoup(response.content, 'html.parser')
        script_tag = soup.find('script', type='application/json')

        if not script_tag:
            break

        json_data = json.loads(script_tag.string)
        stocks_list = json_data.get('props', {}).get('pageProps', {}).get('list', [])

        if not stocks_list:
            break

        for stock in stocks_list:
            main_info = stock.get('mainInfo', {})
            all_stocks.append({
                'symbol': main_info.get('name', ''),
            })

        page += 1
        time.sleep(1)

    df = pd.DataFrame(all_stocks)
    df['type'] = 'acao'
    df['country'] = 'brazil'
    return df

def get_stocks_investpy():
    """
    Obtém lista de ativos de diferentes países usando investpy
    """
    usa_stocks = []

    # Lista de países principais
    countries = ['united states']

    for country in countries:
        try:
            # Obtém ações do país
            stocks = investpy.get_stocks(country=country)

            # Adiciona informações básicas
            for _, stock in stocks.iterrows():
                stock_info = {
                    'symbol': stock['symbol'],
                    'country': country
                }
                usa_stocks.append(stock_info)

        except Exception as e:
            print(f"Erro ao obter ações do país {country}: {str(e)}")
            continue
    df = pd.DataFrame(usa_stocks)
    df['type'] = 'stock'

    return df

def enrich_with_yfinance(stocks, collection):
    """
    Enriquece os dados dos ativos com informações do YFinance
    """
    current_date = datetime.now()
    enriched_stocks = []

    for index, stock in stocks.iterrows(): # Iterate through DataFrame rows using iterrows()
        try:
            # Adapta o símbolo para o formato do YFinance se necessário
            if stock['country'] == 'brazil':
                yf_symbol = f"{stock['symbol']}.SA"
            else:
                yf_symbol = stock['symbol']

            # Obtém dados adicionais do YFinance
            yf_stock = yf.Ticker(yf_symbol)
            info = yf_stock.info

            # Enriquece o dicionário com dados do YFinance
            enriched_stock = {
                **stock,  # Mantém os dados do Investpy
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'updated_at': current_date
            }

            # Filtro para identificar o documento
            filter_query = {'symbol': stock['symbol']}

            # Atualiza ou insere o documento
            collection.update_one(
                filter_query,
                {'$set': enriched_stock},
                upsert=True
            )

            time.sleep(0.5)  # Pausa para evitar sobrecarga na API

        except Exception as e:
            print(f"Erro ao processar {stock['symbol']}: {str(e)}")
            continue

def main():
    # Conecta ao MongoDB
    collection = connect_mongodb()

    # Cria índice único
    collection.create_index([("symbol", 1), ("updated_at", 1)], unique=True)

    df_final = df[~df['symbol'].isin(df_stored['symbol'])]


    # Enriquece dados com YFinance
    print(f"Enriquecendo dados de {len(df_final)} ações com YFinance...")
    enrich_with_yfinance(df_final, collection)

    print("Processo finalizado!")

bdrs = get_all_bdrs()
acoes = get_all_acoes()
usa_stocks = get_stocks_investpy()
df = pd.concat([bdrs, acoes, usa_stocks])
df_stored = get_data_from_mongodb_to_df()

if __name__ == "__main__":
    main()

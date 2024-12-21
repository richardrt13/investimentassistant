import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING
from typing import List, Dict
import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor

class PortfolioETL:
    def __init__(self):
        """
        Inicializa o processo de ETL para dados de portfólio.
        """
        # Obtém a URI do MongoDB das variáveis de ambiente
        mongo_uri = os.getenv('MONGO_URI')
        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable is not set")
            
        self.client = MongoClient(mongo_uri)
        self.db = self.client['StockIdea']
        self.prices_collection = self.db['historical_prices']
        self.setup_logging()
        self.setup_indexes()
    
    def setup_logging(self):
        """Configura o sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('portfolio_etl.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('PortfolioETL')

    def setup_indexes(self):
        """Configura índices necessários no MongoDB"""
        self.prices_collection.create_index([
            ('ticker', ASCENDING),
            ('date', ASCENDING)
        ], unique=True)

    def get_active_tickers(self) -> List[str]:
        """
        Obtém a lista de tickers ativos no portfólio.
        
        Returns:
            List[str]: Lista de tickers ativos
        """
        transactions = self.db['transactions'].find()
        tickers = set()
        for transaction in transactions:
            tickers.add(transaction['Ticker'])
        return list(tickers)

    def get_last_update_date(self, ticker: str) -> datetime:
        """
        Obtém a data da última atualização para um ticker específico.
        
        Args:
            ticker (str): Ticker do ativo
            
        Returns:
            datetime: Data da última atualização ou None se não houver dados
        """
        last_record = self.prices_collection.find_one(
            {'ticker': ticker},
            sort=[('date', -1)]
        )
        return last_record['date'] if last_record else None

    def fetch_price_data(self, ticker: str, start_date: datetime) -> Dict:
        """
        Busca dados de preço para um ticker específico.
        
        Args:
            ticker (str): Ticker do ativo
            start_date (datetime): Data inicial para busca
            
        Returns:
            Dict: Dados de preço do ativo
        """
        try:
            # Verifica a última data de atualização
            last_update = self.get_last_update_date(ticker)
            
            if last_update:
                # Se já existem dados, busca apenas os dados novos
                start_date = last_update + timedelta(days=1)
                if start_date >= datetime.now():
                    self.logger.info(f"Dados de {ticker} já estão atualizados")
                    return None
            else:
                # Se não existem dados, busca histórico completo (5 anos)
                start_date = datetime.now() - timedelta(days=1825)
                self.logger.info(f"Buscando histórico completo para {ticker}")

            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date)
            
            if data.empty:
                self.logger.warning(f"Nenhum dado novo encontrado para {ticker}")
                return None
            
            return {
                'ticker': ticker,
                'data': data
            }
        except Exception as e:
            self.logger.error(f"Erro ao buscar dados para {ticker}: {e}")
            return None

    def process_price_data(self, price_data: Dict):
        """
        Processa e salva os dados de preço no MongoDB.
        
        Args:
            price_data (Dict): Dados de preço a serem processados
        """
        if not price_data:
            return
    
        ticker = price_data['ticker']
        data = price_data['data']
    
        records = []
        for date, row in data.iterrows():
            # Convert Timestamp to datetime and remove timezone
            clean_date = date.tz_localize(None)
            
            # Convert numpy float64 to native Python float
            record = {
                'ticker': ticker,
                'date': clean_date,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']),
                'dividends': float(row['Dividends']),
                'stock_splits': float(row['Stock Splits']),
                'adjusted_close': float(row['Close'])
            }
            records.append(record)
    
        if records:
            try:
                operations = [
                    {
                        'updateOne': {
                            'filter': {'ticker': record['ticker'], 'date': record['date']},
                            'update': {'$set': record},
                            'upsert': True
                        }
                    }
                    for record in records
                ]
                
                result = self.prices_collection.bulk_write(operations)
                self.logger.info(f"Dados processados para {ticker}: {len(records)} registros novos")
                
            except Exception as e:
                self.logger.error(f"Erro ao salvar dados para {ticker}: {str(e)}")

    def run_etl(self):
        """
        Executa o processo de ETL completo.
        """
        start_time = time.time()
        self.logger.info("Iniciando processo de ETL")
        
        tickers = self.get_active_tickers()
        self.logger.info(f"Processando {len(tickers)} tickers")
        
        # Usando ThreadPoolExecutor para paralelizar as requisições
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Primeiro fazemos o fetch dos dados
            future_to_ticker = {
                executor.submit(self.fetch_price_data, ticker, None): ticker 
                for ticker in tickers
            }
            
            # Processamos os resultados conforme eles chegam
            for future in future_to_ticker:
                price_data = future.result()
                if price_data:
                    self.process_price_data(price_data)
        
        execution_time = time.time() - start_time
        self.logger.info(f"Processo de ETL concluído em {execution_time:.2f} segundos")

    def get_historical_prices(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Recupera dados históricos do MongoDB.
        
        Args:
            ticker (str): Ticker do ativo
            start_date (datetime): Data inicial
            end_date (datetime): Data final
            
        Returns:
            pd.DataFrame: DataFrame com dados históricos
        """
        query = {
            'ticker': ticker,
            'date': {
                '$gte': start_date,
                '$lte': end_date
            }
        }
        
        cursor = self.prices_collection.find(
            query,
            {'_id': 0, 'date': 1, 'adjusted_close': 1}
        ).sort('date', ASCENDING)
        
        data = list(cursor)
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df

if __name__ == "__main__":
    # Exemplo de uso
    etl = PortfolioETL()
    etl.run_etl()

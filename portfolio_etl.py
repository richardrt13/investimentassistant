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
        return list({transaction['Ticker'] for transaction in transactions})

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

    def fetch_price_data(self, ticker: str) -> Dict:
        """
        Busca dados de preço para um ticker específico.
        
        Args:
            ticker (str): Ticker do ativo
            
        Returns:
            Dict: Dados de preço do ativo ou None se não houver novos dados
        """
        try:
            last_update = self.get_last_update_date(ticker)
            start_date = last_update + timedelta(days=1) if last_update else datetime.now() - timedelta(days=1825)

            if start_date >= datetime.now():
                self.logger.info(f"Dados de {ticker} já estão atualizados")
                return None

            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date)

            if data.empty:
                self.logger.warning(f"Nenhum dado encontrado para {ticker}")
                return None

            return {
                'ticker': ticker,
                'data': data
            }
        except Exception as e:
            self.logger.error(f"Erro ao buscar dados para {ticker}: {e}")

    def process_and_save_data(self, price_data: Dict):
        """
        Processa e salva os dados de preço no MongoDB de forma mais direta e eficiente.
        
        Args:
            price_data (Dict): Dados de preço a serem processados
        """
        if not price_data or 'ticker' not in price_data or 'data' not in price_data:
            return
    
        ticker = price_data['ticker']
        data = price_data['data']
    
        operations = []
    
        for date, row in data.iterrows():
            try:
                record = {
                    'ticker': ticker,
                    'date': date.to_pydatetime().replace(tzinfo=None),
                    'close': float(row['Close']),
                    'adjusted_close': float(row['Close'])
                }
    
                # Directly create InsertOne operation
                operations.append(
                    pymongo.InsertOne({
                        'ticker': ticker,
                        'date': record['date'],
                        'close': record['close'],
                        'adjusted_close': record['adjusted_close']
                    })
                )
    
                # Execute bulk insert in batches of 100
                if len(operations) >= 100:
                    self.prices_collection.bulk_write(operations)
                    operations.clear()
    
            except Exception as e:
                self.logger.error(f"Erro ao processar registro para {ticker}: {e}")

    # Final batch insert if there are remaining operations
    if operations:
        try:
            self.prices_collection.bulk_write(operations)
        except Exception as e:
            self.logger.error(f"Erro ao salvar últimos dados para {ticker}: {e}")

    def run_etl(self):
        """
        Executa o processo de ETL completo.
        """
        start_time = time.time()
        self.logger.info("Iniciando processo de ETL")

        tickers = self.get_active_tickers()
        self.logger.info(f"Processando {len(tickers)} tickers")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.fetch_price_data, ticker) for ticker in tickers]

            for future in futures:
                price_data = future.result()
                if price_data:
                    self.process_and_save_data(price_data)

        self.logger.info(f"Processo de ETL concluído em {time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    etl = PortfolioETL()
    etl.run_etl()

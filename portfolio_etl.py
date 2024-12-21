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

    def validate_record(self, record: Dict) -> bool:
        """
        Validates a single record before saving to MongoDB.
        
        Args:
            record (Dict): The record to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if all required fields are present
            required_fields = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            if not all(field in record for field in required_fields):
                return False
                
            # Verify data types
            if not isinstance(record['ticker'], str):
                return False
            if not isinstance(record['date'], datetime):
                return False
            
            # Verify numeric fields are float and not NaN
            numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits', 'adjusted_close']
            for field in numeric_fields:
                if field in record:
                    if not isinstance(record[field], (int, float)):
                        return False
                    if pd.isna(record[field]):
                        return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating record: {str(e)}")
            return False

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
    
        try:
            operations = []
            invalid_records = 0
            for date, row in data.iterrows():
                # Convert Timestamp to datetime and remove timezone
                clean_date = date.to_pydatetime().replace(tzinfo=None)
                
                # Create the document to be inserted/updated
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
                
                # Validate the record before adding to operations
                if self.validate_record(record):
                    # Create the update operation
                    operation = {
                        'updateOne': {
                            'filter': {
                                'ticker': ticker,
                                'date': clean_date
                            },
                            'update': {
                                '$set': record
                            },
                            'upsert': True
                        }
                    }
                    operations.append(operation)
                else:
                    invalid_records += 1
                    self.logger.warning(f"Invalid record found for {ticker} on {clean_date}")
    
                # Process in smaller batches to avoid any potential size limits
                if len(operations) >= 100:
                    try:
                        self.prices_collection.bulk_write(operations)
                        self.logger.info(f"Batch processed for {ticker}: {len(operations)} records")
                        operations = []  # Clear the processed operations
                    except Exception as batch_error:
                        error_details = str(batch_error)
                        self.logger.error(f"Error in bulk write for {ticker}: {error_details}")
                        # Try to process one by one if batch fails
                        for single_op in operations:
                            try:
                                self.prices_collection.bulk_write([single_op])
                            except Exception as single_error:
                                self.logger.error(f"Error processing single record for {ticker}: {str(single_error)}")
                        operations = []  # Clear the operations regardless of success
    
            # Process any remaining operations
            if operations:
                try:
                    self.prices_collection.bulk_write(operations)
                    self.logger.info(f"Final batch processed for {ticker}: {len(operations)} records")
                except Exception as final_error:
                    error_details = str(final_error)
                    self.logger.error(f"Error in final bulk write for {ticker}: {error_details}")
                    # Try to process remaining operations one by one
                    for single_op in operations:
                        try:
                            self.prices_collection.bulk_write([single_op])
                        except Exception as single_error:
                            self.logger.error(f"Error processing single record for {ticker}: {str(single_error)}")
    
            # Log summary of invalid records if any were found
            if invalid_records > 0:
                self.logger.warning(f"Total invalid records for {ticker}: {invalid_records}")
    
        except Exception as e:
            self.logger.error(f"Error processing data for {ticker}: {str(e)}")
            raise  # Re-raise the exception for the calling code to handle


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
                    self.validate_record()
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

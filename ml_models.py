# ml_models.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta

def prepare_data(ticker, years=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    # Calcular indicadores técnicos
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    
    # Calcular retornos futuros (variável alvo)
    data['Future_Return'] = data['Close'].pct_change(periods=30).shift(-30)
    
    # Remover linhas com valores NaN
    data = data.dropna()
    
    # Selecionar features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
    X = data[features]
    y = data['Future_Return']
    
    return X, y

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def train_model(X, y):
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Seleção de features
    selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold='median')
    selector.fit(X_train_scaled, y_train)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Definir o modelo e os parâmetros para o GridSearch
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Realizar GridSearch com validação cruzada
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_selected, y_train)
    
    # Obter o melhor modelo
    best_model = grid_search.best_estimator_
    
    # Avaliar o modelo
    train_predictions = best_model.predict(X_train_selected)
    test_predictions = best_model.predict(X_test_selected)
    
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
    print(f"Train R2: {train_r2}, Test R2: {test_r2}")
    
    return best_model, selector, scaler

def predict_future_return(model, selector, scaler, X_future):
    X_future_scaled = scaler.transform(X_future)
    X_future_selected = selector.transform(X_future_scaled)
    return model.predict(X_future_selected)

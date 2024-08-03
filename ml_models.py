import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

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
    selector = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42), threshold='median')
    selector.fit(X_train_scaled, y_train)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Definir o modelo e os parâmetros para o GridSearch
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Realizar GridSearch com validação cruzada
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_selected, y_train)
    
    # Obter o melhor modelo
    best_model = grid_search.best_estimator_
    
    train_predictions = best_model.predict(X_train_selected)
    test_predictions = best_model.predict(X_test_selected)
    
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    
    print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
    print(f"Train R2: {train_r2}, Test R2: {test_r2}")
    print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
    
    # Criar gráficos de performance
    performance_plots = create_performance_plots(best_model, X_train_selected, y_train, X_test_selected, y_test)
    
    return best_model, selector, scaler, performance_plots

def create_performance_plots(model, X_train, y_train, X_test, y_test):
    # Predições
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Gráfico de dispersão das predições vs valores reais
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(y_train, train_pred, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    ax1.set_xlabel("Valores Reais")
    ax1.set_ylabel("Predições")
    ax1.set_title("Treino: Predições vs Valores Reais")

    ax2.scatter(y_test, test_pred, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel("Valores Reais")
    ax2.set_ylabel("Predições")
    ax2.set_title("Teste: Predições vs Valores Reais")

    plt.tight_layout()
    scatter_plot = fig

    # Gráfico de resíduos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    residuals_train = y_train - train_pred
    residuals_test = y_test - test_pred

    ax1.scatter(train_pred, residuals_train, alpha=0.5)
    ax1.set_xlabel("Predições")
    ax1.set_ylabel("Resíduos")
    ax1.set_title("Treino: Resíduos vs Predições")
    ax1.axhline(y=0, color='r', linestyle='--')

    ax2.scatter(test_pred, residuals_test, alpha=0.5)
    ax2.set_xlabel("Predições")
    ax2.set_ylabel("Resíduos")
    ax2.set_title("Teste: Resíduos vs Predições")
    ax2.axhline(y=0, color='r', linestyle='--')

    plt.tight_layout()
    residuals_plot = fig

    # Curva de aprendizado
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Treino')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, test_mean, label='Validação Cruzada')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel("Tamanho do Conjunto de Treino")
    plt.ylabel("Score")
    plt.title("Curva de Aprendizado")
    plt.legend()
    learning_curve_plot = plt

    # Importância das features
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title("Importância das Features")
    plt.tight_layout()
    feature_importance_plot = plt

    return {
        'scatter_plot': scatter_plot,
        'residuals_plot': residuals_plot,
        'learning_curve': learning_curve_plot,
        'feature_importance': feature_importance_plot
    }

def save_model(ticker, model, selector, scaler):
    try:
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        joblib.dump(model, f'saved_models/{ticker}_model.joblib')
        joblib.dump(selector, f'saved_models/{ticker}_selector.joblib')
        joblib.dump(scaler, f'saved_models/{ticker}_scaler.joblib')
    except Exception as e:
        print(f"Erro ao salvar o modelo para {ticker}: {str(e)}")

def load_model(ticker):
    try:
        model = joblib.load(f'saved_models/{ticker}_model.joblib')
        selector = joblib.load(f'saved_models/{ticker}_selector.joblib')
        scaler = joblib.load(f'saved_models/{ticker}_scaler.joblib')
        return model, selector, scaler
    except Exception as e:
        print(f"Erro ao carregar o modelo para {ticker}: {str(e)}")
        return None, None, None

def train_or_load_model(ticker, force_train=False):
    model_path = f'saved_models/{ticker}_model.joblib'
    
    if os.path.exists(model_path) and not force_train:
        print(f"Tentando carregar modelo existente para {ticker}")
        model, selector, scaler = load_model(ticker)
        if model is None:
            print(f"Falha ao carregar modelo para {ticker}. Treinando novo modelo.")
            X, y = prepare_data(ticker)
            model, selector, scaler, performance_plots = train_model(X, y)
            save_model(ticker, model, selector, scaler)
        else:
            performance_plots = None
    else:
        print(f"Treinando novo modelo para {ticker}")
        X, y = prepare_data(ticker)
        model, selector, scaler, performance_plots = train_model(X, y)
        save_model(ticker, model, selector, scaler)
    
    return model, selector, scaler, performance_plots

def predict_future_return(model, selector, scaler, X_future):
    X_future_scaled = scaler.transform(X_future)
    X_future_selected = selector.transform(X_future_scaled)
    return model.predict(X_future_selected)

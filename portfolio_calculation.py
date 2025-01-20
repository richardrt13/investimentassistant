import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

@st.cache_data(ttl=3600)
def get_cumulative_return(ticker):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    hist = stock.history(start=start_date, end=end_date)
    if len(hist) > 0:
        cumulative_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
    else:
        cumulative_return = None
    return cumulative_return

def calculate_returns(prices):
    if prices.empty:
        return pd.DataFrame()
    returns = prices.pct_change().dropna()
    # Remove infinitos e NaNs
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns

# Função para calcular o desempenho do portfólio
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility


# Função para calcular o índice de Sharpe negativo (para otimização)
def negative_sharpe_ratio(weights, returns, risk_free_rate):
    p_return, p_volatility = portfolio_performance(weights, returns)
    return -(p_return - risk_free_rate) / p_volatility

# Função para otimizar o portfólio
def optimize_portfolio(returns, risk_free_rate):
    num_assets = returns.shape[1]
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets*[1./num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Função para gerar portfólios aleatórios
def generate_random_portfolios(returns, num_portfolios=5000):
    results = []
    n_assets = returns.shape[1]
    for _ in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        p_return, p_volatility = portfolio_performance(weights, returns)
        results.append({
            'Return': p_return,
            'Volatility': p_volatility,
            'Sharpe': (p_return - risk_free_rate) / p_volatility,
            'Weights': weights
        })
    return pd.DataFrame(results)

def detect_price_anomalies(prices, window=20, threshold=2):
    returns = prices.pct_change().dropna()
    model = ARIMA(returns, order=(1,1,1))
    results = model.fit()
    residuals = results.resid
    mean = residuals.rolling(window=window).mean()
    std = residuals.rolling(window=window).std()
    z_scores = (residuals - mean) / std
    return abs(z_scores) > threshold

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def optimize_weights(ativos_df):
    """
    Otimiza os pesos para maximizar a correlação entre os scores e a rentabilidade
    """
    # Primeiro, calcular todos os scores e armazenar em um array
    scores = calculate_scores(ativos_df)
    
    def objective(weights):
        # Calcular o score ponderado usando os pesos
        weighted_scores = np.sum(scores * weights.reshape(-1, 1), axis=0)
        
        # Calcular a correlação entre os scores ponderados e a rentabilidade
        correlation = np.corrcoef(weighted_scores, ativos_df['Rentabilidade Acumulada (5 anos)'])[0, 1]
        
        # Retornar o negativo da correlação (queremos maximizar)
        return -correlation

    # Restrições: soma dos pesos = 1 e todos os pesos >= 0
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, 1) for _ in range(7)]  # 7 pesos a serem otimizados

    # Pesos iniciais
    initial_weights = np.array([1/7] * 7)

    # Otimização
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

def calculate_scores(ativos_df):
    """
    Calcula os scores individuais para cada métrica
    """
    scores = np.zeros((7, len(ativos_df)))
    
    # ROE/P/L score
    scores[0] = ativos_df['ROE'] / ativos_df['P/L']
    
    # P/VP score (inverso)
    scores[1] = 1 / ativos_df['P/VP']
    
    # Volume score (log)
    scores[2] = np.log(ativos_df['Volume'])
    
    # Revenue growth score
    scores[3] = ativos_df['revenue_growth']
    
    # Income growth score
    scores[4] = ativos_df['income_growth']
    
    # Debt stability score
    scores[5] = ativos_df['debt_stability']
    
    # Dividend Yield score
    scores[6] = ativos_df['Dividend Yield']
    
    # Normalizar os scores
    for i in range(scores.shape[0]):
        scores[i] = (scores[i] - np.min(scores[i])) / (np.max(scores[i]) - np.min(scores[i]))
    
    return scores

def calculate_score(ativos_df, weights):
    """
    Calcula o score final usando os pesos otimizados
    """
    scores = calculate_scores(ativos_df)
    return np.sum(scores * weights.reshape(-1, 1), axis=0)

def calculate_adjusted_score(row, optimized_weights):
    base_score = (
        optimized_weights[0] * row['ROE'] / row['P/L'] +
        optimized_weights[1] / row['P/VP'] +
        optimized_weights[2] * np.log(row['Volume']) +
        optimized_weights[3] * row['revenue_growth'] +
        optimized_weights[4] * row['income_growth'] +
        optimized_weights[5] * row['debt_stability'] +
        optimized_weights[6] * row['Dividend Yield']
    )

    # Fator de qualidade
    quality_factor = (row['ROE'] + row['ROIC']) / 2

    # Aplicação do fator de qualidade
    adjusted_base_score = base_score * (1 + quality_factor * 0.1)

    # Cálculo da penalidade por anomalias
    anomaly_penalty = sum([row[col] for col in ['price_anomaly', 'rsi_anomaly']])

    # Aplicação da penalidade por anomalias
    final_score = adjusted_base_score * (1 - 0.05 * anomaly_penalty)

    return final_score


def adjust_weights_for_anomalies(weights, anomaly_scores):
    adjusted_weights = weights * (1 - anomaly_scores)
    return adjusted_weights / adjusted_weights.sum()

def calculate_anomaly_scores(returns):
    anomaly_scores = returns.apply(lambda x: detect_price_anomalies(x).mean())
    return anomaly_scores

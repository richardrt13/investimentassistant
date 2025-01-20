import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

class FinancialAnalysis:
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate

    @st.cache_data(ttl=3600)
    def get_cumulative_return(self, ticker):
        """Get cumulative return for a given ticker over the past 5 years."""
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        hist = stock.history(start=start_date, end=end_date)
        if len(hist) > 0:
            return (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
        return None

    def calculate_returns(self, prices):
        """Calculate returns from price data."""
        if prices.empty:
            return pd.DataFrame()
        returns = prices.pct_change().dropna()
        return returns.replace([np.inf, -np.inf], np.nan).dropna()

    def portfolio_performance(self, weights, returns):
        """Calculate portfolio return and volatility."""
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return portfolio_return, portfolio_volatility

    def negative_sharpe_ratio(self, weights, returns):
        """Calculate negative Sharpe ratio for optimization."""
        p_return, p_volatility = self.portfolio_performance(weights, returns)
        return -(p_return - self.risk_free_rate) / p_volatility

    def optimize_portfolio(self, returns):
        """Optimize portfolio weights using Sharpe ratio."""
        num_assets = returns.shape[1]
        args = (returns,)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0.0, 1.0)
        bounds = tuple(bound for _ in range(num_assets))
        
        result = minimize(
            self.negative_sharpe_ratio,
            num_assets*[1./num_assets],
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x

    def generate_random_portfolios(self, returns, num_portfolios=5000):
        """Generate random portfolio allocations."""
        results = []
        n_assets = returns.shape[1]
        
        for _ in range(num_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            p_return, p_volatility = self.portfolio_performance(weights, returns)
            results.append({
                'Return': p_return,
                'Volatility': p_volatility,
                'Sharpe': (p_return - self.risk_free_rate) / p_volatility,
                'Weights': weights
            })
        return pd.DataFrame(results)

    def detect_price_anomalies(self, prices, window=20, threshold=2):
        """Detect price anomalies using ARIMA model."""
        returns = prices.pct_change().dropna()
        model = ARIMA(returns, order=(1,1,1))
        results = model.fit()
        residuals = results.resid
        mean = residuals.rolling(window=window).mean()
        std = residuals.rolling(window=window).std()
        z_scores = (residuals - mean) / std
        return abs(z_scores) > threshold

    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_scores(self, ativos_df):
        """Calculate individual scores for each metric."""
        scores = np.zeros((7, len(ativos_df)))
        
        # Calculate individual scores
        scores[0] = ativos_df['ROE'] / ativos_df['P/L']
        scores[1] = 1 / ativos_df['P/VP']
        scores[2] = np.log(ativos_df['Volume'])
        scores[3] = ativos_df['revenue_growth']
        scores[4] = ativos_df['income_growth']
        scores[5] = ativos_df['debt_stability']
        scores[6] = ativos_df['Dividend Yield']
        
        # Normalize scores
        for i in range(scores.shape[0]):
            scores[i] = (scores[i] - np.min(scores[i])) / (np.max(scores[i]) - np.min(scores[i]))
        
        return scores

    def optimize_weights(self, ativos_df):
        """Optimize weights to maximize correlation between scores and returns."""
        scores = self.calculate_scores(ativos_df)
        
        def objective(weights):
            weighted_scores = np.sum(scores * weights.reshape(-1, 1), axis=0)
            correlation = np.corrcoef(weighted_scores, ativos_df['Rentabilidade Acumulada (5 anos)'])[0, 1]
            return -correlation

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0, 1) for _ in range(7)]
        initial_weights = np.array([1/7] * 7)
        
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def calculate_adjusted_score(self, row, optimized_weights):
        """Calculate final adjusted score with quality factors and anomaly penalties."""
        base_score = (
            optimized_weights[0] * row['ROE'] / row['P/L'] +
            optimized_weights[1] / row['P/VP'] +
            optimized_weights[2] * np.log(row['Volume']) +
            optimized_weights[3] * row['revenue_growth'] +
            optimized_weights[4] * row['income_growth'] +
            optimized_weights[5] * row['debt_stability'] +
            optimized_weights[6] * row['Dividend Yield']
        )

        quality_factor = (row['ROE'] + row['ROIC']) / 2
        adjusted_base_score = base_score * (1 + quality_factor * 0.1)
        
        anomaly_penalty = sum([row[col] for col in ['price_anomaly', 'rsi_anomaly']])
        final_score = adjusted_base_score * (1 - 0.05 * anomaly_penalty)
        
        return final_score

    def adjust_weights_for_anomalies(self, weights, anomaly_scores):
        """Adjust portfolio weights based on anomaly scores."""
        adjusted_weights = weights * (1 - anomaly_scores)
        return adjusted_weights / adjusted_weights.sum()

    def calculate_anomaly_scores(self, returns):
        """Calculate anomaly scores for returns."""
        return returns.apply(lambda x: self.detect_price_anomalies(x).mean())

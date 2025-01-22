import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache
import streamlit as st

class PortfolioCache:
    def __init__(self, cache_duration=timedelta(minutes=15)):
        self.cache_duration = cache_duration
        self.cache = {}
        
    @st.cache_data(ttl=900)  # 15 minutes in seconds
    def get_cached_performance_data(self, user_id, last_transaction_time):
        """
        Get cached portfolio performance data, using last transaction time as part of the cache key
        """
        return self._calculate_portfolio_performance(user_id)
    
    def _calculate_portfolio_performance(self, user_id):
        """
        Calculate portfolio performance with all required metrics
        """
        portfolio_data, invested_value = get_portfolio_performance(user_id)
        
        if portfolio_data.empty:
            return None, None
            
        # Calculate daily and cumulative returns
        daily_portfolio_value = portfolio_data.sum(axis=1)
        daily_returns = daily_portfolio_value.pct_change()
        portfolio_cumulative_returns = (1 + daily_returns).cumprod() - 1
        
        # Calculate per-asset metrics
        asset_metrics = {}
        for ticker in portfolio_data.columns:
            initial_value = invested_value[ticker]
            current_value = portfolio_data[ticker].iloc[-1]
            if initial_value > 0:
                asset_return = ((current_value - initial_value) / initial_value) * 100
                asset_metrics[ticker] = {
                    'return': asset_return,
                    'current_value': current_value,
                    'initial_value': initial_value
                }
        
        return {
            'portfolio_data': portfolio_data,
            'invested_value': invested_value,
            'daily_value': daily_portfolio_value,
            'cumulative_returns': portfolio_cumulative_returns,
            'asset_metrics': asset_metrics,
            'last_updated': datetime.now()
        }
    
    def invalidate_cache(self, user_id):
        """
        Invalidate cache for a specific user
        """
        if user_id in self.cache:
            del self.cache[user_id]
            st.cache_data.clear()

def get_last_transaction_time(user_id, collection):
    """
    Get the timestamp of the most recent transaction for a user
    """
    last_transaction = collection.find_one(
        {'user_id': user_id},
        sort=[('Date', -1)]
    )
    return last_transaction['Date'] if last_transaction else None

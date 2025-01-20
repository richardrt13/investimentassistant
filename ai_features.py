import pandas as pd
import streamlit as st
import google.generativeai as genai
from typing import Optional, Dict

class PortfolioAnalyzer:
    def __init__(self):
        try:
            genai.configure(api_key=st.secrets["api_key"])
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            st.warning(f"Não foi possível configurar o modelo Gemini: {e}")
            self.model = None

    def analyze_portfolio(self, portfolio_data: pd.DataFrame, invested_value: pd.Series) -> Optional[str]:
        """
        Analisa a carteira usando Gemini para gerar insights e recomendações
        """
        if self.model is None:
            return None

        # Preparar os dados da carteira para análise
        portfolio_summary = self._prepare_portfolio_summary(portfolio_data, invested_value)
        
        # Criar o prompt para o modelo
        prompt = self._create_analysis_prompt(portfolio_summary)
        
        try:
            # Gerar análise
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Erro ao gerar análise: {e}")
            return None

    def get_optimization_suggestions(self, portfolio_data: pd.DataFrame, 
                                  market_data: Dict) -> Optional[str]:
        """
        Gera sugestões de otimização baseadas nos dados da carteira e do mercado
        """
        if self.model is None:
            return None

        prompt = self._create_optimization_prompt(portfolio_data, market_data)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Erro ao gerar sugestões de otimização: {e}")
            return None

    def _prepare_portfolio_summary(self, portfolio_data: pd.DataFrame, 
                                 invested_value: pd.Series) -> Dict:
        """
        Prepara um resumo estruturado da carteira para análise
        """
        summary = {
            "total_invested": invested_value.sum(),
            "current_value": portfolio_data.iloc[-1].sum(),
            "assets": []
        }

        for ticker in portfolio_data.columns:
            asset_data = {
                "ticker": ticker,
                "invested": invested_value[ticker],
                "current_value": portfolio_data[ticker].iloc[-1],
                "return": ((portfolio_data[ticker].iloc[-1] / invested_value[ticker]) - 1) * 100,
                "weight": portfolio_data[ticker].iloc[-1] / portfolio_data.iloc[-1].sum() * 100
            }
            summary["assets"].append(asset_data)

        return summary

    def _create_analysis_prompt(self, portfolio_summary: Dict) -> str:
        """
        Cria um prompt estruturado para análise da carteira
        """
        prompt = f"""Analise a seguinte carteira de investimentos e forneça insights detalhados:

        Valor Total Investido: R$ {portfolio_summary['total_invested']:.2f}
        Valor Atual: R$ {portfolio_summary['current_value']:.2f}
        
        Composição da Carteira:
        """
        
        for asset in portfolio_summary["assets"]:
            prompt += f"""
            - {asset['ticker']}:
              * Valor Investido: R$ {asset['invested']:.2f}
              * Valor Atual: R$ {asset['current_value']:.2f}
              * Retorno: {asset['return']:.2f}%
              * Peso na Carteira: {asset['weight']:.2f}%
            """

        prompt += """
        Por favor, forneça de forma resumida:
        1. Uma análise geral da carteira considerando diversificação e performance
        2. Pontos fortes e fracos identificados
        3. Riscos potenciais e oportunidades
        4. Sugestões específicas para otimização e rebalanceamento
        5. Análise da distribuição setorial e concentração de riscos
        """

        return prompt

    def _create_optimization_prompt(self, portfolio_data: pd.DataFrame, 
                                  market_data: Dict) -> str:
        """
        Cria um prompt para sugestões de otimização
        """
        prompt = f"""Com base nos dados da carteira e condições atuais de mercado, sugira otimizações:

        Dados de Mercado:
        - Ibovespa: {market_data.get('ibovespa_return', 'N/A')}% YTD
        - Taxa Selic: {market_data.get('selic', 'N/A')}%
        
        Carteira Atual:
        """
        
        for ticker in portfolio_data.columns:
            current_value = portfolio_data[ticker].iloc[-1]
            weight = current_value / portfolio_data.iloc[-1].sum() * 100
            prompt += f"- {ticker}: {weight:.2f}% da carteira\n"

        prompt += """
        Por favor, forneça de forma resumida:
        1. Sugestões específicas de rebalanceamento
        2. Ativos que poderiam ser incluídos ou removidos
        3. Estratégias para otimizar a relação risco-retorno
        4. Considerações sobre o momento atual do mercado
        5. Plano de ação prático para implementar as sugestões
        """

        return prompt

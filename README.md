# Documentação do Projeto de Gestão de Carteira de Investimentos

## Índice
1. Descrição do Projeto

2. Tecnologias Utilizadas

3. Estrutura do Projeto

4. Funcionalidades Principais

5. Fluxo da Aplicação

6. Funções e Explicações

7. Interface com o Usuário

8. Conclusão

## 1. Descrição do Projeto

Este projeto é uma aplicação de gestão de carteira de investimentos desenvolvida em Python utilizando o Streamlit. A aplicação permite ao usuário registrar transações de compra e venda de ativos, acompanhar o desempenho da carteira, realizar otimização de portfólios com base no Índice de Sharpe, e simular alocações de aportes futuros. A solução busca maximizar o retorno ajustado ao risco, utilizando dados financeiros e históricos dos ativos.

## 2. Tecnologias Utilizadas
Python - Linguagem principal de programação.

Streamlit - Framework para construção de interfaces web interativas.

Pandas - Manipulação e análise de dados.

NumPy - Suporte a cálculos numéricos.

yFinance - API para obtenção de dados financeiros.

SciPy - Utilizada para otimização numérica.

Plotly - Criação de gráficos interativos.

ARIMA - Utilizada para modelagem de séries temporais e detecção de anomalias.

## 3. Estrutura do Projeto

portfolio_performance.py: Arquivo principal contendo toda a lógica de backend e frontend da aplicação.

ativos.csv: Arquivo com os dados dos ativos financeiros.

MongoDB: Utilizado para armazenar as transações da carteira.

## 4. Funcionalidades Principais

### Acompanhamento da Carteira:

Registro de transações (compra e venda de ativos).

Exibição do desempenho atual da carteira (valor total investido, retorno, valor atual).

Comparação de retorno da carteira com o índice Ibovespa.

### Recomendação de Ativos:

Sugestão de alocação de capital baseado em otimização do Índice de Sharpe.

Análise fundamentalista dos ativos (P/L, ROE, Dividend Yield, etc.).

Detecção de anomalias nos preços dos ativos.

### Otimização de Portfólio:

Cálculo da alocação ótima de ativos utilizando otimização com base no Índice de Sharpe.

Ajuste da alocação com base em fatores de crescimento e anomalias detectadas.

### Gráficos e Visualizações:

Gráficos interativos para retorno acumulado por ativo.

Fronteira eficiente do portfólio otimizado.

Comparação de retorno entre a carteira e o Ibovespa.

## 5. Fluxo da Aplicação

### Acompanhamento da Carteira
O usuário registra transações de compra e venda de ativos.

A aplicação coleta dados históricos dos ativos.

A aplicação exibe o desempenho atual da carteira e realiza comparações com benchmarks como o Ibovespa.

Gráficos são gerados para visualização do retorno acumulado por ativo e do valor da carteira ao longo do tempo.

### Recomendação de Ativos
O usuário insere o valor que deseja investir.

A aplicação obtém dados fundamentalistas dos ativos e filtra aqueles com melhores métricas.

É feita uma análise de anomalias e uma otimização da alocação com base no Índice de Sharpe.

A alocação recomendada é exibida junto com explicações detalhadas para cada ativo.


## 6. Funções e Explicações
### Aqui estão as principais funções do código e suas responsabilidades:

load_assets(): Carrega os dados dos ativos a partir de um arquivo CSV.

get_fundamental_data(ticker): Obtém dados fundamentais de um ativo (P/L, ROE, Dividend Yield, etc.) usando a API do yFinance.

get_stock_data(tickers, years): Coleta dados históricos de preços ajustados para um ou mais ativos.

portfolio_performance(weights, returns): Calcula o retorno e a volatilidade esperados de um portfólio dado seus pesos e retornos históricos.

optimize_portfolio(returns, risk_free_rate): Realiza a otimização do portfólio com base no Índice de Sharpe.

generate_random_portfolios(returns, num_portfolios): Gera múltiplos portfólios aleatórios para simulação e comparação de desempenho.

plot_efficient_frontier(returns, optimal_portfolio): Gera um gráfico interativo da fronteira eficiente.

detect_price_anomalies(prices): Detecta anomalias nos preços dos ativos usando ARIMA.

calculate_rsi(prices): Calcula o Índice de Força Relativa (RSI) para análise técnica.

get_ibovespa_data(start_date, end_date): Obtém dados históricos do índice Ibovespa para comparação de retorno.


## 7. Interface com o Usuário
A aplicação utiliza o Streamlit para fornecer uma interface amigável e interativa para o usuário. Algumas das principais seções da interface incluem:

### Página de Acompanhamento:

Input de transações (compra e venda).

Visualização do desempenho da carteira com gráficos interativos.

Aporte inteligente com sugestão de alocação otimizada.

### Página de Recomendação de Ativos:

Seletor de setores e tipos de ativos.

Geração de recomendações otimizadas.

Análises detalhadas e explicações para cada ativo recomendado.

## 8. Conclusão

Este projeto é uma solução completa para gestão e otimização de carteiras de investimentos, permitindo ao usuário acompanhar suas transações, otimizar sua alocação de capital, e entender os fatores por trás das recomendações feitas. A aplicação pode ser expandida para incluir mais funcionalidades como novos tipos de ativos, integração com mais APIs de mercado, e melhoria nos algoritmos de otimização.

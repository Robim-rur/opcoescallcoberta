import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="Scanner Estatístico Buy Side", layout="wide")

st.title("📊 Scanner de Opções: Maior Probabilidade de Lucro")
st.subheader("Filtro: Preço < R$ 10.00 | Alvo: +5% de Valorização")

# Lista de ativos com liquidez em opções (preços baixos)
TICKERS_SA = [
    "COGN3.SA", "MGLU3.SA", "HAPV3.SA", "BHIA3.SA", "CASH3.SA", 
    "AZUL4.SA", "RAIZ4.SA", "MRVE3.SA", "CVCB3.SA", "PETZ3.SA", 
    "LWSA3.SA", "JHSF3.SA", "POMO4.SA", "POSI3.SA", "USIM5.SA", "BEEF3.SA"
]

def calcular_probabilidade(S, K, vol, t=30/365):
    """
    Calcula a probabilidade do preço terminar abaixo do strike (não ser exercido).
    S: Preço Atual, K: Strike, vol: Volatilidade, t: Tempo (em anos)
    """
    if vol == 0 or S == 0: return 0
    d2 = (np.log(S / K) + (- 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    prob_nao_exercido = norm.cdf(d2)
    return prob_nao_exercido

def scanner_estatistico(tickers):
    resultados = []
    progresso = st.progress(0)
    
    for idx, ticker in enumerate(tickers):
        try:
            asset = yf.Ticker(ticker)
            # Histórico para calcular volatilidade histórica (proxy para a implícita)
            hist = asset.history(period="3mo")
            if len(hist) < 20: continue
            
            preco_atual = hist['Close'].iloc[-1]
            
            if preco_atual < 10.0:
                # Cálculo de Volatilidade (Anualizada)
                returns = np.log(hist['Close'] / hist['Close'].shift(1))
                volatilidade = returns.std() * np.sqrt(252)
                
                strike_alvo = preco_atual * 1.05
                # Probabilidade de sucesso: Preço terminar abaixo do strike 
                # (você ganha o prêmio e mantém o ativo que valorizou até quase 5%)
                prob = calcular_probabilidade(preco_atual, strike_alvo, volatilidade)
                
                resultados.append({
                    "Ticker": ticker.replace(".SA", ""),
                    "Preço Atual": round(preco_atual, 2),
                    "Strike (+5%)": round(strike_alvo, 2),
                    "Volatilidade (Anual)": f"{volatilidade*100:.1f}%",
                    "Prob. de Sucesso*": prob
                })
        except Exception as e:
            pass
        progresso.progress((idx + 1) / len(tickers))
    
    progresso.empty()
    return pd.DataFrame(resultados)

# Execução do Scanner
if st.button('🔍 Escanear Mercado e Calcular Probabilidades'):
    with st.spinner('Analisando volatilidade e preços...'):
        df = scanner_estatistico(TICKERS_SA)
        
        if not df.empty:
            # Ordenação por maior probabilidade de lucro (Prob. de Sucesso)
            df = df.sort_values(by="Prob. de Sucesso*", ascending=False)
            
            # Formatação para exibição
            df_display = df.copy()
            df_display["Prob. de Sucesso*"] = df_display["Prob. de Sucesso*"].map("{:.2%}".format)
            
            st.write("### Resultado da Análise Estatística")
            st.markdown("*Ordenado da maior para a menor probabilidade de a ação NÃO ultrapassar os 5% (ideal para manter o ativo e o prêmio).*")
            
            # Tabela Estilizada
            st.table(df_display)
            
            # Gráfico Comparativo
            st.write("---")
            st.write("### Comparativo Visual: Probabilidade vs Ativo")
            fig = go.Figure(go.Bar(
                x=df["Ticker"], 
                y=df["Prob. de Sucesso*"] * 100,
                marker_color='royalblue'
            ))
            fig.update_layout(title="Probabilidade de Lucro por Ativo (%)", template="plotly_dark", yaxis_title="% Probabilidade")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Nenhum ativo abaixo de R$ 10 encontrado nos critérios.")

st.sidebar.info("""
**Explicação Estatística:**
A probabilidade é calculada usando a volatilidade histórica dos últimos 90 dias. 
Uma probabilidade de **80%** significa que, estatisticamente, há 80% de chance de o ativo terminar o mês abaixo do strike (mantendo a ação em carteira e o prêmio no bolso).
""")

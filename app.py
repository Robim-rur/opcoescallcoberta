import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

# Configuração
st.set_page_config(page_title="Scanner de Rentabilidade 7%", layout="wide")

st.title("🎯 Estratégia Buy Side: Alvo 7%")
st.subheader("Foco: Rentabilidade Total (Valorização + Prêmio)")

# Lista de ativos abaixo de R$ 10 com liquidez
TICKERS_SA = ["COGN3.SA", "MGLU3.SA", "HAPV3.SA", "BHIA3.SA", "RAIZ4.SA", "MRVE3.SA", "PETZ3.SA", "JHSF3.SA", "AZUL4.SA"]

def calcular_dados(ticker, alvo_perc):
    try:
        asset = yf.Ticker(ticker)
        hist = asset.history(period="60d")
        if hist.empty: return None
        
        preco_atual = hist['Close'].iloc[-1]
        if preco_atual > 10.0: return None
        
        # Estatísticas
        returns = np.log(hist['Close'] / hist['Close'].shift(1))
        vol_diaria = returns.std()
        
        strike_alvo = preco_atual * (1 + alvo_perc)
        
        # Probabilidade de NÃO ser exercido (Manter a ação)
        # d2 do modelo Black-Scholes para probabilidade ITM/OTM
        d2 = (np.log(strike_alvo / preco_atual) / (vol_diaria * np.sqrt(21)))
        prob_manter = norm.cdf(d2)
        
        return {
            "Ticker": ticker.replace(".SA", ""),
            "Preço Atual": round(preco_atual, 2),
            "Strike (7%)": round(strike_alvo, 2),
            "Prob. Manter Ação": round(prob_manter * 100, 1),
            "Lucro Se Exercido": f"{alvo_perc*100:.0f}% + Prêmio"
        }
    except:
        return None

# Interface
st.sidebar.header("Configurações")
alvo = st.sidebar.slider("Alvo de Venda (%)", 5.0, 10.0, 7.0) / 100

if st.button('🚀 Escanear Melhores Taxas'):
    resultados = []
    with st.spinner('Calculando probabilidades para o próximo pregão...'):
        for t in TICKERS_SA:
            res = calcular_dados(t, alvo)
            if res: resultados.append(res)
            
    if resultados:
        df = pd.DataFrame(resultados).sort_values(by="Prob. Manter Ação", ascending=False)
        
        # Exibição
        st.write(f"### Ranking de Ativos para Alvo de {alvo*100:.0f}%")
        
        # Estilização
        def color_prob(val):
            color = 'green' if val > 75 else 'orange' if val > 65 else 'red'
            return f'color: {color}'

        st.table(df.style.applymap(color_prob, subset=['Prob. Manter Ação']))
        
        # Plano de Trade
        st.write("---")
        selecionado = st.selectbox("Simular Ordem para:", df["Ticker"])
        detalhe = df[df["Ticker"] == selecionado].iloc[0]
        
        st.info(f"""
        **Plano de sábado para {selecionado}:**
        - Preço de referência: **R$ {detalhe['Preço Atual']}**
        - Strike para buscar: **R$ {detalhe['Strike (7%)']}**
        - Se a ação subir e você for exercido: Ganha **{alvo*100:.0f}%** de lucro bruto.
        - Se a ação não chegar lá: Você fica com o prêmio da opção e **mantém os papéis**.
        """)
    else:
        st.warning("Nenhum ativo disponível nos critérios.")

st.markdown("""
> **Dica do Especialista:** Ao aceitar o exercício em 7%, você pode focar em vender opções **ATM (no dinheiro)** ou ligeiramente **OTM (fora do dinheiro)**. Isso maximiza o valor do prêmio que você recebe na largada.
""")

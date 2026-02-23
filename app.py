import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

# Configuração
st.set_page_config(page_title="Planejador Noturno Buy Side", layout="wide")

st.title("🌙 Planejador de Venda Coberta (Operação Pós-Mercado)")
st.info("Dados baseados no fechamento do último pregão. Ideal para planejar ordens de abertura.")

TICKERS_SA = ["COGN3.SA", "MGLU3.SA", "HAPV3.SA", "BHIA3.SA", "RAIZ4.SA", "MRVE3.SA", "PETZ3.SA", "JHSF3.SA", "USIM5.SA"]

def analisar_ativo_noturno(ticker):
    asset = yf.Ticker(ticker)
    # Pegamos 60 dias para uma volatilidade robusta
    hist = asset.history(period="60d")
    if hist.empty: return None
    
    ultimo_fechamento = hist['Close'].iloc[-1]
    data_fechamento = hist.index[-1].strftime('%d/%m/%Y')
    
    if ultimo_fechamento > 10.0: return None
    
    # Cálculo de Volatilidade Histórica Anualizada
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
    vol_diaria = log_returns.std()
    vol_anual = vol_diaria * np.sqrt(252)
    
    # Alvo de 5%
    strike_alvo = ultimo_fechamento * 1.05
    
    # Estatística: Probabilidade de ficar ABAIXO do strike em 21 dias úteis (1 mês)
    # Usamos o modelo simplificado de probabilidade log-normal
    d2 = (np.log(strike_alvo / ultimo_fechamento) / (vol_diaria * np.sqrt(21)))
    prob_sucesso = norm.cdf(d2)
    
    return {
        "Ticker": ticker.replace(".SA", ""),
        "Fechamento": round(ultimo_fechamento, 2),
        "Data": data_fechamento,
        "Strike Alvo": round(strike_alvo, 2),
        "Prob. de Manter Ação (%)": round(prob_sucesso * 100, 2),
        "Volatilidade": round(vol_anual * 100, 2)
    }

# Interface
if st.button('🔍 Gerar Relatório de Oportunidades para o Próximo Pregão'):
    resultados = []
    for t in TICKERS_SA:
        res = analisar_ativo_noturno(t)
        if res: resultados.append(res)
    
    df = pd.DataFrame(resultados)
    # Ordenar por maior probabilidade de sucesso (estatística a seu favor)
    df = df.sort_values(by="Prob. de Manter Ação (%)", ascending=False)
    
    st.write("### 📈 Ranking Estatístico (Ações < R$ 10)")
    st.table(df)
    
    # Seção de Plano de Execução
    st.write("---")
    st.write("### 📝 Plano de Execução (Para seu Home Broker)")
    
    escolha = st.selectbox("Selecione o ativo que deseja operar:", df["Ticker"])
    row = df[df["Ticker"] == escolha].iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Estratégia para {escolha}:**")
        st.write(f"1. Comprar {escolha} na abertura até **R$ {row['Fechamento'] * 1.01:.2f}**")
        st.write(f"2. Vender CALL com Strike próximo a **R$ {row['Strike Alvo']:.2f}**")
    
    with col2:
        st.metric("Confiança Estatística", f"{row['Prob. de Manter Ação (%)']}%")
        st.caption("Esta probabilidade indica a chance de você ganhar o prêmio da opção e NÃO ser obrigado a vender a ação, permanecendo com ela em carteira.")

st.sidebar.warning("""
**Nota de Risco:**
Operar à noite significa que você está ignorando possíveis notícias que saiam após o fechamento. 
Sempre confira se não há divulgação de balanços antes da abertura do mercado.
""")

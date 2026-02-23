import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta  # Biblioteca para análise técnica
import plotly.graph_objects as go

st.set_page_config(page_title="Sinalizador de Entrada Buy Side", layout="wide")

st.title("🎯 Sinalizador de Entradas Estatísticas")
st.subheader("Foco: Ações < R$ 10 com Alta Probabilidade de Repique")

# Lista de ativos com boa liquidez no Buy Side
TICKERS = ["MGLU3.SA", "COGN3.SA", "HAPV3.SA", "RAIZ4.SA", "MRVE3.SA", "PETZ3.SA", "AZUL4.SA", "CSNA3.SA", "USIM5.SA", "CIEL3.SA"]

def check_signal(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: return None
        
        # Cálculos Técnicos
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA200'] = ta.sma(df['Close'], length=200)
        
        last_price = df['Close'].iloc[-1]
        last_rsi = df['RSI'].iloc[-1]
        sma200 = df['SMA200'].iloc[-1]
        
        if last_price > 10.0: return None

        # Lógica do Sinal
        # Se RSI < 35, a probabilidade de subir em breve é estatisticamente alta
        status = "AGUARDAR"
        cor = "white"
        
        if last_rsi < 35:
            status = "🔥 OPORTUNIDADE (Sobrecompra)"
            cor = "green"
        elif last_rsi > 70:
            status = "⚠️ EVITAR (Esticada demais)"
            cor = "red"

        return {
            "Ticker": ticker.replace(".SA", ""),
            "Preço": round(last_price, 2),
            "RSI (Força)": round(last_rsi, 2),
            "Acima da Média 200?": "Sim" if last_price > sma200 else "Não",
            "Sinal": status
        }
    except:
        return None

# Interface do Usuário
if st.button('🔍 Escanear Oportunidades para Segunda-Feira'):
    resultados = []
    with st.spinner('Analisando gráficos históricos...'):
        for t in TICKERS:
            res = check_signal(t)
            if res: resultados.append(res)
    
    if resultados:
        df_res = pd.DataFrame(resultados)
        
        # Exibição da Tabela
        st.write("### Relatório de Entradas")
        st.dataframe(df_res.style.apply(lambda x: ['background-color: #004d00' if v == '🔥 OPORTUNIDADE (Sobrecompra)' else '' for v in x], axis=1), use_container_width=True)

        # Explicação do Alvo
        st.info("""
        **Como usar este App no sábado:**
        1. Olhe os ativos marcados como **OPORTUNIDADE**.
        2. Eles estão com o RSI baixo, o que significa que "cansaram de cair".
        3. Sua meta é entrar na segunda-feira e buscar os **7% de lucro**.
        """)
        
        # Gráfico de Apoio
        escolha = st.selectbox("Veja o gráfico de:", df_res["Ticker"])
        df_plot = yf.download(f"{escolha}.SA", period="6mo")
        fig = go.Figure(data=[go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'])])
        fig.update_layout(title=f"Gráfico de {escolha}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Nenhum ativo nos critérios encontrados hoje.")

st.sidebar.markdown("""
### Por que isso funciona?
O **RSI (IFR)** mede a velocidade e a mudança dos movimentos de preço. 
Estatisticamente, quando ele cai abaixo de 30-35, o preço está em uma região onde os compradores costumam aparecer. 
Para você que opera no **Buy Side**, é o melhor momento para entrar.
""")

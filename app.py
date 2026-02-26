import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Scanner – D+ > D- + EMA169 + Flip do SAR (ranking por probabilidade de gain)")

# =========================================================
# CONFIGURAÇÃO DA ESTATÍSTICA
# =========================================================

GAIN_PCT = 0.02     # 2% de alvo
HORIZON = 10        # até 10 pregões

# =========================================================
# LISTA DOS ATIVOS  (sua lista completa)
# =========================================================

ativos_scan = sorted(set([
"RRRP3.SA","ALOS3.SA","ALPA4.SA","ABEV3.SA","ARZZ3.SA","ASAI3.SA","AZUL4.SA","B3SA3.SA","BBAS3.SA","BBDC3.SA",
"BBDC4.SA","BBSE3.SA","BEEF3.SA","BPAC11.SA","BRAP4.SA","BRFS3.SA","BRKM5.SA","CCRO3.SA","CMIG4.SA","CMIN3.SA",
"COGN3.SA","CPFE3.SA","CPLE6.SA","CRFB3.SA","CSAN3.SA","CSNA3.SA","CYRE3.SA","DXCO3.SA","EGIE3.SA","ELET3.SA",
"ELET6.SA","EMBR3.SA","ENEV3.SA","ENGI11.SA","EQTL3.SA","EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GOLL4.SA",
"HAPV3.SA","HYPE3.SA","ITSA4.SA","ITUB4.SA","JBSS3.SA","KLBN11.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","MRFG3.SA",
"MRVE3.SA","MULT3.SA","NTCO3.SA","PETR3.SA","PETR4.SA","PRIO3.SA","RADL3.SA","RAIL3.SA","RAIZ4.SA","RENT3.SA",
"RECV3.SA","SANB11.SA","SBSP3.SA","SLCE3.SA","SMTO3.SA","SUZB3.SA","TAEE11.SA","TIMS3.SA","TOTS3.SA","TRPL4.SA",
"UGPA3.SA","USIM5.SA","VALE3.SA","VIVT3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA",
"AAPL34.SA","AMZO34.SA","GOGL34.SA","MSFT34.SA","TSLA34.SA","META34.SA","NFLX34.SA",
"NVDC34.SA","MELI34.SA","BABA34.SA","DISB34.SA","VISA34.SA","WMTB34.SA",
"BOVA11.SA","IVVB11.SA","SMAL11.SA","GOLD11.SA","DIVO11.SA","NDIV11.SA"
]))

# =========================================================
# INDICADORES
# =========================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def calcular_dmi(df, n=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(n).mean()

    plus_di = 100 * plus_dm.rolling(n).mean() / atr
    minus_di = 100 * minus_dm.rolling(n).mean() / atr

    return plus_di, minus_di

def parabolic_sar(df, step=0.02, max_step=0.2):

    high = df["High"].values
    low = df["Low"].values

    sar = np.zeros(len(df))
    trend = 1
    af = step
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, len(df)):

        sar[i] = sar[i-1] + af * (ep - sar[i-1])

        if trend == 1:
            sar[i] = min(sar[i], low[i-1], low[i])

            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)

            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = step
        else:
            sar[i] = max(sar[i], high[i-1], high[i])

            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)

            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = step

    return pd.Series(sar, index=df.index)

# =========================================================
# PROCESSAMENTO
# =========================================================

@st.cache_data(show_spinner=False)
def processar():

    resultados = []

    for ticker in ativos_scan:

        try:
            df = yf.download(ticker, period="12y", interval="1d", progress=False)
            if df is None or len(df) < 300:
                continue

            df = df.dropna()

            df["EMA169"] = ema(df["Close"], 169)
            df["PDI"], df["MDI"] = calcular_dmi(df)
            df["SAR"] = parabolic_sar(df)

            df["flip_sar"] = (df["Close"] > df["SAR"]) & (df["Close"].shift(1) <= df["SAR"].shift(1))

            df["sinal"] = (
                (df["Close"] > df["EMA169"]) &
                (df["PDI"] > df["MDI"]) &
                (df["flip_sar"])
            )

            ganhos = 0
            total = 0

            idx = df.index

            for i in range(len(df) - HORIZON):

                if df["sinal"].iloc[i]:

                    total += 1
                    preco = df["Close"].iloc[i]
                    alvo = preco * (1 + GAIN_PCT)

                    max_fut = df["High"].iloc[i+1:i+HORIZON+1].max()

                    if max_fut >= alvo:
                        ganhos += 1

            prob = ganhos / total * 100 if total > 0 else 0

            hoje = False
            if df["sinal"].iloc[-1]:
                hoje = True

            resultados.append({
                "Ativo": ticker,
                "Sinal hoje": "SIM" if hoje else "",
                "Ocorrências históricas": total,
                "Probabilidade de atingir gain (%)": round(prob,2)
            })

        except:
            pass

    return pd.DataFrame(resultados)

with st.spinner("Processando..."):
    tabela = processar()

tabela = tabela.sort_values("Probabilidade de atingir gain (%)", ascending=False)

st.subheader("Ranking – maior probabilidade histórica de atingir o gain")

st.dataframe(tabela, use_container_width=True)

st.info(
f"""
Sinal diário quando:

• Close acima da EMA 169
• DI+ maior que DI−
• Flip do SAR para compra

Probabilidade calculada para atingir {GAIN_PCT*100:.1f}% de gain
em até {HORIZON} pregões após o sinal.
"""
)

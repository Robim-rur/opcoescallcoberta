import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Scanner – Votação de Indicadores + Probabilidade de Alvo")

# ============================================================
# LISTA DOS ATIVOS
# ============================================================

ativos_scan = sorted(set([
    "RRRP3.SA","ALOS3.SA","ALPA4.SA","ABEV3.SA","ARZZ3.SA","ASAI3.SA","AZUL4.SA","B3SA3.SA","BBAS3.SA","BBDC3.SA",
    "BBDC4.SA","BBSE3.SA","BEEF3.SA","BPAC11.SA","BRAP4.SA","BRFS3.SA","BRKM5.SA","CCRO3.SA","CMIG4.SA","CMIN3.SA",
    "COGN3.SA","CPFE3.SA","CPLE6.SA","CRFB3.SA","CSAN3.SA","CSNA3.SA","CYRE3.SA","DXCO3.SA","EGIE3.SA","ELET3.SA",
    "ELET6.SA","EMBR3.SA","ENEV3.SA","ENGI11.SA","EQTL3.SA","EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GOLL4.SA",
    "HAPV3.SA","HYPE3.SA","ITSA4.SA","ITUB4.SA","JBSS3.SA","KLBN11.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","MRFG3.SA",
    "MRVE3.SA","MULT3.SA","NTCO3.SA","PETR3.SA","PETR4.SA","PRIO3.SA","RADL3.SA","RAIL3.SA","RAIZ4.SA","RENT3.SA",
    "RECV3.SA","SANB11.SA","SBSP3.SA","SLCE3.SA","SMTO3.SA","SUZB3.SA","TAEE11.SA","TIMS3.SA","TOTS3.SA","TRPL4.SA",
    "UGPA3.SA","USIM5.SA","VALE3.SA","VIVT3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA",

    "AAPL34.SA","AMZO34.SA","GOGL34.SA","MSFT34.SA","TSLA34.SA","META34.SA","NFLX34.SA","NVDC34.SA","MELI34.SA",

    "HGLG11.SA","XPLG11.SA","VISC11.SA","XPML11.SA","KNRI11.SA","MXRF11.SA","HGRE11.SA","IRDM11.SA","CPTS11.SA"
]))

# ============================================================

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd_hist(close):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    return macd - signal

def stochastic(df, k=14):
    low_min = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    return 100 * (df["Close"] - low_min) / (high_max - low_min)

def adx_calc(df, n=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where(plus_dm > 0, 0.0)
    minus_dm = minus_dm.where(minus_dm > 0, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(n).mean()

    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_val = dx.rolling(n).mean()

    # >>> GARANTIA DE SERIES (corrige o erro do seu ambiente)
    adx_val = pd.Series(adx_val.values, index=df.index)
    plus_di = pd.Series(plus_di.values, index=df.index)
    minus_di = pd.Series(minus_di.values, index=df.index)

    return adx_val, plus_di, minus_di

def obv(df):
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()

# ============================================================

def tipo_ativo(ticker):

    if ticker.endswith("34.SA"):
        return "BDR", 0.08

    if ticker.endswith("11.SA"):
        return "FII", 0.06

    return "Ação", 0.08

# ============================================================

def calcula_probabilidade(df, sinais, alvo):

    idxs = sinais[sinais].index

    if len(idxs) < 10:
        return None

    acertos = 0
    total = 0

    for idx in idxs[:-20]:

        entrada = df.loc[idx, "Close"]
        futuro = df.loc[idx:].iloc[1:21]

        if futuro["High"].max() >= entrada * (1 + alvo):
            acertos += 1

        total += 1

    if total == 0:
        return None

    return round(100 * acertos / total, 1)

# ============================================================

def analisar(ticker):

    df = yf.download(ticker, period="3y", interval="1d", progress=False)

    if df is None or len(df) < 250:
        return None

    df = df.copy()

    df["EMA21"] = ema(df["Close"], 21)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI"] = rsi(df["Close"])
    df["MACD_H"] = macd_hist(df["Close"])
    df["STO"] = stochastic(df)

    adx_val, plus_di, minus_di = adx_calc(df)

    df["ADX"] = adx_val
    df["+DI"] = plus_di
    df["-DI"] = minus_di

    df["MM20"] = df["Close"].rolling(20).mean()
    df["BBmid"] = df["MM20"]

    df["OBV"] = obv(df)
    df["OBV_slope"] = df["OBV"].diff(5)

    df = df.dropna()

    if len(df) < 200:
        return None

    sinais = pd.DataFrame(index=df.index)

    sinais["i1"] = df["Close"] > df["EMA21"]
    sinais["i2"] = df["Close"] > df["EMA50"]
    sinais["i3"] = df["RSI"] > 50
    sinais["i4"] = df["MACD_H"] > 0
    sinais["i5"] = df["STO"] > 50
    sinais["i6"] = df["ADX"] > 20
    sinais["i7"] = df["+DI"] > df["-DI"]
    sinais["i8"] = df["Close"] > df["MM20"]
    sinais["i9"] = df["Close"] > df["BBmid"]
    sinais["i10"] = df["OBV_slope"] > 0

    votos = sinais.sum(axis=1)
    maioria = votos >= 6

    tipo, alvo = tipo_ativo(ticker)

    prob = calcula_probabilidade(df, maioria, alvo)

    last_votos = int(votos.iloc[-1])
    last = df.iloc[-1]

    if last_votos < 6:
        return None

    return {
        "Ativo": ticker.replace(".SA",""),
        "Tipo": tipo,
        "Preço": round(float(last["Close"]), 2),
        "Indicadores OK": last_votos,
        "Alvo": f"{int(alvo*100)}%",
        "Probabilidade (%)": prob
    }

# ============================================================

if st.button("Escanear ativos"):

    resultados = []

    with st.spinner("Processando..."):
        for t in ativos_scan:
            r = analisar(t)
            if r:
                resultados.append(r)

    if len(resultados) == 0:
        st.warning("Nenhum ativo passou no critério mínimo de maioria.")
    else:
        df = pd.DataFrame(resultados)
        df = df.sort_values("Probabilidade (%)", ascending=False, na_position="last")
        st.dataframe(df, use_container_width=True)
        st.write(f"Total de ativos em maioria de sinais: {len(df)}")

st.sidebar.markdown("""
Scanner por votação de indicadores.

Entrada quando:
pelo menos 6 de 10 indicadores estão positivos.

A probabilidade é medida pelo próprio histórico do ativo,
verificando se o preço atingiu o alvo em até 20 pregões.
""")

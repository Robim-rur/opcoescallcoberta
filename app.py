import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Scanner – D+ Diário + D+ Semanal + Setup 1-2-3 + EMA 169")

# ==========================================================
# LISTA DE ATIVOS
# ==========================================================

ativos_scan = sorted(set([
"RRRP3.SA","ALOS3.SA","ALPA4.SA","ABEV3.SA","ARZZ3.SA","ASAI3.SA","AZUL4.SA","B3SA3.SA","BBAS3.SA","BBDC3.SA",
"BBDC4.SA","BBSE3.SA","BEEF3.SA","BPAC11.SA","BRAP4.SA","BRFS3.SA","BRKM5.SA","CCRO3.SA","CMIG4.SA","CMIN3.SA",
"COGN3.SA","CPFE3.SA","CPLE6.SA","CRFB3.SA","CSAN3.SA","CSNA3.SA","CYRE3.SA","DXCO3.SA","EGIE3.SA","ELET3.SA",
"ELET6.SA","EMBR3.SA","ENEV3.SA","ENGI11.SA","EQTL3.SA","EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GOLL4.SA",
"HAPV3.SA","HYPE3.SA","ITSA4.SA","ITUB4.SA","JBSS3.SA","KLBN11.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","MRFG3.SA",
"MRVE3.SA","MULT3.SA","NTCO3.SA","PETR3.SA","PETR4.SA","PRIO3.SA","RADL3.SA","RAIL3.SA","RAIZ4.SA","RENT3.SA",
"RECV3.SA","SANB11.SA","SBSP3.SA","SLCE3.SA","SMTO3.SA","SUZB3.SA","TAEE11.SA","TIMS3.SA","TOTS3.SA","TRPL4.SA",
"UGPA3.SA","USIM5.SA","VALE3.SA","VIVT3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA","AURE3.SA","BHIA3.SA","CASH3.SA",
"CVCB3.SA","DIRR3.SA","ENAT3.SA","GMAT3.SA","IFCM3.SA","INTB3.SA","JHSF3.SA","KEPL3.SA","MOVI3.SA","ORVR3.SA",
"PETZ3.SA","PLAS3.SA","POMO4.SA","POSI3.SA","RANI3.SA","RAPT4.SA","STBP3.SA","TEND3.SA","TUPY3.SA",
"BRSR6.SA","CXSE3.SA","AAPL34.SA","AMZO34.SA","GOGL34.SA","MSFT34.SA","TSLA34.SA","META34.SA","NFLX34.SA",
"NVDC34.SA","MELI34.SA","BABA34.SA","DISB34.SA","PYPL34.SA","JNJB34.SA","PGCO34.SA","KOCH34.SA","VISA34.SA",
"WMTB34.SA","NIKE34.SA","ADBE34.SA","AVGO34.SA","CSCO34.SA","COST34.SA","CVSH34.SA","GECO34.SA","GSGI34.SA",
"HDCO34.SA","INTC34.SA","JPMC34.SA","MAEL34.SA","MCDP34.SA","MDLZ34.SA","MRCK34.SA","ORCL34.SA","PEP334.SA",
"PFIZ34.SA","PMIC34.SA","QCOM34.SA","SBUX34.SA","TGTB34.SA","TMOS34.SA","TXN34.SA","UNHH34.SA","UPSB34.SA",
"VZUA34.SA","ABTT34.SA","AMGN34.SA","AXPB34.SA","BAOO34.SA","CATP34.SA","HONB34.SA","BOVA11.SA","IVVB11.SA",
"SMAL11.SA","HASH11.SA","GOLD11.SA","GARE11.SA","HGLG11.SA","XPLG11.SA","VILG11.SA","BRCO11.SA","BTLG11.SA",
"XPML11.SA","VISC11.SA","HSML11.SA","MALL11.SA","KNRI11.SA","JSRE11.SA","PVBI11.SA","HGRE11.SA","MXRF11.SA",
"KNCR11.SA","KNIP11.SA","CPTS11.SA","IRDM11.SA","DIVO11.SA","NDIV11.SA","SPUB11.SA"
]))

# ==========================================================
# INDICADORES
# ==========================================================

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def calcular_dmi(df, periodo=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
        low = low.iloc[:, 0]
        close = close.iloc[:, 0]

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(periodo).mean()

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    plus_di = 100 * (plus_dm.rolling(periodo).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(periodo).mean() / atr)

    return plus_di, minus_di


# ==========================================================
# SETUP 1-2-3 DE COMPRA (DIÁRIO)
# padrão simples e objetivo
# ==========================================================

def setup_123_compra(df):

    cond = (
        (df["Low"].shift(2) > df["Low"].shift(1)) &
        (df["Low"] > df["Low"].shift(1)) &
        (df["Close"] > df["High"].shift(1))
    )

    return cond


# ==========================================================
# SCANNER
# ==========================================================

@st.cache_data(show_spinner=False)
def scan():

    sinais = []

    for ticker in ativos_scan:

        try:

            df = yf.download(ticker, period="4y", interval="1d", progress=False)

            if df is None or len(df) < 250:
                continue

            df = df.dropna()

            if isinstance(df["Close"], pd.DataFrame):
                df["Close"] = df["Close"].iloc[:, 0]

            # ----------------------
            # DIÁRIO
            # ----------------------

            df["EMA169"] = ema(df["Close"], 169)

            pdi_d, mdi_d = calcular_dmi(df, 14)

            df["PDI"] = pdi_d
            df["MDI"] = mdi_d

            df["SETUP123"] = setup_123_compra(df)

            cond_diario = (
                (df["Close"] > df["EMA169"]) &
                (df["PDI"] > df["MDI"]) &
                (df["SETUP123"])
            )

            # ----------------------
            # SEMANAL
            # ----------------------

            weekly = df.resample("W-FRI").agg({
                "High": "max",
                "Low": "min",
                "Close": "last"
            }).dropna()

            pdi_w, mdi_w = calcular_dmi(weekly, 14)

            weekly["PDI_W"] = pdi_w
            weekly["MDI_W"] = mdi_w

            df = df.merge(
                weekly[["PDI_W", "MDI_W"]],
                left_index=True,
                right_index=True,
                how="left"
            )

            df[["PDI_W", "MDI_W"]] = df[["PDI_W", "MDI_W"]].ffill()

            cond_semanal = df["PDI_W"] > df["MDI_W"]

            # ----------------------
            # SINAL FINAL
            # ----------------------

            sinal = cond_diario & cond_semanal

            if sinal.iloc[-1]:

                sinais.append({
                    "Ativo": ticker,
                    "Fechamento": float(df["Close"].iloc[-1]),
                    "PDI diário": float(df["PDI"].iloc[-1]),
                    "MDI diário": float(df["MDI"].iloc[-1]),
                    "PDI semanal": float(df["PDI_W"].iloc[-1]),
                    "MDI semanal": float(df["MDI_W"].iloc[-1])
                })

        except:
            continue

    return pd.DataFrame(sinais)


with st.spinner("Buscando sinais..."):
    resultado = scan()

st.subheader("Sinais de compra de hoje")

if resultado.empty:
    st.warning("Nenhum ativo atende simultaneamente todas as condições hoje.")
else:
    st.dataframe(resultado, use_container_width=True)

st.info(
"""
Regras do scanner:

DIÁRIO
• Preço acima da EMA 169
• DI+ maior que DI-
• Setup 1-2-3 de compra

SEMANAL
• DI+ maior que DI-

Scanner apenas para localizar ativos em condição técnica.
"""
)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Scanner – Probabilidade de Gain 6% em até 20 pregões (EMA169 + DMI + SAR)")

# ============================================================
# LISTA FIXA DE ATIVOS (178)
# ============================================================

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

# ============================================================
# PARÂMETROS
# ============================================================

GAIN = 0.06
LOSS = 0.04
JANELA = 20
PERIODO_DMI = 14
EMA_PERIODO = 169

# ============================================================
# INDICADORES
# ============================================================

def calcular_ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def calcular_dmi(df, n=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(n).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / atr

    return plus_di, minus_di

def calcular_sar(df, af=0.02, max_af=0.2):

    high = df["High"].values
    low = df["Low"].values

    sar = np.zeros(len(df))
    trend = 1
    ep = high[0]
    af_atual = af
    sar[0] = low[0]

    for i in range(1, len(df)):

        sar[i] = sar[i-1] + af_atual * (ep - sar[i-1])

        if trend == 1:
            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af_atual = af
            else:
                if high[i] > ep:
                    ep = high[i]
                    af_atual = min(af_atual + af, max_af)
        else:
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af_atual = af
            else:
                if low[i] < ep:
                    ep = low[i]
                    af_atual = min(af_atual + af, max_af)

    return pd.Series(sar, index=df.index)

# ============================================================
# PROBABILIDADE (CORRIGIDA)
# ============================================================

def probabilidade_gain(df, sinais_idx):

    sucessos = 0
    total = 0

    for i in sinais_idx:

        if i + JANELA >= len(df):
            continue

        entrada = df["Close"].iloc[i]
        alvo = entrada * (1 + GAIN)
        stop = entrada * (1 - LOSS)

        bateu_gain = False

        for j in range(i + 1, i + JANELA + 1):

            if df["Low"].iloc[j] <= stop:
                bateu_gain = False
                break

            if df["High"].iloc[j] >= alvo:
                bateu_gain = True
                break

        total += 1

        if bateu_gain:
            sucessos += 1

    if total == 0:
        return 0.0, 0

    return sucessos / total * 100, total

# ============================================================
# ANÁLISE DO ATIVO
# ============================================================

def analisar_ativo(ticker):

    df = yf.download(ticker, period="6y", interval="1d", progress=False)

    if df is None or len(df) < 300:
        return None

    df = df.dropna()

    df["EMA"] = calcular_ema(df["Close"], EMA_PERIODO)

    plus_di, minus_di = calcular_dmi(df, PERIODO_DMI)

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["sar"] = calcular_sar(df)

    df["sar_compra"] = (
        (df["Close"] > df["sar"]) &
        (df["Close"].shift(1) <= df["sar"].shift(1))
    )

    df["condicao"] = (
        (df["Close"] > df["EMA"]) &
        (df["plus_di"] > df["minus_di"]) &
        (df["sar_compra"])
    )

    sinais = df.index[df["condicao"]]

    if len(sinais) == 0:
        return None

    idx_sinais = [df.index.get_loc(i) for i in sinais]

    prob, total = probabilidade_gain(df, idx_sinais)

    hoje = df.iloc[-1]

    passou_hoje = (
        (hoje["Close"] > hoje["EMA"]) and
        (hoje["plus_di"] > hoje["minus_di"]) and
        (hoje["sar_compra"])
    )

    if not passou_hoje:
        return None

    return {
        "Ativo": ticker.replace(".SA", ""),
        "Probabilidade de atingir 6% em até 20 pregões sem stop (%)": round(prob, 2),
        "Quantidade histórica de sinais": int(total)
    }

# ============================================================
# EXECUÇÃO
# ============================================================

if st.button("Rodar scanner"):

    resultados = []

    barra = st.progress(0.0)
    total_ativos = len(ativos_scan)

    for i, ticker in enumerate(ativos_scan):

        try:
            r = analisar_ativo(ticker)
            if r is not None:
                resultados.append(r)
        except:
            pass

        barra.progress((i + 1) / total_ativos)

    if len(resultados) == 0:
        st.warning("Nenhum ativo passou no filtro técnico hoje.")
    else:

        tabela = pd.DataFrame(resultados)

        coluna = "Probabilidade de atingir 6% em até 20 pregões sem stop (%)"

        tabela = tabela.sort_values(coluna, ascending=False)

        st.subheader("Ranking – maior probabilidade estatística de atingir +6% antes de −4%")

        st.dataframe(tabela, use_container_width=True)

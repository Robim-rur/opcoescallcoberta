import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Scanner Ichimoku – Roberson", layout="wide")

st.title("📈 Scanner Ichimoku – Continuação de Tendência")
st.subheader("Pullback + retomada | diário com confirmação semanal")

# =====================================================
# LISTA INTEGRAL DE 178 ATIVOS
# =====================================================

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


def ichimoku(df):

    high = df["High"]
    low = df["Low"]

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2

    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2

    cloud_a = senkou_a.shift(26)
    cloud_b = senkou_b.shift(26)

    return tenkan, kijun, cloud_a, cloud_b


def check_ativo(ticker):

    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)

        if df is None or df.empty:
            return None

        df["Tenkan"], df["Kijun"], df["CloudA"], df["CloudB"] = ichimoku(df)

        df = df.dropna()

        if len(df) < 60:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        cloud_top = max(last["CloudA"], last["CloudB"])

        # -------------------------
        # DIÁRIO
        # -------------------------

        preco_acima_nuvem = last["Close"] > cloud_top
        tenkan_acima_kijun = last["Tenkan"] > last["Kijun"]
        retomada = last["Close"] > last["Tenkan"]

        tocou_tenkan = prev["Low"] <= prev["Tenkan"] <= prev["High"]
        tocou_kijun = prev["Low"] <= prev["Kijun"] <= prev["High"]

        cloud_top_prev = max(prev["CloudA"], prev["CloudB"])
        cloud_bot_prev = min(prev["CloudA"], prev["CloudB"])

        tocou_nuvem = (prev["Low"] <= cloud_top_prev) and (prev["High"] >= cloud_bot_prev)

        pullback_valido = tocou_tenkan or tocou_kijun or tocou_nuvem

        # -------------------------
        # SEMANAL (mais realista)
        # -------------------------

        dfw = df.resample("W-FRI").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        })

        dfw["Tenkan"], dfw["Kijun"], dfw["CloudA"], dfw["CloudB"] = ichimoku(dfw)
        dfw = dfw.dropna()

        if dfw.empty:
            return None

        lastw = dfw.iloc[-1]

        # confirmação de tendência no semanal pelo Kijun
        semanal_ok = lastw["Close"] > lastw["Kijun"]

        sinal = (
            preco_acima_nuvem and
            tenkan_acima_kijun and
            retomada and
            pullback_valido and
            semanal_ok
        )

        return {
            "Ticker": ticker.replace(".SA", ""),
            "Preço": round(last["Close"], 2),
            "Diário > nuvem": "Sim" if preco_acima_nuvem else "Não",
            "Tenkan > Kijun": "Sim" if tenkan_acima_kijun else "Não",
            "Pullback": "Sim" if pullback_valido else "Não",
            "Semanal > Kijun": "Sim" if semanal_ok else "Não",
            "Sinal": "✅ CONTEXTO DE ENTRADA" if sinal else "—"
        }

    except Exception:
        return None


if st.button("🔎 Escanear 178 ativos"):

    resultados = []

    with st.spinner("Analisando mercado..."):
        for t in ativos_scan:
            r = check_ativo(t)
            if r is not None:
                resultados.append(r)

    if len(resultados) == 0:
        st.warning("Nenhum ativo retornou dados.")
    else:

        df = pd.DataFrame(resultados)

        st.subheader("Resultado geral")
        st.dataframe(df, use_container_width=True)

        df_ok = df[df["Sinal"] == "✅ CONTEXTO DE ENTRADA"]

        st.subheader("Ativos em contexto de entrada – Ichimoku")

        st.dataframe(df_ok, use_container_width=True)

        st.write(f"Total em contexto: {len(df_ok)}")


st.sidebar.markdown("""
### Regras (versão ajustada)

DIÁRIO
- Preço acima da nuvem
- Tenkan acima do Kijun
- Candle anterior tocou Tenkan, Kijun ou nuvem
- Candle atual fechou acima do Tenkan

SEMANAL
- Fechamento acima do Kijun

Scanner de continuação de tendência.
Somente compra.
Filtro pensado para não matar sinal.
""")

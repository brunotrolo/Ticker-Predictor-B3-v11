
import streamlit as st
from sklearn.calibration import calibration_curve
import numpy as np
import pandas as pd, numpy as np
from datetime import date, timedelta
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from b3_utils import load_b3_tickers, ensure_sa_suffix, is_known_b3_ticker, search_b3
import plotly.graph_objects as go

# v11: dependência opcional (graceful fallback)
try:
    from neuralprophet import NeuralProphet
    _NP_AVAILABLE = True
except Exception:
    _NP_AVAILABLE = False


from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title="B3 + ML — v10.4 (UX)", page_icon="✨", layout="wide")

# Small CSS for chips
st.markdown("""
<style>
.chip { font-size: 0.85rem; display: inline-block; }
.kpi-caption { opacity: 0.75; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ================== Helpers ==================

# ================== v10.5 helpers ==================
def _compute_reliability(y_true, y_prob, n_bins=10):
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        return prob_true, prob_pred
    except Exception as e:
        return None, None

def _plot_reliability(prob_true, prob_pred):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot([0,1],[0,1], linestyle="--")
    ax.plot(prob_pred, prob_true, marker="o")
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequência observada")
    ax.set_title("Curva de Confiabilidade (Calibration)")
    fig.tight_layout()
    return fig

def _build_trades_table(prices: pd.Series, signals: pd.Series, horizon:int=1, cost_bps:int=0, slip_bps:int=0):
    """Cria uma tabela simples de trades a partir de sinais binários (1=long, 0=flat).
    Usa buy&hold de 'horizon' dias após sinal 1, com custo/slippage simétricos.
    """
    try:
        df = pd.DataFrame({"price": prices, "signal": signals}).dropna()
        entries = df.index[df["signal"].diff().fillna(0) > 0]  # momentos em que sinal vai 0->1
        rows = []
        for t in entries:
            exit_idx = df.index.get_loc(t) + horizon
            if exit_idx >= len(df):
                continue
            t_exit = df.index[exit_idx]
            p0 = df.loc[t, "price"]
            p1 = df.iloc[exit_idx]["price"]
            gross = (p1 / p0 - 1.0)
            # custos (por lado)
            cost = (cost_bps/1e4) + (slip_bps/1e4)
            net = gross - 2*cost
            rows.append({"Entrada": t, "Saída": t_exit, "Retorno %": net*100.0, "Retorno bruto %": gross*100.0})
        trades = pd.DataFrame(rows)
        if not trades.empty:
            trades["Dias"] = (trades["Saída"] - trades["Entrada"]).dt.days
            trades = trades.sort_values("Retorno %", ascending=False)
        return trades
    except Exception:
        return pd.DataFrame()
# ===================================================

def set_plotly_template(theme_choice: str):
    import plotly.io as pio
    if theme_choice == "Claro":
        pio.templates.default = "plotly"
        st.markdown("<style>body, .stApp {background-color: #ffffff; color: #111111;}</style>", unsafe_allow_html=True)
    else:
        pio.templates.default = "plotly_dark"
        st.markdown("<style>body, .stApp {background-color: #0e1117; color: #e5e5e5;}</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    df = yf.download(ensure_sa_suffix(ticker), start=start, end=end, auto_adjust=True, progress=False)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index()
    return df

def sma(s, w): return s.rolling(window=w, min_periods=w).mean()
def rsi(s, w=14):
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(w).mean()
    ma_down = down.rolling(w).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_indicators(df, want_sma50=False, want_sma200=False):
    if df.empty: return df
    df = df.copy()
    df["SMA20"]=sma(df["Close"],20)
    if want_sma50: df["SMA50"]=sma(df["Close"],50)
    if want_sma200: df["SMA200"]=sma(df["Close"],200)
    df["RSI14"]=rsi(df["Close"])
    return df

def annotate_events(df):
    ev = pd.DataFrame(index=df.index)
    ev["far_below"] = (df["Close"]/df["SMA20"] - 1) <= -0.07   # 7% abaixo da SMA20
    ev["rsi_os"]    = df["RSI14"] <= 30
    return ev

def build_features(df, horizon=1):
    d = df.copy()
    d["ret_1"] = d["Close"].pct_change(1)
    d["ret_3"] = d["Close"].pct_change(3)
    d["ret_5"] = d["Close"].pct_change(5)
    d["ret_10"] = d["Close"].pct_change(10)
    if "SMA50" not in d.columns: d["SMA50"] = sma(d["Close"],50)
    if "SMA200" not in d.columns: d["SMA200"] = sma(d["Close"],200)
    d["dist_sma20"] = d["Close"]/d["SMA20"] - 1
    d["dist_sma50"] = d["Close"]/d["SMA50"] - 1
    d["dist_sma200"] = d["Close"]/d["SMA200"] - 1
    d["vol_5"] = d["Close"].pct_change().rolling(5).std()
    d["vol_10"] = d["Close"].pct_change().rolling(10).std()
    d["rsi"] = d["RSI14"]
    d["future_ret"] = d["Close"].shift(-horizon)/d["Close"] - 1.0
    d["target_up"] = (d["future_ret"] > 0).astype(int)
    feat_cols = ["ret_1","ret_3","ret_5","ret_10","dist_sma20","dist_sma50","dist_sma200","vol_5","vol_10","rsi"]
    d = d.dropna(subset=feat_cols + ["target_up","future_ret"]).reset_index(drop=True)
    X = d[feat_cols].values; y = d["target_up"].values; future_ret = d["future_ret"].values
    return d, X, y, future_ret, feat_cols

def fit_calibrated(model, X_train, y_train, frac_calib=0.2, method="sigmoid"):
    n = len(X_train)
    if n < 40:
        m = clone(model); m.fit(X_train, y_train); return m
    n_cal = max(int(n * frac_calib), 50) if n >= 100 else max(int(n * 0.1), 20)
    n_cal = min(n_cal, n-20) if n > 40 else max(5, n-5)
    m = clone(model); m.fit(X_train[:-n_cal], y_train[:-n_cal])
    cal = CalibratedClassifierCV(m, method=method, cv="prefit")
    cal.fit(X_train[-n_cal:], y_train[-n_cal:])
    return cal

def safe_tscv_params(n_samples, n_splits, test_size_min):
    max_splits = max(1, n_samples // max(1, test_size_min) - 1)
    adj_splits = min(n_splits, max_splits)
    adj_test = test_size_min
    while adj_splits < 2 and adj_test > 20:
        adj_test = max(20, adj_test // 2)
        max_splits = max(1, n_samples // max(1, adj_test) - 1)
        adj_splits = min(n_splits, max_splits)
    return adj_splits, adj_test

def best_threshold_by_return(proba, rets):
    if len(proba) != len(rets) or len(proba) == 0:
        return 0.5
    grid = np.linspace(0.4, 0.7, 61)
    best_thr, best_ret = 0.5, -1e9
    for thr in grid:
        sig = (proba >= thr).astype(int)
        cum = (1 + pd.Series(rets * sig)).prod() - 1
        if cum > best_ret:
            best_ret = float(cum); best_thr = float(thr)
    return best_thr

def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    equity = (1 + pd.Series(returns)).cumprod()
    peak = equity.cummax()
    dd = equity/peak - 1.0
    return float(dd.min())

def best_threshold_by_sharpe(proba, rets):
    if len(proba) != len(rets) or len(proba) == 0:
        return 0.5
    grid = np.linspace(0.4, 0.7, 61)
    best_thr, best_s = 0.5, -1e9
    for thr in grid:
        sig = (proba >= thr).astype(int)
        strat = rets * sig
        mu, sigma = np.nanmean(strat), np.nanstd(strat) + 1e-12
        sharpe = mu / sigma
        if sharpe > best_s:
            best_s, best_thr = float(sharpe), float(thr)
    return best_thr

def best_threshold_by_calmar(proba, rets):
    if len(proba) != len(rets) or len(proba) == 0:
        return 0.5
    grid = np.linspace(0.4, 0.7, 61)
    best_thr, best_c = 0.5, -1e9
    for thr in grid:
        sig = (proba >= thr).astype(int)
        strat = rets * sig
        ret = (1 + pd.Series(strat)).prod() - 1
        dd = abs(max_drawdown(strat)) + 1e-12
        calmar = ret / dd if dd > 0 else -1e-9
        if calmar > best_c:
            best_c, best_thr = float(calmar), float(thr)
    return best_thr

def tscv_with_embargo(n, n_splits, test_size, embargo):
    start = n - n_splits*test_size
    if start < 0: start = 0
    for s in range(start, n, test_size):
        test_start, test_end = s, min(s + test_size, n)
        train_end = max(0, test_start - embargo)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) > 0 and len(test_idx) > 0 and test_end - test_start >= max(5, embargo//2 + 1):
            yield train_idx, test_idx

def time_series_cv_ensemble(X, y, future_ret, n_splits=5, test_size_min=60, seed=42, thr_method="youden", embargo=0):
    n = len(X)
    if n < 80:
        return {"note": "Poucos dados para CV robusta (mín. ~80 amostras)."}, None, None, None, None, None
    n_splits_safe, test_size_safe = safe_tscv_params(n, n_splits, test_size_min)
    if n_splits_safe < 2:
        return {"note": f"Amostra insuficiente para dividir {n_splits}x com teste={test_size_min}. Reduza o período, o 'test_size' ou os 'splits'."}, None, None, None, None, None

    y_pred_proba = np.full(n, np.nan, dtype=float)
    thresholds = []
    last_models = None

    splitter = tscv_with_embargo(n, n_splits_safe, test_size_safe, embargo) if embargo>0 else TimeSeriesSplit(n_splits=n_splits_safe, test_size=test_size_safe).split(range(n))

    for train_idx, test_idx in splitter:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        rets_te = future_ret[test_idx]

        hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=500, random_state=seed, early_stopping=True)
        xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=seed, tree_method="hist")
        lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1)

        use_lgb = (len(X_tr) >= 150)

        hgb_cal = fit_calibrated(hgb, X_tr, y_tr, method="sigmoid")
        xgb_cal = fit_calibrated(xgb, X_tr, y_tr, method="sigmoid")
        models = [hgb_cal, xgb_cal]
        if use_lgb:
            lgb_cal = fit_calibrated(lgb, X_tr, y_tr, method="sigmoid")
            models.append(lgb_cal)

        probs = [m.predict_proba(X_te)[:,1] for m in models]
        proba = np.mean(probs, axis=0)

        if thr_method == "retorno":
            thr_fold = best_threshold_by_return(proba, rets_te)
        elif thr_method == "sharpe":
            thr_fold = best_threshold_by_sharpe(proba, rets_te)
        elif thr_method == "calmar":
            thr_fold = best_threshold_by_calmar(proba, rets_te)
        else:
            fpr, tpr, thr = roc_curve(y_te, proba); j = tpr - fpr
            thr_fold = thr[int(np.argmax(j))]

        thresholds.append(float(thr_fold))
        y_pred_proba[test_idx] = proba
        last_models = models

    mask = ~np.isnan(y_pred_proba)
    if mask.sum() == 0:
        return {"note": "Falha ao gerar previsões OOS."}, None, None, None, None, None

    y_true = y[mask]; y_prob = y_pred_proba[mask]
    metrics = {
        "accuracy": float(accuracy_score(y_true, (y_prob>=0.5).astype(int))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, (y_prob>=0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n_oos": int(mask.sum()),
        "threshold_avg": float(np.nanmean(thresholds)) if thresholds else 0.5,
        "adj_splits": int(n_splits_safe),
        "adj_test_size": int(test_size_safe)
    }
    return metrics, y_prob, y_true, thresholds, last_models, mask

# ================== Sidebar ==================
b3 = load_b3_tickers()
st.sidebar.header("⚙️ Configurações")

# Session defaults
defaults = dict(
    theme_choice="Escuro",
    show_sma50=False,
    show_sma200=False,
    use_ml=False,
    horizon=1,
    splits=5,
    test_size=60,
    thr_method_label="Sharpe OOS",
    min_prob=0.55,
    neutral_band=0.05,
    use_trend=True,
    allow_contrarian=True,
    contrarian_max_dist=-0.05,
    cost_bps=6,
    slip_bps=3,
    min_hold=2,
    # storage for results
    ml_trained=False,
    ml_proba_next=None,
    ml_metrics=None,
    ml_sig=None,
    ml_dates=None,
    ml_rets_oos=None,
    ml_cum_strat=None,
    ml_cum_bh=None,
    ml_dd_strat=None,
    ml_vol_strat=None,
)
for k,v in defaults.items():
    st.session_state.setdefault(k, v)

# Tema fixo: escuro
set_plotly_template("Escuro")

# Busca / seleção ticker
q = st.sidebar.text_input("Buscar empresa ou ticker", "", key="search_q",
                          help="Digite o código (ex.: PETR4) ou parte do nome (ex.: Petrobras).")
res = search_b3(q) if q else b3
ticker = st.sidebar.selectbox("Selecione o ticker", res["ticker"], key="ticker_select",
                              help="Somente tickers da B3 (.SA).")

# Período
st.sidebar.markdown("---")
quick = st.sidebar.selectbox("Período rápido", ["Personalizado", "6M", "1A", "YTD"], index=2, key="quick_period",
                             help="Atalhos de período.")
today = date.today()
if quick == "6M":
    start_default = today - timedelta(days=182)
elif quick == "1A":
    start_default = today - timedelta(days=365)
elif quick == "YTD":
    start_default = date(today.year, 1, 1)
else:
    start_default = today - timedelta(days=365)
start = st.sidebar.date_input("Início", start_default, key="start_date")
end = st.sidebar.date_input("Fim", today, key="end_date")

# Médias no gráfico
st.sidebar.markdown("---")
st.sidebar.markdown("**Médias no gráfico:**")
show_sma50 = st.sidebar.checkbox("Mostrar SMA50 (médio prazo)", value=st.session_state["show_sma50"], key="show_sma50")
show_sma200 = st.sidebar.checkbox("Mostrar SMA200 (longo prazo)", value=st.session_state["show_sma200"], key="show_sma200")

# ML Básico
st.sidebar.markdown("---")
st.sidebar.markdown("**Previsão (ML) — pesada**")
# 🔀 Modo simples + presets
simple_mode = st.sidebar.checkbox(
    "Modo simples (usar presets)",
    value=st.session_state.get("simple_mode", True),
    key="simple_mode",
    help="Deixa tudo fácil: escolha um preset (Conservador/Balanceado/Agressivo) e o app ajusta os parâmetros avançados automaticamente."
)

if simple_mode:
    st.sidebar.markdown("**Presets de estratégia**")
    preset_choice = st.sidebar.selectbox(
        "Escolha um preset", ["Conservador","Balanceado","Agressivo"],
        key="preset_choice",
        help="Define um conjunto pronto de parâmetros (banda, min_prob, tendência, contrarian, CV, custos, holding)."
    )
    colp1, colp2 = st.sidebar.columns(2)

    if colp1.button("Aplicar preset", key="btn_apply_preset"):
        if st.session_state["preset_choice"] == "Conservador":
            st.session_state.update(dict(
                thr_method_label="Calmar OOS",
                min_prob=0.62, neutral_band=0.06,
                use_trend=True, allow_contrarian=False, contrarian_max_dist=-0.03,
                splits=5, test_size=60, cost_bps=8, slip_bps=5, min_hold=3
            ))
        elif st.session_state["preset_choice"] == "Balanceado":
            st.session_state.update(dict(
                thr_method_label="Sharpe OOS",
                min_prob=0.58, neutral_band=0.05,
                use_trend=True, allow_contrarian=True, contrarian_max_dist=-0.05,
                splits=5, test_size=60, cost_bps=6, slip_bps=3, min_hold=2
            ))
        else:  # Agressivo
            st.session_state.update(dict(
                thr_method_label="Retorno OOS (backtest)",
                min_prob=0.54, neutral_band=0.03,
                use_trend=False, allow_contrarian=True, contrarian_max_dist=-0.08,
                splits=4, test_size=40, cost_bps=6, slip_bps=3, min_hold=1
            ))
        st.session_state["ml_trained"] = False
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    if colp2.button("Aplicar + Treinar", key="btn_apply_and_train"):
        st.session_state["ml_trained"] = False
        st.session_state["train_now"] = True
else:
    st.session_state["train_now"] = False
use_ml = st.sidebar.checkbox("Ativar previsão com ML", value=st.session_state["use_ml"], key="use_ml", help="Liga o modelo preditivo (ensemble). Exige mais processamento. Treine para ver probabilidade e backtest.")
# básico em sidebar
st.sidebar.markdown("*Básico*")
st.sidebar.selectbox("Horizonte (dias)", [1,5,10], key="horizon", help="Em quantos dias à frente o modelo tenta prever se o preço sobe. 1d = mais rápido/volátil; 5–10d = mais suave. Aumente se o sinal estiver instável; reduza se quiser mais reatividade.")
st.sidebar.selectbox("Método do limiar", ["Sharpe OOS","Calmar OOS","Retorno OOS (backtest)","Youden (acerto)"], key="thr_method_label", help="Regra para escolher a probabilidade mínima que vira sinal de compra no backtest. Sharpe: suavidade/consistência; Calmar: retorno com menor drawdown; Retorno: mais agressivo; Youden: maximiza taxa de acerto.")
st.sidebar.slider("Confiança mínima (min_prob)", 0.50, 0.75, st.session_state["min_prob"], 0.01, key="min_prob", help="Filtro extra: só entra se a probabilidade do modelo for ≥ este valor. Suba para ter menos trades e mais seletos; desça para ter mais trades (com mais ruído).")

# avançado no sidebar (colapsado visualmente via expander)

if not simple_mode:
    with st.sidebar.expander("Opções avançadas (custos, banda, tendência, contrarian, CV, holding)"):

        st.slider(
            "Banda neutra (± p.p. em torno de 50%)",
            0.00, 0.12, st.session_state["neutral_band"], 0.01, key="neutral_band",
            help="Zona de indecisão (ex.: ±0.05 = 45–55%), onde não opera. Aumente para filtrar ruído quando a probabilidade estiver perto de 50%."
        )

        st.checkbox(
            "Operar long apenas se Preço > SMA200",
            value=st.session_state["use_trend"], key="use_trend",
            help="Só permite compra se o preço estiver acima da média de 200 dias (tendência longa positiva). Ajuda a operar a favor da tendência."
        )

        st.checkbox(
            "Permitir contrarian em sobrevenda (RSI<30)",
            value=st.session_state["allow_contrarian"], key="allow_contrarian",
            help="Autoriza compra contra a tendência quando houver sobrevenda (RSI≤30). Use com cuidado e combine com a distância à SMA20."
        )

        st.slider(
            "Distância máx. à SMA20 (contrarian) — negativo = abaixo",
            -0.20, 0.00, st.session_state["contrarian_max_dist"], 0.01, key="contrarian_max_dist",
            help="Limite de quão abaixo da SMA20 o preço pode estar para permitir contrarian. Ex.: -0.05 = até 5% abaixo. Valores mais negativos deixam o contrarian mais agressivo."
        )

        d1, d2 = st.columns(2)
        d1.slider(
            "Nº de divisões (walk-forward CV)",
            3, 8, st.session_state["splits"], 1, key="splits",
            help="Quantas janelas de validação temporal. Mais splits = validação mais robusta, mas exige mais dados. Com poucos dados, use 3–4."
        )
        d2.slider(
            "Tamanho do bloco de teste (dias)",
            20, 120, st.session_state["test_size"], 5, key="test_size",
            help="Dias de cada janela de teste. 30–60 costuma equilibrar. Se faltar dados, reduza; se quiser janelas mais longas, aumente."
        )

        e1, e2, e3 = st.columns(3)
        e1.number_input(
            "Custo por trade (bps)", 0, 50, st.session_state["cost_bps"], key="cost_bps",
            help="Custo por lado (compra ou venda) em basis points. 10 bps = 0,10%. Use 6–10 bps como base; aumente para ativos menos líquidos."
        )
        e2.number_input(
            "Slippage (bps)", 0, 50, st.session_state["slip_bps"], key="slip_bps",
            help="Escorregão de execução por lado (diferença entre preço teórico e o que pegou). 3–10 bps; aumente em ativos pouco líquidos."
        )
        e3.number_input(
            "Dias mínimos em posição", 0, 10, st.session_state["min_hold"], key="min_hold",
            help="Após entrar, mantém a posição pelo menos X dias. Ajuda a reduzir entra-e-sai e whipsaw. 1–3 dias é comum."
        )

# ================== Header & Data ==================
st.title("📊 Análise Didática de Ações da B3 — v10.5")
st.caption("Somente tickers da B3 (.SA) — dados do Yahoo Finance - Vai Corinthians!")

if not is_known_b3_ticker(st.session_state["ticker_select"]):
    st.error("Ticker fora da lista da B3."); st.stop()
with st.spinner("Baixando dados..."):
    df = fetch_data(st.session_state["ticker_select"], st.session_state["start_date"], st.session_state["end_date"])
if df.empty:
    st.warning("Sem dados disponíveis para este período."); st.stop()
df = add_indicators(df, want_sma50=st.session_state["show_sma50"], want_sma200=st.session_state["show_sma200"])
events = annotate_events(df)

price = float(df["Close"].iloc[-1])
sma20 = float(df["SMA20"].iloc[-1])
rsi_val = float(df["RSI14"].iloc[-1])
delta20 = (price/sma20-1)*100 if sma20 else np.nan
regime = None
if "SMA200" in df.columns and not df["SMA200"].isna().iloc[-1]:
    regime = "Acima da SMA200 (tendência longa positiva)" if price > df["SMA200"].iloc[-1] else "Abaixo da SMA200 (tendência longa negativa)"

# ================== Banner Resumo em 10s ==================
def _chip(txt, color):
    return f"<span class='chip' style='background:{color};padding:4px 8px;border-radius:999px;color:#111;font-weight:600;margin-right:6px;'>{txt}</span>"

chip_sma = ("Bem abaixo da média", "#FCA5A5") if delta20 <= -7 else \
           ("Abaixo da média", "#FDE68A") if delta20 < -2 else \
           ("Perto da média", "#A7F3D0") if delta20 < 2 else \
           ("Acima da média", "#93C5FD") if delta20 < 7 else \
           ("Bem acima da média", "#C4B5FD")

chip_rsi = ("Sobrevenda (≤30)", "#86EFAC") if rsi_val <= 30 else \
           ("Neutro (30–70)", "#E5E7EB") if rsi_val < 70 else \
           ("Sobrecompra (≥70)", "#FCA5A5")

st.markdown("### ⚡ Resumo em 10s", unsafe_allow_html=True)
cA, cB = st.columns([0.62, 0.38])

with cA:
    st.markdown(
        _chip(chip_sma[0], chip_sma[1]) +
        _chip(chip_rsi[0], chip_rsi[1]) +
        (_chip("Longo prazo ↑", "#BBF7D0") if regime and "positiva" in regime else
         _chip("Longo prazo ↓", "#FECACA") if regime else ""),
        unsafe_allow_html=True
    )

with cB:
    if st.session_state.get("ml_trained") and st.session_state.get("ml_proba_next") is not None:
        figg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=st.session_state["ml_proba_next"]*100,
            number={'suffix': "%"},
            gauge={'axis': {'range': [0,100]}, 'bar': {'thickness': 0.25}, 'threshold': {'line': {'color': "white",'width': 2}, 'value': 50}},
            domain={'x':[0,1],'y':[0,1]},
            title={'text': f"Prob. de alta ({int(st.session_state['horizon'])}d)"}
        ))
        figg.update_layout(height=160, margin=dict(l=10,r=10,t=40,b=0))
        st.plotly_chart(figg, use_container_width=True)
    else:
        st.caption("Ative e treine o ML para ver a probabilidade de alta.")

# KPIs
k1,k2,k3 = st.columns(3)
k1.metric("Fechamento", f"R$ {price:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
k2.metric("Δ vs SMA20", f"{delta20:+.2f}%")
k3.metric("RSI(14)", f"{rsi_val:.1f}")
if regime: st.caption(regime)

# ================== Tabs ==================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Gráfico", "📚 Indicadores", "🤖 ML", "🧪 Backtest", "ℹ️ Glossário", "📊 Confiabilidade & Trades", "🧠 NeuralProphet"
])


# ---- Tab 1: Gráfico ----
with tab1:
    def plot_price(df, show_sma50, show_sma200):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Preço"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA20"))
        if show_sma50 and "SMA50" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50"))
        if show_sma200 and "SMA200" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA200"], name="SMA200"))
        # Eventos: afastamento forte
        ext_idxs = df.index[events["far_below"]]
        if len(ext_idxs) > 0:
            fig.add_trace(go.Scatter(x=df.loc[ext_idxs,"Date"], y=df.loc[ext_idxs,"Close"],
                                     mode="markers", marker=dict(size=8, symbol="triangle-up"),
                                     name="Afastado da SMA20 (−7% ou mais)"))
        # Sinais ML, se houver
        if st.session_state.get("ml_trained") and st.session_state.get("ml_sig") is not None:
            sig = st.session_state["ml_sig"]
            dates_oos = st.session_state["ml_dates"]
            idxs = np.where(sig==1)[0]
            if len(idxs) > 0:
                fig.add_trace(go.Scatter(x=dates_oos[idxs], y=df.set_index("Date").reindex(dates_oos)["Close"].values[idxs],
                                         mode="markers", marker=dict(size=9, symbol="star"),
                                         name="Sinal ML (long)"))
        fig.update_layout(title=f"{st.session_state['ticker_select']} — Preço e Médias", xaxis_title="Data", yaxis_title="Preço (R$)")
        st.plotly_chart(fig, use_container_width=True)

    def plot_rsi_tab(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI(14)"))
        fig.add_hline(y=70, line_dash="dash"); fig.add_hline(y=30, line_dash="dash")
        os_idxs = df.index[events["rsi_os"]]
        if len(os_idxs) > 0:
            fig.add_trace(go.Scatter(x=df.loc[os_idxs,"Date"], y=df.loc[os_idxs,"RSI14"],
                                     mode="markers", marker=dict(size=7, symbol="diamond"),
                                     name="RSI≤30 (sobrevenda)"))
        fig.update_layout(title=f"{st.session_state['ticker_select']} — RSI(14)", xaxis_title="Data", yaxis_title="RSI")
        st.plotly_chart(fig, use_container_width=True)

    plot_price(df, st.session_state["show_sma50"], st.session_state["show_sma200"])
    plot_rsi_tab(df)
    st.info("Dica: cole PETR4, VALE3, ITUB4... Se faltar .SA, o app adiciona.")

# ---- Tab 2: Indicadores ----
with tab2:
    st.markdown("### 💡 O que o gráfico está tentando te contar")
    # 1) SMA20
    st.markdown("#### 🪜 1. Entendendo a SMA20 — “a linha da média”")
    st.markdown(
        "A **SMA20** é como a média dos últimos 20 preços de fechamento — a **linha de equilíbrio** que mostra a direção geral do preço.\n\n"
        "• Se o **preço está acima** da linha, há **força** (tendência de alta).\n"
        "• Se **está abaixo**, há **fraqueza** (tendência de queda)."
    )
    st.markdown(f"👉 No caso de **{st.session_state['ticker_select']}**, o preço atual é **R$ {price:,.2f}**, cerca de **{delta20:+.2f}%** em relação à média dos últimos 20 dias.".replace(",", "X").replace(".", ",").replace("X","."))

    if delta20 <= -2:
        st.warning("A ação está **abaixo da média** — sinal de **fraqueza de curto prazo**.")
    elif delta20 < 2:
        st.info("O preço está **próximo da média** — o mercado está **em equilíbrio**.")
    else:
        st.success("O preço está **acima da média** — sinal de **força de curto prazo**.")

    st.caption("É como se o preço pudesse ficar “afastado da linha” por um tempo; quando isso acontece, pode haver **exagero** — como uma **corda muito esticada**.")

    # 2) RSI
    st.markdown("#### ⚖️ 2. Entendendo o RSI(14) — “o termômetro da força”")
    st.markdown("Pense no **RSI** como um termômetro de energia do mercado. Vai de **0 a 100** e mostra quem está dominando: compradores ou vendedores.")
    st.table(pd.DataFrame({
        "Faixa": ["70 a 100", "50", "0 a 30"],
        "Situação": ["Sobrecompra", "Neutro", "Sobrevenda"],
        "O que significa": ["Subiu rápido demais — pode corrigir pra baixo.", "Equilíbrio entre compra e venda.", "Caiu rápido demais — pode reagir pra cima."]
    }))
    st.markdown(f"No caso de **{st.session_state['ticker_select']}**, o **RSI(14)** está em **{rsi_val:.1f}**.")
    if rsi_val >= 70:
        st.warning("Está em **sobrecompra** — subiu rápido; pode corrigir.")
    elif rsi_val <= 30:
        st.success("Está em **sobrevenda** — caiu rápido; pode reagir em breve.")
    else:
        st.info("Está em **zona neutra** — o mercado está **equilibrado**.")

    # 3) Juntando
    st.markdown("#### 🧩 3. Juntando as duas informações")
    if (delta20 <= -2) and (rsi_val <= 35):
        st.info("“Caiu bastante e **pode dar um respiro** em breve.” (pressão de venda **perdendo força**)")
    elif (delta20 >= 2) and (rsi_val >= 65):
        st.warning("“Subiu bastante e **pode descansar**.” (compra ficando **esticada**)")
    else:
        st.info("“Quadro **equilibrado** — sem sinal forte de excesso.”")

    # 4) Comportamento e resumo
    st.markdown("#### 💬 Em resumo")
    resumo_rows = []
    resumo_rows.append(["SMA20", "Preço comparado à média de 20 dias", 
        "Preço bem abaixo da média — ação pressionada." if delta20 <= -7 else
        "Preço abaixo da média — tendência fraca." if delta20 < -2 else
        "Preço perto da média — equilíbrio." if delta20 < 2 else
        "Preço acima da média — força." if delta20 < 7 else
        "Preço bem acima da média — atenção a exageros."
    ])
    resumo_rows.append(["RSI(14)", "Energia do mercado (0–100)",
        "Sobrevenda (≤30) — pode reagir." if rsi_val <= 30 else
        "Neutro (30–70) — equilíbrio." if rsi_val < 70 else
        "Sobrecompra (≥70) — pode corrigir."
    ])
    resumo_rows.append(["Conclusão geral", "Combinação de média e força (preço + RSI)",
        "Fraca, mas pode haver repique." if (delta20 <= -2 and rsi_val <= 35) else
        "Forte, mas atenção a correções." if (delta20 >= 2 and rsi_val >= 65) else
        "Equilíbrio — sem sinal forte."
    ])
    st.table(pd.DataFrame(resumo_rows, columns=["Indicador","O que está mostrando","Significado prático"]))

    with st.expander("🕯️ Como ler candles (clique para ver)"):
        st.markdown("""
- **Candle** mostra Abertura, Máxima, Mínima e Fechamento do período.
- Corpo cheio: **fechou acima da abertura** (alta). Corpo vazio/escuro: **fechou abaixo** (baixa).
- **Pavios** (linhas finas) mostram onde o preço foi mas **não ficou**.
- Sequências de candles fortes indicam **impulso**; sombras longas indicam **reversões** possíveis.
""")

# ---- Tab 3: ML ----
with tab3:
    st.subheader("Previsão")
    st.caption(f"Horizonte: **{int(st.session_state['horizon'])}d** • Limiar: **{st.session_state['thr_method_label']}** • Confiança mínima: **{st.session_state['min_prob']:.2f}**")

    if st.session_state.get("train_now", False) or st.button("Treinar/Atualizar modelo", type="primary"):
        with st.spinner("Treinando e validando (walk-forward)..."):
            d, X, y, future_ret, feat_cols = build_features(df, horizon=int(st.session_state["horizon"]))
            finite_rows = np.isfinite(X).all(axis=1)
            d, X, y, future_ret = d.loc[finite_rows].reset_index(drop=True), X[finite_rows], y[finite_rows], future_ret[finite_rows]

            if len(X) < 80:
                st.warning("Poucos dados úteis após sanitização (NaN/Inf). Aumente o período, reduza o horizonte ou ajuste test_size/splits.")
            else:
                thr_lbl = st.session_state["thr_method_label"]
                key = "retorno" if thr_lbl.startswith("Retorno") else ("sharpe" if thr_lbl.startswith("Sharpe") else ("calmar" if thr_lbl.startswith("Calmar") else "youden"))
                embargo = int(st.session_state["horizon"])
                metrics, y_prob, y_true, thresholds, last_models, oos_mask = time_series_cv_ensemble(
                    X, y, future_ret, n_splits=int(st.session_state["splits"]), test_size_min=int(st.session_state["test_size"]), thr_method=key, embargo=embargo
                )
                if isinstance(metrics, dict) and "note" in metrics and y_prob is None:
                    st.warning(metrics["note"] + " — Tente **um período maior**, **test_size menor** ou **menos splits**.")
                else:
                    # Próximo passo
                    proba_next = None
                    if last_models is not None and len(d) > 0:
                        x_next = d[feat_cols].values[-1:].copy()
                        proba_next = float(np.mean([m.predict_proba(x_next)[:,1] for m in last_models]))

                    # Sinais a partir do limiar médio e filtros
                    thr_avg = metrics["threshold_avg"]
                    prob_oos = y_prob
                    rets_oos = future_ret[oos_mask]
                    dates_oos = d.loc[oos_mask, "Date"].values
                    px_oos = d.loc[oos_mask, "Close"].values
                    sma200_oos = d.loc[oos_mask, "SMA200"].values
                    rsi_oos = d.loc[oos_mask, "rsi"].values
                    dist20_oos = d.loc[oos_mask, "dist_sma20"].values

                    sig = (prob_oos >= thr_avg).astype(int)
                    low_b, high_b = 0.5 - st.session_state["neutral_band"], 0.5 + st.session_state["neutral_band"]
                    neutral = (prob_oos >= low_b) & (prob_oos <= high_b)
                    sig[neutral] = 0
                    sig[prob_oos < st.session_state["min_prob"]] = 0
                    if st.session_state["use_trend"]:
                        above_trend = np.isfinite(sma200_oos) & (px_oos > sma200_oos)
                        contrarian = (rsi_oos < 30) & (dist20_oos <= st.session_state["contrarian_max_dist"])
                        allow = above_trend | (st.session_state["allow_contrarian"] & contrarian)
                        sig = sig * allow.astype(int)
                    if st.session_state["min_hold"] > 0 and len(sig) > 0:
                        locked = 0
                        for i in range(len(sig)):
                            if sig[i] == 1 and locked == 0:
                                locked = int(st.session_state["min_hold"])
                            if locked > 0:
                                sig[i] = 1
                                locked -= 1
                    # custos
                    changes = np.diff(np.concatenate([[0], sig.astype(int)]))
                    per_side_cost = (st.session_state["cost_bps"] + st.session_state["slip_bps"]) / 10000.0
                    txn_costs = np.zeros_like(prob_oos, dtype=float)
                    for t, ch in enumerate(changes):
                        if ch != 0:
                            txn_costs[t] -= per_side_cost

                    strat = rets_oos * sig + txn_costs
                    cum_strat = (1 + pd.Series(strat)).cumprod() - 1
                    cum_bh = (1 + pd.Series(rets_oos)).cumprod() - 1
                    dd_strat = max_drawdown(strat)
                    vol_strat = float(np.nanstd(strat))

                    # guardar em sessão para outras abas
                    st.session_state.update(dict(
                        ml_trained=True,
                        ml_proba_next=proba_next,
                        ml_metrics=metrics,
                        ml_sig=sig,
                        ml_dates=dates_oos,
                        ml_rets_oos=rets_oos,
                        ml_cum_strat=cum_strat.values,
                        ml_cum_bh=cum_bh.values,
                        ml_dd_strat=dd_strat,
                        ml_vol_strat=vol_strat,
                    ))

    # Mostrar métricas se já treinado
    if st.session_state.get("ml_trained") and st.session_state.get("ml_metrics"):
        m = st.session_state["ml_metrics"]
        colA, colB, colC, colD, colE = st.columns(5)
        colA.metric("Acurácia (OOS) — % de acertos OOS", f"{m['accuracy']*100:.1f}%")
        colB.metric("Balanced Acc. — acerto ajustado por classe", f"{m['balanced_accuracy']*100:.1f}%")
        colC.metric("ROC AUC", f"{m['roc_auc']:.3f}")
        colD.metric("Brier — qualidade da probabilidade (↓ melhor)", f"{m['brier']:.3f}")
        colE.metric("OOS", f"{m['n_oos']}")
        st.caption(f"CV: splits={m['adj_splits']} • test_size={m['adj_test_size']} • Limiar: {st.session_state['thr_method_label']} • Embargo: {int(st.session_state['horizon'])}d")

        proba_next = st.session_state.get("ml_proba_next", None)
        if proba_next is not None:
            st.metric(f"Prob. de alta em {int(st.session_state['horizon'])} dia(s)", f"{proba_next*100:.1f}%")

        # Callout dinâmico
        auc = m['roc_auc']; brier = m['brier']
        msg_auc = "vantagem forte" if auc >= 0.65 else "vantagem moderada" if auc >= 0.60 else "vantagem pequena" if auc >= 0.53 else "sinal fraco (≈ acaso)"
        msg_brier = "probabilidades bem calibradas" if brier < 0.23 else "probabilidades razoáveis" if brier < 0.26 else "probabilidades pouco informativas"
        if auc < 0.55:
            st.warning(f"Sinal **fraco** ({msg_auc}; {msg_brier}). Considere **aumentar min_prob**, usar **Sharpe/Calmar**, ativar **tendência** e **holding**.")
        else:
            st.success(f"Sinal com **{msg_auc}** ({msg_brier}). Ajuste **min_prob** e **banda neutra** para dosar seletividade vs. número de trades.")

# ---- Tab 4: Backtest ----
with tab4:
    if not (st.session_state.get("ml_trained") and st.session_state.get("ml_cum_strat") is not None):
        st.info("Treine o modelo na aba **ML** para ver o backtest.")
    else:
        perf_df = pd.DataFrame({
            "Data": st.session_state["ml_dates"],
            "Estratégia (long nos sinais)": st.session_state["ml_cum_strat"],
            "Buy & Hold (OOS)": st.session_state["ml_cum_bh"],
        }).melt("Data", var_name="Série", value_name="Retorno Acumulado")
        figp = px.line(perf_df, x="Data", y="Retorno Acumulado", color="Série", title="Backtest — Retorno Acumulado (fora da amostra)")
        st.plotly_chart(figp, use_container_width=True)

        dd = st.session_state["ml_dd_strat"]; vol = st.session_state["ml_vol_strat"]
        c1,c2 = st.columns(2)
        c1.metric("Máx. drawdown (estratégia)", f"{dd*100:.1f}%")
        c2.metric("Vol (por passo)", f"{vol*100:.2f}%")
        st.caption("Inclui custos por lado e holding, conforme definidos no painel.")

        # Curva de confiabilidade
        if st.session_state.get("ml_metrics") and st.session_state.get("ml_dates") is not None:
            # Reconstituir a prob OOS
            # Para simplicidade, recomputamos bins usando a prob não guardada explicitamente — omitido neste UX para leveza.
            st.info("Para análise de calibração detalhada, volte à aba **ML** após treinar (a versão completa pode incluir a curva).")

# ---- Tab 5: Glossário ----
with tab5:
    st.markdown("### 📚 Glossário rápido")
    st.markdown("""
- **Candle**: barra de um período; mostra Abertura, Máxima, Mínima, Fechamento.
- **SMA (Média Móvel Simples)**: média dos fechamentos de um período; mostra tendência.
- **RSI (Índice de Força Relativa)**: velocidade das altas/quedas recentes (0–100).
- **Sobrecompra/Sobrevenda**: regiões (≥70 / ≤30) onde há risco de correção/repique.
- **Embargo**: espaço entre treino e teste para evitar vazamento de informação.
- **Sharpe/Calmar**: medidas de qualidade do retorno (suavidade e controle de quedas).
- **Holding**: dias mínimos mantendo a posição depois de entrar.
- **bps**: basis points; 10 bps = 0,10%.
""")

with tab5:


    # ===== v11: NeuralProphet Tab =====
with tab7:
    st.subheader("🧠 NeuralProphet — previsão de tendência")
    st.caption("Modelo neural inspirado no Prophet (tendência + sazonalidades) — documentação oficial.")

    # Seletor do horizonte futuro (quantos dias você quer prever)
    np_h = st.number_input(
        "Dias para prever (futuro)",
        min_value=1, max_value=365, value=30, step=1,
        help="Quantos dias à frente você quer projetar com o NeuralProphet."
    )

    if not _NP_AVAILABLE:
        st.warning("NeuralProphet não está instalado no ambiente. Adicione `neuralprophet` ao requirements.txt e faça o deploy novamente.")
        st.code("pip install neuralprophet", language="bash")
    else:
        try:
            # Usa o mesmo DataFrame de preços já carregado no app
            price_df = None
            if 'df_price' in locals():
                price_df = df_price.copy()
            elif 'df' in locals():
                price_df = df.copy()

            if price_df is None:
                st.info("Carregue um ticker para habilitar o NeuralProphet.")
            else:
                # Preparação no padrão NeuralProphet (colunas ds, y)
                if 'Date' in price_df.columns:
                    price_df = price_df.rename(columns={'Date':'ds'})
                if 'Close' in price_df.columns:
                    price_df = price_df.rename(columns={'Close':'y'})
                if 'Adj Close' in price_df.columns and 'y' not in price_df.columns:
                    price_df = price_df.rename(columns={'Adj Close':'y'})

                np_df = price_df[['ds','y']].dropna().copy()

                # Modelo (simples) — tendência linear + sazonalidade semanal/anual
                m = NeuralProphet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                )
                # Frequência de dias úteis da B3: 'B'
                metrics = m.fit(np_df, freq='B')

                # Cria datas futuras e prevê; mantém previsões históricas para contexto
                df_future = m.make_future_dataframe(np_df, periods=int(np_h), n_historic_predictions=True)
                forecast = m.predict(df_future)

                # Gráfico: histórico (y) + previsão (yhat1)
                import plotly.express as px
                fig = px.line(forecast, x='ds', y='yhat1', title='Projeção (yhat1) — NeuralProphet')
                fig.add_scatter(x=np_df['ds'], y=np_df['y'], mode='lines', name='Fechamento (histórico)')
                st.plotly_chart(fig, use_container_width=True)

                # Tendência prevista no fim do horizonte
                split_date = np_df['ds'].max()
                last_close = float(np_df.iloc[-1]['y'])
                fut_part = forecast[forecast['ds'] > split_date]
                last_forecast = float((fut_part.tail(1)['yhat1']).values[0]) if not fut_part.empty else float(forecast.tail(1)['yhat1'].values[0])
                pct = (last_forecast/last_close - 1.0)*100.0
                st.metric("Tendência prevista (fim do horizonte)", "alta" if pct>=0 else "baixa", f"{pct:.2f}%")

                # Componente de tendência (se disponível)
                if 'trend' in forecast.columns:
                    fig_t = px.line(forecast, x='ds', y='trend', title='Componente de Tendência')
                    st.plotly_chart(fig_t, use_container_width=True)

                # Download das previsões futuras
                fut = fut_part[['ds','yhat1']].rename(columns={'ds':'Data','yhat1':'Preço previsto'})
                if not fut.empty:
                    st.download_button(
                        "Baixar previsão futura (CSV)",
                        data=fut.to_csv(index=False).encode("utf-8"),
                        file_name="neuralprophet_forecast.csv",
                        mime="text/csv"
                    )

                st.info("Baseado no guia rápido e na classe `NeuralProphet` da documentação oficial.")
        except Exception as e:
            st.error(f"Falha ao rodar o NeuralProphet: {e}")

    
    # ===== v10.5: Confiabilidade & Trades =====
    try:
        # Tenta obter y_true e y_prob OOS do estado
        y_true = st.session_state.get("ml_oos_y_true")
        y_prob = st.session_state.get("ml_oos_y_prob")
        prices = st.session_state.get("oos_prices") or st.session_state.get("prices_close")
        signals = st.session_state.get("oos_signals") or st.session_state.get("signals_ml")
        horizon = st.session_state.get("horizon", 1)
        cost_bps = st.session_state.get("cost_bps", 6)
        slip_bps = st.session_state.get("slip_bps", 3)
    
        if y_true is not None and y_prob is not None and len(y_true)==len(y_prob) and len(y_true)>20:
            pt, pp = _compute_reliability(np.array(y_true), np.array(y_prob), n_bins=10)
            if pt is not None:
                st.subheader("Curva de Confiabilidade")
                fig = _plot_reliability_plotly(pt, pp)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Não foi possível calcular a curva de confiabilidade para este conjunto.")
        else:
            st.info("Treine o modelo para ver a curva de confiabilidade (probabilidades OOS).")
    
        st.subheader("Tabela de Trades (OOS)")
        df_trades = None
        # Se já existir uma tabela pronta no estado, usa
        if st.session_state.get("trades_oos_df") is not None:
            df_trades = st.session_state["trades_oos_df"]
        else:
            # Caso contrário, tenta construir uma tabela simples a partir de preços e sinais
            if isinstance(prices, (pd.Series,)) and isinstance(signals, (pd.Series,)):
                df_trades = _build_trades_table(prices, signals, horizon, cost_bps, slip_bps)
    
        if df_trades is not None and not df_trades.empty:
            st.dataframe(df_trades.head(50), use_container_width=True)
            colA, colB = st.columns(2)
            with colA:
                st.metric("Trades", len(df_trades))
            with colB:
                st.metric("Retorno médio/trade", f"{df_trades['Retorno %'].mean():.2f}%")
            csv = df_trades.to_csv(index=False).encode("utf-8")
            st.download_button("Baixar CSV de trades", data=csv, file_name="trades_oos.csv", mime="text/csv")
        else:
            st.info("Nenhum trade OOS disponível para montar a tabela.")
    except Exception as e:
        st.warning(f"Não foi possível renderizar a aba Confiabilidade & Trades: {e}")



def _plot_reliability_plotly(prob_true, prob_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Perfeito"))
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Modelo"))
    fig.update_layout(
        title="Curva de Confiabilidade (Calibration)",
        xaxis_title="Probabilidade prevista",
        yaxis_title="Frequência observada"
    )
    return fig



# 📊 Análise Didática de Ações da B3 — v10.5.2 (Presets + Dark + Confiabilidade & Trades)

Ferramenta educacional para **ler** indicadores técnicos de ações da **B3** 🇧🇷 e **testar** uma estratégia preditiva com *machine learning* (ensemble).  
Agora com **Modo simples + Presets**, **tema escuro fixo**, **Curva de Confiabilidade** em **Plotly** e **Tabela de Trades OOS**.

> **Aviso:** Este projeto é didático. **Não é recomendação de investimento.**

---

## ✨ O que há nesta versão
- **Presets (Modo simples)**: Conservador | Balanceado | Agressivo — ajustam min_prob, banda, tendência, contrarian, CV, custos e holding com 1 clique.
- **Tema escuro fixo**: sem seletor; `.streamlit/config.toml` já define o modo escuro.
- **Aba “📊 Confiabilidade & Trades”**:
  - **Curva de Confiabilidade (Calibration)** em **Plotly** (sem `matplotlib`), mostrando se as probabilidades são “honestas”.
  - **Tabela de Trades (OOS)** com retorno por trade, duração e **download CSV**.
- **Cache leve** para dados (`@st.cache_data`) e textos mais didáticos nas métricas.
- **Correção de abas**: 6 labels ⇒ 6 variáveis (evita `ValueError` no `st.tabs`).

---

## 🚀 Como rodar
```bash
# Python 3.10+ recomendado
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Abra no navegador o endereço exibido (geralmente http://localhost:8501).

### Publicar no Streamlit Cloud
1. Envie o projeto para o **GitHub** (mantenha a estrutura).  
2. No Streamlit Cloud: **New app** → selecione repo/branch → `streamlit_app.py`.  
3. Deploy. Se der erro de dependências, verifique `requirements.txt`.

---

## 🧱 Estrutura
```
.
├── streamlit_app.py
├── b3_utils.py
├── data/
│   └── b3_tickers.csv
├── .streamlit/
│   └── config.toml   # tema escuro
├── requirements.txt
└── README.md
```

**Dados**: Yahoo Finance via `yfinance` (ajuste automático de proventos).  
**Lista de tickers**: `data/b3_tickers.csv` (pode atualizar quando quiser).

---

## 🧭 Passo a passo rápido
1. Digite o **ticker** (ex.: `PETR4` — o app adiciona `.SA`).  
2. Escolha o **período** (`6M`, `1A`, `YTD` ou intervalo).  
3. (Opcional) Ligue o **ML** → **Modo simples** → escolha um **Preset** → **Aplicar + Treinar**.  
4. Veja as **métricas** e a **probabilidade de alta** (gauge).  
5. Confira o **Backtest** (estratégia x Buy & Hold).  
6. Use a aba **📊 Confiabilidade & Trades** para validar as probabilidades e revisar operações.  
7. Consulte **Indicadores** e **Glossário** para reforçar o aprendizado.

---

## 🧠 Modo simples + Presets
Para simplicidade, o **Modo simples** aplica automaticamente parâmetros **coerentes** de risco/retorno.

| Preset | Objetivo | Limiar (otimização) | `min_prob` | Banda neutra | Tendência (Preço>SMA200) | Contrarian (RSI<30) | Dist. máx. à SMA20 | Holding | Splits/Teste | Custos |
|---|---|---|---|---|---|---|---|---|---|---|
| **Conservador** | reduzir *drawdown* | **Calmar OOS** | 0.62 | ±0.06 | ON | OFF | −0,03 | 3 dias | 5 / 60 | 8/5 bps |
| **Balanceado** | suavizar | **Sharpe OOS** | 0.58 | ±0.05 | ON | ON | −0,05 | 2 dias | 5 / 60 | 6/3 bps |
| **Agressivo** | retorno bruto | **Retorno OOS** | 0.54 | ±0.03 | OFF | ON | −0,08 | 1 dia | 4 / 40 | 6/3 bps |

> Aplique um preset e **ajuste manualmente** se desejar.

---

## ⚙️ Parâmetros (explicação simples)

### Básico
- **Horizonte (dias)**: em quantos dias o modelo tenta prever **se sobe**.  
  `1d` = reativo/volátil; `5–10d` = mais suave.  
- **Método do limiar**: como escolhemos o **corte** que vira **sinal**.  
  - **Sharpe OOS**: consistência (bom risco/retorno).  
  - **Calmar OOS**: retorno com **menor queda máxima**.  
  - **Retorno OOS**: retorno bruto (mais agressivo).  
  - **Youden (acerto)**: foca taxa de acerto.  
- **Confiança mínima (`min_prob`)**: só opera se a probabilidade for ≥ este valor.  
  ↑ `min_prob` → menos trades, mais seletos. ↓ `min_prob` → mais trades (mais ruído).

### Avançado
- **Banda neutra**: perto de 50% (ex.: ±0,05) onde **não opera** (evita ruído).  
- **Tendência (Preço > SMA200)**: opera **a favor** da tendência longa.  
- **Contrarian (RSI<30)**: permite **repique** em sobrevenda; use com a **Dist. à SMA20**.  
- **Dist. máx. à SMA20 (contrarian)**: limite negativo (ex.: −0,05 = até 5% abaixo).  
- **Splits / Test Size**: *walk‑forward* (validação temporal). Poucos dados → `splits=3–4`, `test_size=30–60`.  
- **Custos / Slippage** (bps): custos por lado realistas (6–10 bps / 3–10 bps).  
- **Holding mínimo**: mantém a posição por X dias (evita entra‑e‑sai).

---

## 🤖 Como o modelo funciona (didático)
1. **Features**: indicadores (SMA, RSI, retornos, volatilidade, distância à média etc.).  
2. **Alvo**: 1 se `Close[t+h] > Close[t]` (subiu em `h` dias), senão 0.  
3. **Ensemble**: HGB + XGBoost + LightGBM → média de probabilidades.  
4. **Calibração**: ajusta para que 70% signifique “~70% de chance” em média.  
5. **Walk‑forward**: treina no passado e testa no futuro (OOS).  
6. **Limiar + filtros**: transformam probabilidade em **sinal** (banda, tendência, contrarian).  
7. **Backtest**: compara estratégia x Buy & Hold, com **custos** e **slippage**.

**Métricas**:  
- **Acurácia** (simples), **Balanced Acc.** (ajusta desequilíbrio), **ROC AUC** (0,5=sorte; 0,60+ indica vantagem), **Brier** (qualidade da prob.), **OOS** (tamanho do teste).

---

## 📊 Confiabilidade & Trades (como usar)
- **Curva de Confiabilidade**: após treinar, abre a aba e veja a curva “Modelo” vs. “Perfeito”.  
  - Linha próxima da diagonal ⇒ probabilidades **bem calibradas**.  
  - Abaixo da diagonal ⇒ modelo **otimista** (superestima chances).  
  - Acima da diagonal ⇒ modelo **conservador** (subestima).  
- **Tabela de Trades (OOS)**: lista operações fora da amostra do treino (mais realistas).  
  - Baixe o **CSV** para estudar top/bottom trades, duração e padrões.

---

## 🔧 Solução de problemas
- **Erro no `st.tabs` (vars vs labels)**: v10.5.2 já corrige (6 labels ⇒ 6 variáveis).  
- **Dependências**: esta versão dispensa `matplotlib` (usa Plotly).  
- **CV temporal quebrando**: reduza `splits` e/ou `test_size`.  
- **IDs duplicados**: não repita `key` entre widgets.

---

## 📄 Licença e créditos
Defina a licença (ex.: **MIT**).  
**Dados:** Yahoo Finance (`yfinance`) • **Gráficos:** Plotly • **ML:** scikit‑learn, XGBoost, LightGBM • **UI:** Streamlit.

---

Dúvidas? Abra uma *issue* ou mande mensagem. 😉

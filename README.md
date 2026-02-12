# Stocron by RTR | The Ultimate Market Intelligence Engine
> **30+ Years of History. Zero Noise. Pure Alpha.**

[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker&logoColor=white&style=for-the-badge)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/AI-Chronos--T5%20%2B%20LightGBM-orange?style=for-the-badge)](https://huggingface.co/amazon/chronos-t5-base)
[![YouTube](https://img.shields.io/badge/YouTube-RTR%20Unfiltered-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@RTR-Unfiltered)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📉 The Problem
Retail investors are playing a rigged game. Most analysis tools suffer from three critical flaws:
1.  **Recency Bias:** They only look at the last 5-10 years of a Bull market.
2.  **Survivorship Bias:** They ignore delisted companies, making history look safer than it actually was.
3.  **Linear Thinking:** They use simple indicators (RSI, MA) in a complex, non-linear macro environment.

You cannot build generational wealth by just "looking at the chart." You need to understand the **Regime**, the **Fundamentals**, and the **Macro-Economic** backdrop.

[<img src="https://img.youtube.com/vi/zA_az-prPns/hqdefault.jpg" target="_blank" width="720" height="480"
/>](https://www.youtube.com/embed/zA_az-prPns)

## 🛡️ Why This Tool Exists
**Stocron** was built for the **RTR Unfiltered** ecosystem to answer one question:
*"If I had the same data, tools, and computing power as a hedge fund, but with 30 years of unfiltered Indian market context, how would I trade?"*

This is not just a screener. It is a **Time Machine**. It allows you to validate strategies across decades, factoring in inflation, oil prices, and corporate governance failures.

---

## ⚡ Super Powers

### 1. The "Dual-Brain" AI Core 🧠
We don't rely on a single model. Stocron uses a hybrid architecture:
* **The Forecaster (Chronos-T5):** A Transformer-based model (pretrained by Amazon) that treats stock charts like a language to predict future price sequences.
* **The Classifier (LightGBM):** A gradient-boosting decision engine that analyzes hundreds of features (Financials, Macro, Technicals) to generate a binary `BUY`/`HOLD` signal.
* **The Explainer (Qwen2.5):** A local LLM that generates human-readable explanations for AI decisions.
<img width="720" height="480" alt="capture_260206_211651" src="https://github.com/user-attachments/assets/2c5e38f4-d9c6-459a-9e61-4343c440b0b4" />

### 2. Time Travel & Survivorship ⏳
Most backtests are fake because they test on companies that exist *today*.
* **Delisted Database:** We track companies that failed, ensuring your strategy survives the worst.
* **Time Travel Engine:** Go back to Jan 1st, 2008. The system "forgets" the future, forcing the AI to trade only on what it knew then.
<img width="720" height="480" alt="capture_260206_211525" src="https://github.com/user-attachments/assets/a18bbf4c-af11-43ce-bd9e-5bf584fec8d4" />

### 3. Scenario Simulator 🌪️
Don't just predict; prepare.
* *What if Crude Oil hits $120?*
* *What if the Repo Rate jumps to 8%?*
The simulator stresses your portfolio against hypothetical macro-economic shocks.
<img width="720" height="480" alt="capture_260206_211600" src="https://github.com/user-attachments/assets/da1589dd-46a6-4854-89ae-cfebdbd4a411" />

### 4. Governance Guard 🕵️
The tool doesn't just chase profits; it filters out fraud.
* **Beneish M-Score:** Detects earnings manipulation.
* **Altman Z-Score:** Predicts bankruptcy risk.
* **Piotroski F-Score:** Measures fundamental strength.
<img width="720" height="480" alt="capture_260206_211845" src="https://github.com/user-attachments/assets/818436c3-3ada-441b-9ea8-83f98ac9a862" />

---

## 🏗️ Architecture

The system is containerized for stability and reproducibility.

```text
       ┌──────────────┐
       │     USER     │
       └──────┬───────┘
              │ (Browser)
              ▼
    ┌────────────────────┐
    │    STREAMLIT UI    │
    └─────────┬──────────┘
              │ (Request)
              ▼
    ┌────────────────────┐          ┌──────────────────┐
    │  ANALYSIS ENGINE   │◄─────────│   DATA LAKE DB   │
    └─────────┬──────────┘          │  (JSON / CSV)    │
              │                     └─────────▲────────┘
              │ (Inference)                   │
              │                               │ (Fetch)
    ┌─────────▼──────────┐          ┌─────────┴────────┐
    │     ML MODELS      │          │   INTERNET / NSE │
    │ ┌────────────────┐ │          └──────────────────┘
    │ │ Chronos-T5     │ │
    │ ├────────────────┤ │
    │ │ LightGBM       │ │
    │ ├────────────────┤ │
    │ │ Qwen2.5 (LLM)  │ │
    │ └────────────────┘ │
    └────────────────────┘
```

---

## Installation & Usage
> We strongly recommend running Stocron via Docker to avoid dependency hell.

### Prerequisites

1. Docker Desktop installed and running.
2. If you have an NVIDIA GPU installed on your PC, open the docker-compose.yml file and refer to line 35 for GPU acceleration configuration.

```bash
# 1) Clone
git clone https://github.com/RTR95/stockIt_by_RTR.git
cd stockIt_by_RTR

# 2) Build + start the app container
docker-compose up -d --build

# 3) Download missing live symbols (recommended)
docker exec -it stocron-by-rtr python 2-download_all_stocks.py --with-financials

# If you don't want Warren Buffet's scoring, just download stock info without financials
docker exec -it stocron-by-rtr python 2-download_all_stocks.py

# 4) Delisted stocks (recommended for survivorship bias)
docker exec -it stocron-by-rtr python scripts/download_delisted_stocks.py --export
docker exec -it stocron-by-rtr python scripts/download_delisted_stocks.py --download --years 30
docker exec -it stocron-by-rtr python 2-download_all_stocks.py --build-db-only

# 5) (Optional) Enable real crude oil data
export FRED_API_KEY="your_fred_api_key"

# 6) Download ML models (optional; required for ML forecaster/explainer)
docker exec -it stocron-by-rtr python 3-download_models.py

# 7) Train classifier (required for ML signals)
docker exec -it stocron-by-rtr python 4-train_classifier.py

Open👉 http://localhost:8501

Note: The container mounts `./models` and `./config` so downloaded models and
auto-updated settings persist on the host.
```
---

## 🧩 ML Logic & Explainability
We believe in "Unfiltered" truth. The AI shouldn't be a black box.

SHAP Integration: We use SHAP (SHapley Additive exPlanations) to break down every signal.

Example Output: "The model is Bullish because 'ROE > 15%' (+20 impact) and 'Oil Prices Dropped' (+10 impact), despite 'RSI being Overbought' (-5 impact)."
<img width="720" height="480" alt="capture_260206_211651" src="https://github.com/user-attachments/assets/3d0ab635-8f04-48d5-a9d1-8916a33c28e4" />

---

## FAQs and General Information

### 1. How to refresh all stocks to latest metrics?
Use the same command again, but add --download-all so it refreshes every symbol (not just missing ones). Example:
```bash
docker-compose exec stocron-by-rtr python 2-download_all_stocks.py --with-financials --download-all
```
Notes:
- Without --download-all, it only downloads missing symbols.
- If you want to refresh a subset, use --refresh-symbol with comma-separated tickers.

### 2. RTR's recommended ML list
I use *chronos-t5-base* for forecaster, *CatBoost* for classifier with **NO LLM** for Explainer.
This is a weekend project and I haven't designed the tool for GPU acceleration. So LLM will run on CPU which would be a lot slower (10 to 15mins per stock). If you still want, go for it.

### 3. How accurate are the predictions? What's the backtest performance?
Stocron is a research tool, not a promise of returns. It focuses on long-horizon regime awareness and survivorship-aware testing rather than headline accuracy. Performance varies by period, sector, and macro regime. Use the time-travel engine to validate your own strategy across decades, including bad regimes and delisted stocks.

### 4. How is this different from normal technical indicators?
It's not just RSI/MA. Stocron blends a transformer forecaster, a fundamentals/macro-aware classifier, and explainability (SHAP + LLM) to evaluate regime, fundamentals, and macro context-not just price patterns.

### 5. Can I simulate macro events like oil shocks or rate hikes?
Yes. The scenario simulator lets you stress portfolios against macro shocks like crude spikes or rate jumps to see how signals and risk profiles change.

### 6. Does it give buy/sell signals or just research insights?
It generates a binary BUY/HOLD signal from the classifier plus explanations, but it’s designed as a research engine. The goal is decision support, not automated trading.

### 7. Can I use my own data sources or models?
The system is modular and data-lake driven (JSON/CSV). Advanced users can extend data sources or swap models, but *it's not a plug-and-play feature yet.*

### 8. Is this financial advice or just research?
**Research only.** It’s educational and analytical, not financial advice. Always use your own judgment and risk management.

---

## 📜 Disclaimer
This tool is for **educational and research purposes only**. It is built for the RTR Unfiltered community to analyze market logic. It is NOT financial advice. Markets are subject to risk. *Use your own brain before making financial decisions*.

## License
MIT License — see [LICENSE](LICENSE).


_**Built with 🧠 by RTR.**_

# 🤖 Botardo - Trading Bot

Bot de trading automatizado que combina **Smart Money Concepts (SMC)** y **Will Street Power of 3 (PO3)** para operar futuros de criptomonedas.

## 📁 Estructura del Proyecto

```
Botardo/
├── botardo.py           # 🤖 Bot completo (SMC + PO3 + Backtest)
├── download_data.py     # 📥 Descarga datos de Binance
├── requirements.txt     # 📦 Dependencias
├── Colab_Backtest.ipynb # ☁️  Notebook para Google Colab
├── data/                # 📊 Datos históricos
└── README.md            # 📖 Este archivo
```

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/TobiasActis/Botardo.git
cd Botardo

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

### 1. Descargar Datos

```bash
python download_data.py
```

### 2. Ejecutar Backtest

**Configuración Óptima (Recomendada):**
```bash
python botardo.py \
    --data_1m "data/BTCUSDT_1m_2024-01-01_to_now.csv" \
    --initial_capital 500 \
    --risk_per_trade 0.06 \
    --leverage 10 \
    --smc_rr 2.0 \
    --po3_min_rr 2.0 \
    --start "2024-01-01"
```

## ⚙️ Parámetros

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--initial_capital` | Capital inicial en USDT | 500 |
| `--risk_per_trade` | % de riesgo por trade | 0.06 (6%) |
| `--leverage` | Apalancamiento máximo | 10x |
| `--po3_min_rr` | Risk/Reward mínimo PO3 | 2.0:1 |
| `--smc_standalone` | Confluencia mínima SMC | 8 |
| `--smc_rr` | Risk/Reward ratio SMC | 2.0:1 |

## 📊 Estrategia

1. **Smart Money Concepts (SMC)**: Señales primarias
   - Order Blocks, Fair Value Gaps (FVG)
   - Break of Structure (BOS), Change of Character (CHoCH)
   - Umbral de confluencia: 8 puntos

2. **RSI Divergences**: Confirmación de reversiones ✨
   - Divergencias alcistas/bajistas para detectar reversiones tempranas
   - Zonas extremas estrictas: Overbought (>75) y Oversold (<25)
   - Impacto: +3 puntos de confluencia cuando hay divergencia alineada

3. **Liquidity Zones (BSL/SSL)**: Targeting inteligente ✨
   - Buy Side Liquidity (BSL): Máximos recientes arriba del precio
   - Sell Side Liquidity (SSL): Mínimos recientes abajo del precio
   - Take Profit ajustado automáticamente a zonas de liquidez
   - Impacto: +2 puntos de confluencia, mejor precisión en TPs

4. **EMA 12 Trend Filter**: Filtro de tendencia ✨ NUEVO v3
   - Solo LONG si precio > EMA12, solo SHORT si precio < EMA12
   - Elimina señales contra-tendencia (mejora win rate +10%)
   - Bonus +2 puntos si tendencia fuerte (>2% separación)
   - **Impacto crítico**: Win rate 65% → 75%, Sharpe 4.18 → 6.96

5. **Will Street PO3**: Confirmación adicional
   - Power of Three en velas 4h
   - Risk/Reward: 2:1

5. **Gestión de Riesgo**:
   - 6% del capital por trade
   - Stop Loss: 0.75 × ATR
   - Take Profit: Ajustado a liquidez o 1.5 × ATR (RR 2:1)
   - Apalancamiento: 10x

## 🎯 Resultados (Backtest 2024-2025)

### Sistema Profesional: RSI + Liquidity + EMA12 Trend Filter ✨

**Configuración Óptima** (SMC=8, RR=2.0, Risk=6%):
- 💰 **Retorno Total**: +39.99% (casi 2 años)
- 📈 **Retorno Anualizado**: ~20%
- 📊 **Sharpe Ratio**: 6.96 🚀 (Excepcional)
- 📉 **Max Drawdown**: 12.18% ⬇️⬇️
- 🎯 **Win Rate**: 75.00% 🔥🔥
- 🔢 **Total Trades**: 20 (selectivo)
- 💎 **Profit Factor**: 1.91 ⬆️⬆️

**Evolución del Sistema:**
- ✅ Win Rate: 44% → 65% → **75%** (sistema v3)
- ✅ Sharpe Ratio: 3.61 → 4.18 → **6.96** (calidad excepcional)
- ✅ Max Drawdown: 17.48% → 15.62% → **12.18%** (muy estable)
- ✅ Profit Factor: 1.34 → 1.44 → **1.91** (casi duplicado)

## ☁️ Google Colab

[Ejecutar en Colab](https://colab.research.google.com/github/TobiasActis/Botardo/blob/main/Colab_Backtest.ipynb)

## ⚠️ Advertencia

- NO es asesoramiento financiero
- Trading con apalancamiento es de alto riesgo
- Practica primero en testnet

## 👤 Autor

Tobias Actis - [GitHub](https://github.com/TobiasActis)

---

**⚡ Happy Trading! ⚡**

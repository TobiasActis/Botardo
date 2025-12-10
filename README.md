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

2. **RSI Divergences**: Confirmación de reversiones ✨ NUEVO
   - Divergencias alcistas/bajistas para detectar reversiones tempranas
   - Zonas extremas: Overbought (>70) y Oversold (<30)
   - Impacto: +3 puntos de confluencia cuando hay divergencia alineada

3. **Liquidity Zones (BSL/SSL)**: Targeting inteligente ✨ NUEVO
   - Buy Side Liquidity (BSL): Máximos recientes arriba del precio
   - Sell Side Liquidity (SSL): Mínimos recientes abajo del precio
   - Take Profit ajustado automáticamente a zonas de liquidez
   - Impacto: +2 puntos de confluencia, mejor precisión en TPs

4. **Will Street PO3**: Confirmación adicional
   - Power of Three en velas 4h
   - Risk/Reward: 2:1

5. **Gestión de Riesgo**:
   - 6% del capital por trade
   - Stop Loss: 0.75 × ATR
   - Take Profit: Ajustado a liquidez o 1.5 × ATR (RR 2:1)
   - Apalancamiento: 10x

## 🎯 Resultados (Backtest 2024-2025)

### Sistema Mejorado con RSI + Liquidity Zones ✨

**Configuración Óptima** (SMC=8, RR=2.0, Risk=6%):
- 💰 **Retorno Total**: +36.95% (2 años)
- 📈 **Retorno Anualizado**: ~17%
- 📊 **Sharpe Ratio**: 4.18 ⬆️ (+15.8%)
- 📉 **Max Drawdown**: 15.62% ⬇️ (-10.6%)
- 🎯 **Win Rate**: 65.52% ⬆️⬆️ (+48%)
- 🔢 **Total Trades**: 29
- 💎 **Profit Factor**: 1.44 ⬆️ (+7.5%)

**Mejoras vs Sistema Original:**
- ✅ Win Rate: 44% → 65.52% (+21.52 puntos porcentuales)
- ✅ Sharpe Ratio: 3.61 → 4.18 (mejor calidad de retornos)
- ✅ Max Drawdown: 17.48% → 15.62% (mayor estabilidad)
- ✅ Profit Factor: 1.34 → 1.44 (mejor rentabilidad por trade)

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

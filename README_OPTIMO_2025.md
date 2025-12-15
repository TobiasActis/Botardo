# 🏆 Botardo - Configuración Óptima 2025

## Resultados Backtest (BTCUSDT 45m, 2025)
- **Capital inicial:** $500
- **Balance final:** $900.97
- **P&L total:** +120.58%
- **Win rate:** 76.92%
- **Profit factor:** 10.12
- **Max drawdown:** -1.73%
- **Trades:** 26 (20 ganadores / 6 perdedores)

## Condiciones y Parámetros para Replicar

### Estrategia
- **Tipo:** Mean Reversion + Trend Filter + Salida Parcial
- **Timeframe:** 45 minutos
- **Activo:** BTCUSDT (BTCUSDT_45m.csv)

### Indicadores y Filtros
- **Bollinger Bands:** 20 períodos, 2 std
- **RSI:** 14 períodos
  - LONG: RSI < 20
  - SHORT: RSI > 80
- **EMA50/EMA200:**
  - Solo LONG si EMA50 > EMA200 y separación >2%
  - Solo SHORT si EMA50 < EMA200 y separación >2%
- **ATR:** 14 períodos
  - Filtro: ATR/ATR_avg > 1.2 y < 3.0
  - Filtro de vela extrema: rango vela < 2.0x ATR

### Gestión de Riesgo
- **Riesgo por trade:** 2%
- **Leverage:** 3x
- **Stop Loss:** 1.5% del entry
- **Take Profit:** 3% del entry (RR 1:2)
- **Trailing Stop:** 2.0x ATR
- **Gestión adaptativa:** Si 2+ pérdidas seguidas, riesgo se reduce a la mitad

### Lógica de Salida
- **Salida parcial:** 50% de la posición se cierra al TP
- **El 50% restante:** sigue con trailing stop/rsi_exit

### Ejecución
- **Script:** test_optimized.py
- **Data:** data/BTCUSDT_45m.csv
- **Periodo:** 2025-01-01 a 2025-12-11

---

> ¡Esta configuración logró resultados excepcionales! Guarda este README como referencia para futuras iteraciones o restauraciones.

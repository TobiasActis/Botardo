# Resumen de Backtest Botardo 45m (2019–2025)

## Configuración utilizada
- Símbolo: BTCUSDT
- Timeframe: 45 minutos
- Mercado: Futuros
- Capital inicial: $500
- Parámetros óptimos:
  - bb_std: 2
  - rsi_oversold: 20
  - rsi_overbought: 80
  - sl_pct: 0.015
  - tp_multiplier: 2
  - risk_per_trade: 0.02
  - leverage: 3
- Lógica: Coincide con README (condiciones de entrada/salida relajadas)
- Data: 2019–2025, resampleada desde 1m

## Resultados anuales

| Año  | Balance Final | Trades | Ganadores | Perdedores | Winrate |
|------|---------------|--------|-----------|------------|---------|
| 2019 | $727.90       | 14     | 12        | 2          | 85.7%   |
| 2020 | $660.10       | 14     | 9         | 5          | 64.3%   |
| 2021 | $1,548.96     | 48     | 38        | 10         | 79.2%   |
| 2022 | $1,240.56     | 38     | 30        | 8          | 78.9%   |
| 2023 | $572.25       | 6      | 4         | 2          | 66.7%   |
| 2024 | $566.95       | 8      | 6         | 2          | 75.0%   |
| 2025 | $758.24       | 22     | 15        | 7          | 68.2%   |

## Archivos clave
- Código: botardo.py, analisis_45m.py
- Data: data/BTCUSDT_45m.csv
- Resultados: results/resumen_45m_years.json

---

> Generado automáticamente el 15/12/2025

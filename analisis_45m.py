import pandas as pd
from botardo import Botardo

# Cargar datos 45m
file = 'data/BTCUSDT_45m.csv'
df = pd.read_csv(file, parse_dates=['timestamp'])

resultados = {}

for year in range(2018, 2026):
    df_year = df[df['timestamp'].dt.year == year].reset_index(drop=True)
    if len(df_year) == 0:
        print(f'No data for {year}')
        continue
    bot = Botardo(capital=500, risk_per_trade=0.02, leverage=3)
    bot.bb_std = 2
    bot.rsi_oversold = 20
    bot.rsi_overbought = 80
    bot.sl_pct = 0.015
    bot.tp_multiplier = 2
    trades = bot.run_backtest(df_year)
    print(f'\nAño: {year}')
    bot.print_results()
    if len(trades) > 0:
        resultados[year] = {
            'capital_inicial': 500,
            'balance_final': bot.capital,
            'trades': len(trades),
            'ganadores': int((trades['pnl_usd'] > 0).sum()),
            'perdedores': int((trades['pnl_usd'] < 0).sum()),
            'winrate': 100 * (trades['pnl_usd'] > 0).sum() / len(trades),
        }
    else:
        resultados[year] = {
            'capital_inicial': 500,
            'balance_final': bot.capital,
            'trades': 0,
            'ganadores': 0,
            'perdedores': 0,
            'winrate': 0,
        }

# Opcional: guardar resultados en JSON
import json
with open('results/resumen_45m_years.json', 'w') as f:
    json.dump(resultados, f, indent=2)

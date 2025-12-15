"""
Prueba los parámetros optimizados en 2025
"""

import pandas as pd
from botardo import Botardo

# Cargar datos de 2025
df_45m = pd.read_csv('data/BTCUSDT_45m.csv', parse_dates=['timestamp'])
df_2025 = df_45m[df_45m['timestamp'] >= '2025-01-01'].reset_index(drop=True)


print("="*70)
print("BACKTEST - PARÁMETROS ACTUALES")
print("="*70)

# Crear bot con parámetros actuales
bot = Botardo(capital=500, risk_per_trade=0.03, leverage=3)

# Configura aquí los parámetros que quieras probar:
bot.bb_std = 1.5
bot.rsi_oversold = 30
bot.rsi_overbought = 70
bot.sl_pct = 0.015
bot.tp_multiplier = 2.5

print(f"\nParámetros utilizados:")
print(f"  BB std: {bot.bb_std}")
print(f"  RSI: {bot.rsi_oversold}/{bot.rsi_overbought}")
print(f"  SL: {bot.sl_pct*100}%")
print(f"  TP multiplier: {bot.tp_multiplier}x")
print(f"\nPeriodo: {df_2025['timestamp'].iloc[0].date()} hasta {df_2025['timestamp'].iloc[-1].date()}")
print(f"Velas: {len(df_2025)}\n")

# Ejecutar backtest
trades = bot.run_backtest(df_2025)

if len(trades) == 0:
    print("❌ No se generaron trades")
    exit()

# Calcular métricas
df_trades = pd.DataFrame(trades)
total_pnl = df_trades['pnl_pct'].sum()
wins = len(df_trades[df_trades['pnl_pct'] > 0])
losses = len(df_trades[df_trades['pnl_pct'] < 0])
be = len(df_trades[df_trades['pnl_pct'] == 0])
win_rate = wins / len(df_trades) * 100

# Balance final
df_trades['balance'] = 500 + df_trades['pnl_usd'].cumsum()
final_balance = df_trades['balance'].iloc[-1]

# Max drawdown
df_trades['cummax'] = df_trades['balance'].cummax()
df_trades['dd'] = (df_trades['balance'] - df_trades['cummax']) / df_trades['cummax'] * 100
max_dd = df_trades['dd'].min()

# Profit factor
win_sum = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].sum() if wins > 0 else 0
loss_sum = abs(df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].sum()) if losses > 0 else 1
pf = win_sum / loss_sum if loss_sum > 0 else 0


# Presentación mejorada de resultados
print("\n" + "="*50)
print("RESUMEN DE RESULTADOS 2025")
print("="*50)

print(f"| {'Capital inicial':<20} | $ {500.00:>10.2f} |")
print(f"| {'Balance final':<20} | $ {final_balance:>10.2f} |")
print(f"| {'P&L total':<20} | $ {final_balance - 500:>10.2f} ({total_pnl:+.2f}%) |")
print(f"| {'Trades totales':<20} | {len(df_trades):>10} | ({wins}W / {losses}L / {be}BE)")
print(f"| {'Win rate':<20} | {win_rate:>10.2f} % |")
print(f"| {'Profit factor':<20} | {pf:>10.2f} |")
print(f"| {'Max drawdown':<20} | {max_dd:>10.2f} % |")
print("="*50)

# Advertencia si el resultado es peor que el capital inicial
if final_balance < 500:
    print("\n⚠️  El balance final es menor al capital inicial. Revisa los parámetros o el contexto de mercado.\n")

# Guardar resultados para comparación futura
import json
import os
resultados = {
    'fecha': str(pd.Timestamp.now()),
    'capital_inicial': 500.0,
    'balance_final': float(final_balance),
    'pnl_total': float(final_balance - 500),
    'pnl_pct': float(total_pnl),
    'trades': int(len(df_trades)),
    'wins': int(wins),
    'losses': int(losses),
    'be': int(be),
    'winrate': float(win_rate),
    'profit_factor': float(pf),
    'max_drawdown': float(max_dd),
    'parametros': {
        'bb_std': bot.bb_std,
        'rsi_oversold': bot.rsi_oversold,
        'rsi_overbought': bot.rsi_overbought,
        'sl_pct': bot.sl_pct,
        'tp_multiplier': bot.tp_multiplier
    }
}
os.makedirs('results', exist_ok=True)
with open('results/last_result.json', 'w') as f:
    json.dump(resultados, f, indent=2)



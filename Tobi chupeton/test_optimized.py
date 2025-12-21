
"""
Prueba los parámetros optimizados en todos los años disponibles
"""


import pandas as pd
from botardo import Botardo
import json
import os


# Selección automática de archivos por año
archivos = []
for anio in range(2018, 2026):
    if anio in [2018, 2019]:
        archivo = f"data/BTCUSDT_SPOT_1m_{anio}-01-01_to_{anio}-12-31.csv"
    elif anio == 2020:
        archivo = f"data/BTCUSDT_1m_{anio}-01-01_to_{anio}-12-31.csv"
    elif anio == 2021:
        archivo = f"data/BTCUSDT_1m_2021-01-01_to_now.csv"
    elif anio == 2022:
        archivo = f"data/BTCUSDT_1m_2022-01-01_to_now.csv"
    elif anio == 2023:
        archivo = f"data/BTCUSDT_1m_2023-01-01_to_2023-12-31.csv"
        if not os.path.exists(archivo):
            archivo = f"data/BTCUSDT_1m_2024-01-01_to_now.csv"  # fallback si no existe 2023 propio
    elif anio == 2024:
        archivo = f"data/BTCUSDT_1m_2024-01-01_to_now.csv"
    elif anio == 2025:
        archivo = f"data/BTCUSDT_1m_2025-01-01_to_now.csv"
    archivos.append((str(anio), archivo))

os.makedirs('results', exist_ok=True)
resumen = {}

for anio, archivo in archivos:
    if not os.path.exists(archivo):
        print(f"Archivo no encontrado: {archivo}")
        continue
    print("\n" + "="*70)
    print(f"BACKTEST - {anio} - {archivo}")
    print("="*70)
    df = pd.read_csv(archivo, parse_dates=['timestamp'])
    # Si el archivo contiene datos de varios años, filtrar solo el año correspondiente
    df = df[df['timestamp'].dt.year == int(anio)].reset_index(drop=True)
    if len(df) == 0:
        print(f"No hay datos para {anio} en {archivo}")
        continue

    bot = Botardo(capital=500, risk_per_trade=0.03, leverage=3)
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
    print(f"\nPeriodo: {df['timestamp'].iloc[0].date()} hasta {df['timestamp'].iloc[-1].date()}")
    print(f"Velas: {len(df)}\n")

    trades = bot.run_backtest(df)
    if len(trades) == 0:
        print("❌ No se generaron trades")
        continue

    df_trades = pd.DataFrame(trades)
    total_pnl = df_trades['pnl_pct'].sum()
    wins = len(df_trades[df_trades['pnl_pct'] > 0])
    losses = len(df_trades[df_trades['pnl_pct'] < 0])
    be = len(df_trades[df_trades['pnl_pct'] == 0])
    win_rate = wins / len(df_trades) * 100
    df_trades['balance'] = 500 + df_trades['pnl_usd'].cumsum()
    final_balance = df_trades['balance'].iloc[-1]
    df_trades['cummax'] = df_trades['balance'].cummax()
    df_trades['dd'] = (df_trades['balance'] - df_trades['cummax']) / df_trades['cummax'] * 100
    max_dd = df_trades['dd'].min()
    win_sum = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].sum() if wins > 0 else 0
    loss_sum = abs(df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].sum()) if losses > 0 else 1
    pf = win_sum / loss_sum if loss_sum > 0 else 0

    print("\n" + "="*50)
    print(f"RESUMEN DE RESULTADOS {anio}")
    print("="*50)
    print(f"| {'Capital inicial':<20} | $ {500.00:>10.2f} |")
    print(f"| {'Balance final':<20} | $ {final_balance:>10.2f} |")
    print(f"| {'P&L total':<20} | $ {final_balance - 500:>10.2f} ({total_pnl:+.2f}%) |")
    print(f"| {'Trades totales':<20} | {len(df_trades):>10} | ({wins}W / {losses}L / {be}BE)")
    print(f"| {'Win rate':<20} | {win_rate:>10.2f} % |")
    print(f"| {'Profit factor':<20} | {pf:>10.2f} |")
    print(f"| {'Max drawdown':<20} | {max_dd:>10.2f} % |")
    print("="*50)
    if final_balance < 500:
        print("\n⚠️  El balance final es menor al capital inicial. Revisa los parámetros o el contexto de mercado.\n")

    resultados = {
        'fecha': str(pd.Timestamp.now()),
        'anio': anio,
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
    resumen[anio] = resultados
    with open(f'results/result_{anio}.json', 'w') as f:
        json.dump(resultados, f, indent=2)

# Guardar resumen general
with open('results/resumen_anual.json', 'w') as f:
    json.dump(resumen, f, indent=2)



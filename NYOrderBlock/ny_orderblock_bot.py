import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NYOrderBlockBot:
    def __init__(self, capital=500, risk_per_trade=0.01, leverage=3, tz_offset=-5):
        self.initial_capital = capital
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.tz_offset = tz_offset

    def backtest(self, df_5m, df_1h=None):
        df = df_5m.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp_ny'] = df['timestamp'] + pd.Timedelta(hours=self.tz_offset)
        df['date'] = df['timestamp_ny'].dt.date
        results = []
        for day, day_df in df.groupby('date'):
            pre_session = day_df[(day_df['timestamp_ny'].dt.time >= pd.to_datetime('06:45').time()) &
                                 (day_df['timestamp_ny'].dt.time <= pd.to_datetime('07:30').time())]
            if pre_session.empty:
                continue
            last_candle = pre_session.iloc[-1]
            is_bullish = last_candle['close'] > last_candle['open']
            is_bearish = last_candle['close'] < last_candle['open']
            if is_bearish:
                entry_level = last_candle['low']
                sl = last_candle['high'] + 0.0003
                direction = 'SELL'
            elif is_bullish:
                entry_level = last_candle['high']
                sl = last_candle['low'] - 0.0003
                direction = 'BUY'
            else:
                continue
            session = day_df[(day_df['timestamp_ny'].dt.time >= pd.to_datetime('07:30').time()) &
                             (day_df['timestamp_ny'].dt.time <= pd.to_datetime('11:00').time())]
            if session.empty:
                continue
            entry_idx = None
            for idx, row in session.iterrows():
                if direction == 'SELL' and row['high'] >= entry_level:
                    entry_idx = idx
                    break
                elif direction == 'BUY' and row['low'] <= entry_level:
                    entry_idx = idx
                    break
            if entry_idx is None:
                continue
            entry_row = session.loc[entry_idx]
            entry_price = entry_level
            after_entry = session.loc[entry_idx:]
            tp = None
            for _, r in after_entry.iterrows():
                if direction == 'SELL' and r['low'] < entry_price:
                    tp = r['low']
                    break
                elif direction == 'BUY' and r['high'] > entry_price:
                    tp = r['high']
                    break
            if tp is None:
                continue
            # Filtro: solo operar si el TP está al menos a 0.1% del precio de entrada
            min_target_pct = 0.001  # 0.1%
            target_dist = abs(tp - entry_price) / entry_price
            if target_dist < min_target_pct:
                continue
            # Primer trade
            trades_today = 1
            # El trade se ejecuta aquí, calcular pnl y agregarlo
            # ...existing code for pnl and commission...
            first_trade_win = False
            if 'pnl' in locals():
                first_trade_win = pnl > 0
            # Segundo trade solo si el primero fue ganador
            if first_trade_win:
                entry_idx2 = None
                for idx2, row2 in after_entry.iloc[1:].iterrows():
                    if direction == 'SELL' and row2['high'] >= entry_price:
                        entry_idx2 = idx2
                        break
                    elif direction == 'BUY' and row2['low'] <= entry_price:
                        entry_idx2 = idx2
                        break
                if entry_idx2 is not None:
                    entry_row2 = after_entry.loc[entry_idx2]
                    entry_price2 = entry_price
                    after_entry2 = after_entry.iloc[entry_idx2:]
                    tp2 = None
                    for _, r2 in after_entry2.iterrows():
                        if direction == 'SELL' and r2['low'] < entry_price2:
                            tp2 = r2['low']
                            break
                        elif direction == 'BUY' and r2['high'] > entry_price2:
                            tp2 = r2['high']
                            break
                    if tp2 is not None:
                        target_dist2 = abs(tp2 - entry_price2) / entry_price2
                        if target_dist2 >= min_target_pct:
                            risk_amount2 = self.capital * self.risk_per_trade
                            size2 = risk_amount2 / abs(entry_price2 - sl) if abs(entry_price2 - sl) > 0 else 0
                            commission2 = abs(size2 * entry_price2 * 0.0004)
                            if direction == 'SELL':
                                sl_hit2 = any(r2['high'] >= sl for _, r2 in after_entry2.iterrows())
                                tp_hit2 = any(r2['low'] <= tp2 for _, r2 in after_entry2.iterrows())
                                pnl2 = (entry_price2 - tp2) * size2 if tp_hit2 and (not sl_hit2 or after_entry2[after_entry2['low'] <= tp2].index[0] < after_entry2[after_entry2['high'] >= sl].index[0]) else (sl - entry_price2) * size2 * -1
                            else:
                                sl_hit2 = any(r2['low'] <= sl for _, r2 in after_entry2.iterrows())
                                tp_hit2 = any(r2['high'] >= tp2 for _, r2 in after_entry2.iterrows())
                                pnl2 = (tp2 - entry_price2) * size2 if tp_hit2 and (not sl_hit2 or after_entry2[after_entry2['high'] >= tp2].index[0] < after_entry2[after_entry2['low'] <= sl].index[0]) else (entry_price2 - sl) * size2 * -1
                            pnl2 -= commission2
                            results.append({
                                'date': day,
                                'direction': direction,
                                'entry': entry_price2,
                                'sl': sl,
                                'tp': tp2,
                                'pnl': pnl2,
                                'commission': commission2
                            })
            risk_amount = self.capital * self.risk_per_trade
            # El apalancamiento solo afecta el margen requerido, no el tamaño de la posición para la comisión
            size = risk_amount / abs(entry_price - sl) if abs(entry_price - sl) > 0 else 0
            commission_rate = 0.0004  # 0.04% por trade (solo apertura)
            if direction == 'SELL':
                sl_hit = any(r['high'] >= sl for _, r in after_entry.iterrows())
                tp_hit = any(r['low'] <= tp for _, r in after_entry.iterrows())
                pnl = (entry_price - tp) * size if tp_hit and (not sl_hit or after_entry[after_entry['low'] <= tp].index[0] < after_entry[after_entry['high'] >= sl].index[0]) else (sl - entry_price) * size * -1
            else:
                sl_hit = any(r['low'] <= sl for _, r in after_entry.iterrows())
                tp_hit = any(r['high'] >= tp for _, r in after_entry.iterrows())
                pnl = (tp - entry_price) * size if tp_hit and (not sl_hit or after_entry[after_entry['high'] >= tp].index[0] < after_entry[after_entry['low'] <= sl].index[0]) else (entry_price - sl) * size * -1
            commission = abs(size * entry_price * commission_rate)
            pnl -= commission
            results.append({
                'date': day,
                'direction': direction,
                'entry': entry_price,
                'sl': sl,
                'tp': tp,
                'pnl': pnl,
                'commission': commission
            })
        results_df = pd.DataFrame(results)
        # Estadísticas y gráficos
        if not results_df.empty:
            winners = results_df[results_df['pnl'] > 0]
            losers = results_df[results_df['pnl'] < 0]
            winrate = len(winners) / len(results_df) * 100
            total_wins = winners['pnl'].sum() if not winners.empty else 0
            total_losses = abs(losers['pnl'].sum()) if not losers.empty else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            avg_win = winners['pnl'].mean() if not winners.empty else 0
            avg_loss = losers['pnl'].mean() if not losers.empty else 0
            best_trade = results_df['pnl'].max()
            worst_trade = results_df['pnl'].min()
            equity = [self.initial_capital]
            for pnl in results_df['pnl']:
                equity.append(equity[-1] + pnl)
            results_df['equity'] = equity[1:]
            results_df['cummax'] = results_df['equity'].cummax()
            results_df['drawdown'] = (results_df['equity'] - results_df['cummax']) / results_df['cummax']
            max_drawdown = results_df['drawdown'].min() * 100
            results_df['date'] = pd.to_datetime(results_df['date'])
            results_df['year'] = results_df['date'].dt.year
            results_df['month'] = results_df['date'].dt.to_period('M')
            yearly = results_df.groupby('year')['pnl'].sum() / self.initial_capital * 100
            monthly = results_df.groupby('month')['pnl'].sum() / self.initial_capital * 100
            print(f"Total trades: {len(results_df)} | Ganancia total: {results_df['pnl'].sum():.2f}")
            print(f"Winrate: {winrate:.2f}% | Profit Factor: {profit_factor:.2f} | Max Drawdown: {max_drawdown:.2f}%")
            print(f"Mejor Trade: {best_trade:.2f} | Peor Trade: {worst_trade:.2f}")
            print("\n% Ganancia por año:")
            print(yearly)
            print("\n% Ganancia por mes:")
            print(monthly)
            plt.figure(figsize=(12,5))
            plt.plot(results_df['date'], results_df['equity'], label='Equity Curve')
            plt.title('Equity Curve NY Order Block')
            plt.xlabel('Fecha')
            plt.ylabel('Equity')
            plt.legend()
            plt.tight_layout()
            plt.savefig('ny_orderblock_equity.png')
            plt.close()
            plt.figure(figsize=(10,4))
            monthly.plot(kind='bar', color='skyblue')
            plt.title('P&L Mensual NY Order Block')
            plt.ylabel('P&L')
            plt.tight_layout()
            plt.savefig('ny_orderblock_monthly.png')
            plt.close()
            print("\n✅ Gráficos guardados: ny_orderblock_equity.png y ny_orderblock_monthly.png")
        results_df.to_csv('ny_orderblock_trades.csv', index=False)
        print("\n✅ Trades guardados: ny_orderblock_trades.csv")
        return results_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='NY Order Block Bot (5m NY Session)')
    parser.add_argument('--data', required=True, help='Path to CSV data file (5m)')
    parser.add_argument('--data-1h', required=False, help='Path to CSV data file (1h, para filtro de tendencia)')
    parser.add_argument('--capital', type=float, default=500, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (0.01 = 1%)')
    parser.add_argument('--leverage', type=int, default=3, help='Leverage multiplier')
    parser.add_argument('--start', default='2022-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default='2025-12-31', help='End date YYYY-MM-DD')
    args = parser.parse_args()
    df = pd.read_csv(args.data, parse_dates=['timestamp'])
    df = df[(df['timestamp'] >= args.start) & (df['timestamp'] <= args.end)]
    df_1h = None
    if args.data_1h:
        df_1h = pd.read_csv(args.data_1h, parse_dates=['timestamp'])
        df_1h = df_1h[(df_1h['timestamp'] >= args.start) & (df_1h['timestamp'] <= args.end)]
    bot = NYOrderBlockBot(capital=args.capital, risk_per_trade=args.risk, leverage=args.leverage)
    bot.backtest(df, df_1h)

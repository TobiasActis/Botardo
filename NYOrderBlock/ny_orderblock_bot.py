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

    def backtest(self, df_5m):
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
            risk_amount = self.capital * self.risk_per_trade
            size = (risk_amount * self.leverage) / abs(entry_price - sl) if abs(entry_price - sl) > 0 else 0
            if direction == 'SELL':
                sl_hit = any(r['high'] >= sl for _, r in after_entry.iterrows())
                tp_hit = any(r['low'] <= tp for _, r in after_entry.iterrows())
                pnl = (entry_price - tp) * size if tp_hit and (not sl_hit or after_entry[after_entry['low'] <= tp].index[0] < after_entry[after_entry['high'] >= sl].index[0]) else (sl - entry_price) * size * -1
            else:
                sl_hit = any(r['low'] <= sl for _, r in after_entry.iterrows())
                tp_hit = any(r['high'] >= tp for _, r in after_entry.iterrows())
                pnl = (tp - entry_price) * size if tp_hit and (not sl_hit or after_entry[after_entry['high'] >= tp].index[0] < after_entry[after_entry['low'] <= sl].index[0]) else (entry_price - sl) * size * -1
            results.append({
                'date': day,
                'direction': direction,
                'entry': entry_price,
                'sl': sl,
                'tp': tp,
                'pnl': pnl
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
            yearly = results_df.groupby('year')['pnl'].sum()
            monthly = results_df.groupby('month')['pnl'].sum()
            print(f"Total trades: {len(results_df)} | Ganancia total: {results_df['pnl'].sum():.2f}")
            print(f"Winrate: {winrate:.2f}% | Profit Factor: {profit_factor:.2f} | Max Drawdown: {max_drawdown:.2f}%")
            print(f"Mejor Trade: {best_trade:.2f} | Peor Trade: {worst_trade:.2f}")
            print("\nP&L por año:")
            print(yearly)
            print("\nP&L por mes:")
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
    parser.add_argument('--capital', type=float, default=500, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (0.01 = 1%)')
    parser.add_argument('--leverage', type=int, default=3, help='Leverage multiplier')
    parser.add_argument('--start', default='2022-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default='2025-12-31', help='End date YYYY-MM-DD')
    args = parser.parse_args()
    df = pd.read_csv(args.data, parse_dates=['timestamp'])
    df = df[(df['timestamp'] >= args.start) & (df['timestamp'] <= args.end)]
    bot = NYOrderBlockBot(capital=args.capital, risk_per_trade=args.risk, leverage=args.leverage)
    bot.backtest(df)

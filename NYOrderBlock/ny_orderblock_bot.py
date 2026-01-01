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
        
        # Preparar datos 1h para filtro de tendencia
        trend_data = {}
        if df_1h is not None:
            df_1h = df_1h.copy()
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
            df_1h['timestamp_ny'] = df_1h['timestamp'] + pd.Timedelta(hours=self.tz_offset)
            df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
            df_1h['ema50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
            # Calcular ATR para filtro de volatilidad
            df_1h['tr'] = df_1h[['high', 'low', 'close']].apply(
                lambda x: max(x['high'] - x['low'], 
                             abs(x['high'] - x['close']), 
                             abs(x['low'] - x['close'])), axis=1)
            df_1h['atr'] = df_1h['tr'].rolling(window=14).mean()
            df_1h['atr_pct'] = (df_1h['atr'] / df_1h['close']) * 100  # ATR en %
            # Crear diccionario para lookup rápido
            for _, row in df_1h.iterrows():
                trend_data[row['timestamp_ny']] = {
                    'bullish': row['close'] > row['ema20'] and row['ema20'] > row['ema50'],
                    'bearish': row['close'] < row['ema20'] and row['ema20'] < row['ema50'],
                    'atr_pct': row['atr_pct'] if pd.notna(row['atr_pct']) else 0
                }
        
        results = []
        min_capital_threshold = self.initial_capital * 0.5  # Circuit breaker al 50%
        
        for day, day_df in df.groupby('date'):
            # Circuit breaker: parar si el capital cae por debajo del 50%
            if self.capital < min_capital_threshold:
                print(f"\n⚠️ Circuit breaker activado en {day}: Capital {self.capital:.2f} < {min_capital_threshold:.2f}")
                break
            # Procesar dos sesiones: Europa (EU) y Nueva York (NY)
            sessions = [
                ('EU', '01:00', '02:00', '02:00', '06:00'),  # Europa
                ('NY', '06:45', '07:30', '07:30', '11:00')   # Nueva York
            ]
            
            for session_name, pre_start, pre_end, sess_start, sess_end in sessions:
                pre_session = day_df[(day_df['timestamp_ny'].dt.time >= pd.to_datetime(pre_start).time()) &
                                     (day_df['timestamp_ny'].dt.time <= pd.to_datetime(pre_end).time())]
                if pre_session.empty:
                    continue
                last_candle = pre_session.iloc[-1]
                
                # Filtro de tamaño mínimo de vela (al menos 0.15% de rango)
                candle_range = abs(last_candle['high'] - last_candle['low']) / last_candle['close']
                if candle_range < 0.0015:  # 0.15%
                    continue
                
                is_bullish = last_candle['close'] > last_candle['open']
                is_bearish = last_candle['close'] < last_candle['open']
                
                # Filtro de momentum: solo velas con cuerpo decente (al menos 50% del rango)
                candle_body = abs(last_candle['close'] - last_candle['open'])
                candle_range_val = last_candle['high'] - last_candle['low']
                if candle_range_val == 0 or (candle_body / candle_range_val) < 0.5:
                    continue
                
                # Filtro de tendencia 1h: solo operar a favor
                if df_1h is not None and len(trend_data) > 0:
                    closest_time = None
                    for ts in trend_data.keys():
                        if ts <= last_candle['timestamp_ny']:
                            if closest_time is None or ts > closest_time:
                                closest_time = ts
                    if closest_time and closest_time in trend_data:
                        trend_info = trend_data[closest_time]
                        # Solo SELL en tendencia bajista, solo BUY en tendencia alcista
                        if is_bearish and not trend_info['bearish']:
                            continue
                        if is_bullish and not trend_info['bullish']:
                            continue
                    closest_time = None
                    min_diff = float('inf')
                    for ts in trend_data.keys():
                        diff = abs((last_candle['timestamp_ny'] - ts).total_seconds())
                        if diff < min_diff:
                            min_diff = diff
                            closest_time = ts
                    
                    if closest_time and min_diff < 3600:  # Máximo 1 hora de diferencia
                        trend = trend_data[closest_time]
                        # Solo operar si la tendencia coincide con la dirección
                        if is_bullish and not trend['bullish']:
                            continue
                        if is_bearish and not trend['bearish']:
                            continue
                
                if is_bearish:
                    entry_level = last_candle['low']
                    sl = last_candle['high'] + 0.001
                    direction = 'SELL'
                elif is_bullish:
                    entry_level = last_candle['high']
                    sl = last_candle['low'] - 0.001
                    direction = 'BUY'
                else:
                    continue
                session = day_df[(day_df['timestamp_ny'].dt.time >= pd.to_datetime(sess_start).time()) &
                                 (day_df['timestamp_ny'].dt.time <= pd.to_datetime(sess_end).time())]
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
                
                # Calcular TP como ratio fijo del riesgo (1:1 risk-reward)
                risk = abs(entry_price - sl)
                if direction == 'SELL':
                    tp = entry_price - risk
                else:  # BUY
                    tp = entry_price + risk
                # Primer trade
                trades_today = 1
                # El trade se ejecuta aquí, calcular pnl y agregarlo
                risk_amount = self.capital * self.risk_per_trade
                # El apalancamiento solo afecta el margen requerido, no el tamaño de la posición para la comisión
                size = risk_amount / abs(entry_price - sl) if abs(entry_price - sl) > 0 else 0
                commission_rate = 0.0004  # 0.04% por trade (solo apertura)
                if direction == 'SELL':
                    sl_hit = (after_entry['high'] >= sl).any()
                    tp_hit = (after_entry['low'] <= tp).any()
                    if tp_hit and sl_hit:
                        tp_idx = after_entry[after_entry['low'] <= tp].index[0]
                        sl_idx = after_entry[after_entry['high'] >= sl].index[0]
                        pnl = (entry_price - tp) * size if tp_idx < sl_idx else (sl - entry_price) * size * -1
                    elif tp_hit:
                        pnl = (entry_price - tp) * size
                    else:
                        pnl = (sl - entry_price) * size * -1
                else:
                    sl_hit = (after_entry['low'] <= sl).any()
                    tp_hit = (after_entry['high'] >= tp).any()
                    if tp_hit and sl_hit:
                        tp_idx = after_entry[after_entry['high'] >= tp].index[0]
                        sl_idx = after_entry[after_entry['low'] <= sl].index[0]
                        pnl = (tp - entry_price) * size if tp_idx < sl_idx else (entry_price - sl) * size * -1
                    elif tp_hit:
                        pnl = (tp - entry_price) * size
                    else:
                        pnl = (entry_price - sl) * size * -1
                commission = abs(size * entry_price * commission_rate)
                pnl -= commission
                
                # Actualizar capital con el resultado del trade
                self.capital += pnl
                
                results.append({
                    'date': day,
                    'session': session_name,
                    'direction': direction,
                    'entry': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'pnl': pnl,
                    'commission': commission,
                    'capital': self.capital
                })
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
                        after_entry2 = after_entry.loc[entry_idx2:]
                        
                        # TP fijo 1:1 para segundo trade también
                        risk2 = abs(entry_price2 - sl)
                        if direction == 'SELL':
                            tp2 = entry_price2 - risk2
                        else:  # BUY
                            tp2 = entry_price2 + risk2
                        
                        risk_amount2 = self.capital * self.risk_per_trade
                        size2 = risk_amount2 / abs(entry_price2 - sl) if abs(entry_price2 - sl) > 0 else 0
                        commission2 = abs(size2 * entry_price2 * 0.0004)
                        if direction == 'SELL':
                            sl_hit2 = (after_entry2['high'] >= sl).any()
                            tp_hit2 = (after_entry2['low'] <= tp2).any()
                            if tp_hit2 and sl_hit2:
                                tp_idx2 = after_entry2[after_entry2['low'] <= tp2].index[0]
                                sl_idx2 = after_entry2[after_entry2['high'] >= sl].index[0]
                                pnl2 = (entry_price2 - tp2) * size2 if tp_idx2 < sl_idx2 else (sl - entry_price2) * size2 * -1
                            elif tp_hit2:
                                pnl2 = (entry_price2 - tp2) * size2
                            else:
                                pnl2 = (sl - entry_price2) * size2 * -1
                        else:
                            sl_hit2 = (after_entry2['low'] <= sl).any()
                            tp_hit2 = (after_entry2['high'] >= tp2).any()
                            if tp_hit2 and sl_hit2:
                                tp_idx2 = after_entry2[after_entry2['high'] >= tp2].index[0]
                                sl_idx2 = after_entry2[after_entry2['low'] <= sl].index[0]
                                pnl2 = (tp2 - entry_price2) * size2 if tp_idx2 < sl_idx2 else (entry_price2 - sl) * size2 * -1
                            elif tp_hit2:
                                pnl2 = (tp2 - entry_price2) * size2
                            else:
                                pnl2 = (entry_price2 - sl) * size2 * -1
                        pnl2 -= commission2
                        
                        # Actualizar capital con el resultado del segundo trade
                        self.capital += pnl2
                        
                        results.append({
                            'date': day,
                            'session': session_name,
                            'direction': direction,
                            'entry': entry_price2,
                            'sl': sl,
                            'tp': tp2,
                            'pnl': pnl2,
                            'commission': commission2,
                            'capital': self.capital
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
            total_pnl_pct = results_df['pnl'].sum() / self.initial_capital * 100
            best_trade_pct = best_trade / self.initial_capital * 100
            worst_trade_pct = worst_trade / self.initial_capital * 100
            print(f"Total trades: {len(results_df)} | Ganancia total: {total_pnl_pct:.2f}%")
            print(f"Winrate: {winrate:.2f}% | Profit Factor: {profit_factor:.2f} | Max Drawdown: {max_drawdown:.2f}%")
            print(f"Mejor Trade: {best_trade_pct:.2f}% | Peor Trade: {worst_trade_pct:.2f}%")
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

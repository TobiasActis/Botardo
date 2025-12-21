"""
🤖 BOTARDO v1.0 - Estrategia Mean Reversion + Trend Filter
Timeframe: 15 minutos
Capital: $500 | Risk: 4% por trade

ESTRATEGIA:
1. Mean Reversion usando Bollinger Bands (20, 2)
2. Trend Filter con EMA 50/200
3. RSI para confirmar extremos
4. Stop Loss: 1.5% del entry
5. Take Profit: RR 1:2 (3% del entry)

REGLAS LONG:
- Precio toca o cruza banda inferior de BB
- RSI < 35 (oversold)
- Tendencia alcista (EMA50 > EMA200) O neutral
- Entry: Cierre de vela

REGLAS SHORT:
- Precio toca o cruza banda superior de BB
- RSI > 65 (overbought)
- Tendencia bajista (EMA50 < EMA200) O neutral
- Entry: Cierre de vela

GESTIÓN:
- SL fijo: 1.5%
- TP fijo: 3% (RR 1:2)
- Break even: Al alcanzar 1:1
- Max 1 trade abierto a la vez
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from typing import Optional, Tuple
from loguru import logger
from binance.client import Client
import os

logger.remove()
logger.add(lambda msg: print(msg, end=''), format="{time:HH:mm:ss} | {level: <8} | {message}", level="INFO")


class Botardo:
    """Botardo v1.0 - Mean Reversion + Trend Filter"""
    
    def __init__(self, capital: float = 500, risk_per_trade: float = 0.02, leverage: int = 3):
        self.initial_capital = capital
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        # Indicadores
        self.bb_period = 20
        self.bb_std = 2
        self.rsi_period = 14
        self.rsi_oversold = 20  # Más extremo para LONG
        self.rsi_overbought = 80  # Más extremo para SHORT
        self.ema_fast = 50
        self.ema_slow = 200
        # Gestión
        self.sl_pct = 0.015  # 1.5%
        self.rr_ratio = 2.0   # 1:2
        # Filtro de volatilidad
        self.atr_period = 14
        self.atr_threshold = 3.0  # Skip si ATR > 3.0x promedio (más estricto)
        self.atr_min = 1.2        # Skip si ATR < 1.2x promedio (más estricto)
        # Estado
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.trailing_atr_mult = 2.0  # Trailing stop = 2.0x ATR (más holgado)
        self.trailing_pct = 0.015  # (legacy, fallback)
        self.loss_streak = 0  # Para gestión de riesgo adaptativa
        self.base_risk_per_trade = risk_per_trade
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores necesarios"""
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (self.bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.bb_std * df['bb_std'])
        
        # EMAs
        df['ema50'] = df['close'].ewm(span=self.ema_fast).mean()
        df['ema200'] = df['close'].ewm(span=self.ema_slow).mean()
        
        # Trend
        df['trend'] = np.where(df['ema50'] > df['ema200'], 1,  # Uptrend
                              np.where(df['ema50'] < df['ema200'], -1,  # Downtrend
                                      0))  # Neutral
        
        # ATR para filtro de volatilidad
        df['tr'] = np.maximum(df['high'] - df['low'],
                             np.maximum(abs(df['high'] - df['close'].shift(1)),
                                       abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        df['atr_avg'] = df['atr'].rolling(window=50).mean()  # Promedio 50 períodos
        df['atr_ratio'] = df['atr'] / df['atr_avg']
        
        return df
    
    def check_long_entry(self, row: pd.Series) -> bool:
        """Verifica condiciones para LONG (versión README_OPTIMO_2025)"""
        # 1. Precio toca banda inferior
        touches_lower = row['close'] <= row['bb_lower'] * 1.002  # 0.2% tolerancia
        # 2. RSI oversold (más extremo)
        rsi_ok = row['rsi'] < self.rsi_oversold
        # 3. Tendencia alcista (EMA50 > EMA200)
        trend_ok = row['ema50'] > row['ema200']
        return touches_lower and rsi_ok and trend_ok
    
    def check_short_entry(self, row: pd.Series) -> bool:
        """Verifica condiciones para SHORT (versión README_OPTIMO_2025)"""
        # 1. Precio toca banda superior
        touches_upper = row['close'] >= row['bb_upper'] * 0.998  # 0.2% tolerancia
        # 2. RSI overbought (más extremo)
        rsi_ok = row['rsi'] > self.rsi_overbought
        # 3. Tendencia bajista (EMA50 < EMA200)
        trend_ok = row['ema50'] < row['ema200']
        return touches_upper and rsi_ok and trend_ok
    
    def open_position(self, row: pd.Series, side: str) -> None:
        """Abre una posición LONG o SHORT con gestión de riesgo adaptativa"""
        entry_price = row['close']
        # Calcular SL y TP
        if side == 'LONG':
            sl_price = entry_price * (1 - self.sl_pct)
            tp_price = entry_price * (1 + self.sl_pct * self.rr_ratio)
        else:  # SHORT
            sl_price = entry_price * (1 + self.sl_pct)
            tp_price = entry_price * (1 - self.sl_pct * self.rr_ratio)
        # Gestión de riesgo adaptativa: reducir riesgo tras 2+ pérdidas seguidas
        risk_per_trade = self.base_risk_per_trade
        if self.loss_streak >= 2:
            risk_per_trade = max(self.base_risk_per_trade * 0.5, 0.01)
        # Calcular size basado en riesgo
        risk_amount = self.capital * risk_per_trade
        position_size = (risk_amount * self.leverage) / (abs(entry_price - sl_price))
        self.position = {
            'side': side,
            'entry_price': entry_price,
            'entry_time': row['timestamp'],
            'sl_price': sl_price,
            'tp_price': tp_price,
            'size': position_size,
            'leverage': self.leverage,
            'break_even_activated': False,
            'partial_exit_done': False,
            'size_remaining': position_size
        }
        logger.success(f"✅ {side} abierto @ ${entry_price:,.2f}")
        logger.info(f"   Size: {position_size:.4f} | Leverage: {self.leverage}x")
        logger.info(f"   SL: ${sl_price:,.2f} | TP: ${tp_price:,.2f}")
        logger.info(f"   RSI: {row['rsi']:.1f} | Trend: {row['trend']}")
    
    def update_trailing_stop(self, row: pd.Series) -> None:
        """Trailing stop dinámico basado en ATR (más robusto a volatilidad)"""
        if not self.position:
            return
        entry = self.position['entry_price']
        atr = row['atr'] if 'atr' in row and not pd.isna(row['atr']) else None
        if atr is None or atr == 0:
            atr = entry * self.trailing_pct  # fallback
        trail_dist = atr * self.trailing_atr_mult
        if self.position['side'] == 'LONG':
            if 'max_price' not in self.position:
                self.position['max_price'] = entry
            self.position['max_price'] = max(self.position['max_price'], row['high'])
            new_trail = self.position['max_price'] - trail_dist
            if new_trail > self.position['sl_price']:
                self.position['sl_price'] = new_trail
                logger.info(f"🔄 Trailing Stop LONG (ATR) actualizado @ ${self.position['sl_price']:,.2f}")
        else:
            if 'min_price' not in self.position:
                self.position['min_price'] = entry
            self.position['min_price'] = min(self.position['min_price'], row['low'])
            new_trail = self.position['min_price'] + trail_dist
            if new_trail < self.position['sl_price']:
                self.position['sl_price'] = new_trail
                logger.info(f"🔄 Trailing Stop SHORT (ATR) actualizado @ ${self.position['sl_price']:,.2f}")
    
    def check_exit(self, row: pd.Series) -> Optional[Tuple[str, float]]:
        """Verifica si debe cerrar la posición (TP o trailing stop)"""
        current_price = row['close']
        high = row['high']
        low = row['low']
        pos = self.position
        # Salida parcial al TP
        if pos['side'] == 'LONG':
            if not pos['partial_exit_done'] and high >= pos['tp_price']:
                return ('partial_tp', pos['tp_price'], pos['size'] * 0.5)
            if pos['partial_exit_done']:
                if low <= pos['sl_price']:
                    return ('trailing_stop', pos['sl_price'], pos['size_remaining'])
                if row['rsi'] > 45:
                    return ('rsi_exit', current_price, pos['size_remaining'])
        else:
            if not pos['partial_exit_done'] and low <= pos['tp_price']:
                return ('partial_tp', pos['tp_price'], pos['size'] * 0.5)
            if pos['partial_exit_done']:
                if high >= pos['sl_price']:
                    return ('trailing_stop', pos['sl_price'], pos['size_remaining'])
                if row['rsi'] < 55:
                    return ('rsi_exit', current_price, pos['size_remaining'])
        return None
    
    def close_position(self, row: pd.Series, exit_type: str, exit_price: float, close_size: float = None) -> None:
        """Cierra la posición y registra el trade, actualiza streak de pérdidas"""
        entry_price = self.position['entry_price']
        # Permitir size parcial
        size = self.position['size'] if exit_type != 'partial_tp' else self.position['size'] * 0.5
        if exit_type in ['trailing_stop', 'rsi_exit'] and self.position['partial_exit_done']:
            size = self.position['size_remaining']
        # Calcular P&L
        if self.position['side'] == 'LONG':
            price_diff = exit_price - entry_price
        else:  # SHORT
            price_diff = entry_price - exit_price
        pnl_pct = (price_diff / entry_price) * self.position['leverage']
        pnl_usd = self.capital * pnl_pct * (size / self.position['size'])
        # Actualizar capital
        self.capital += pnl_usd
        # Actualizar streak de pérdidas
        if pnl_usd < 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0
        # Registrar trade
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': row['timestamp'],
            'side': self.position['side'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'leverage': self.position['leverage'],
            'pnl_usd': pnl_usd,
            'pnl_pct': pnl_pct * 100,
            'exit_type': exit_type,
            'capital_after': self.capital
        }
        self.trades.append(trade)
        # Log
        emoji = "✅" if pnl_usd > 0 else "❌"
        logger.info(f"{emoji} Cerrado ${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%) - {exit_type}")
        # Si es salida parcial, marcar y reducir size restante
        if exit_type == 'partial_tp':
            self.position['partial_exit_done'] = True
            self.position['size_remaining'] = self.position['size'] * 0.5
            self.position['entry_price'] = exit_price
        else:
            self.position = None
    
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ejecuta el backtest completo"""
        
        logger.info("📊 Calculando indicadores...")
        df = self.calculate_indicators(df)
        logger.info("✅ Indicadores listos")
        
        logger.info(f"\n🚀 Ejecutando backtest: {len(df)} velas")
        logger.info(f"   Desde: {df['timestamp'].iloc[0]} | Hasta: {df['timestamp'].iloc[-1]}")
        
        # Iterar por cada vela
        for idx in range(self.ema_slow, len(df)):
            row = df.iloc[idx]
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'capital': self.capital
            })
            if self.position:
                self.update_trailing_stop(row)
                exit_info = self.check_exit(row)
                if exit_info:
                    exit_type, exit_price, close_size = exit_info
                    self.close_position(row, exit_type, exit_price, close_size)
            else:
                if self.check_long_entry(row):
                    self.open_position(row, 'LONG')
                elif self.check_short_entry(row):
                    self.open_position(row, 'SHORT')
            progress = (idx / len(df)) * 100
            if progress % 10 < 0.1:
                logger.info(f"📊 Progress: {progress:.1f}% | Balance: ${self.capital:.2f}")
        return pd.DataFrame(self.trades)
        return pd.DataFrame(self.trades)
    
    def print_results(self) -> None:
        """Imprime resultados del backtest"""
        
        if not self.trades:
            logger.error("❌ No hay trades para analizar")
            return
        
        df_trades = pd.DataFrame(self.trades)
        
        # Métricas básicas
        total_trades = len(df_trades)
        winners = df_trades[df_trades['pnl_usd'] > 0]
        losers = df_trades[df_trades['pnl_usd'] < 0]
        
        win_rate = len(winners) / total_trades * 100
        
        total_pnl = df_trades['pnl_usd'].sum()
        pnl_pct = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        avg_win = winners['pnl_usd'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl_usd'].mean() if len(losers) > 0 else 0
        
        best_trade = df_trades['pnl_usd'].max()
        worst_trade = df_trades['pnl_usd'].min()
        
        # Profit Factor
        total_wins = winners['pnl_usd'].sum() if len(winners) > 0 else 0
        total_losses = abs(losers['pnl_usd'].sum()) if len(losers) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Max Drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['capital'].cummax()
        equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Imprimir
        logger.info("\n" + "="*70)
        logger.info("📊 BOTARDO v1.0 - RESULTADOS")
        logger.info("="*70)
        logger.info(f"Capital Inicial:    $  {self.initial_capital:>8.2f}")
        logger.info(f"Balance Final:      $  {self.capital:>8.2f}")
        logger.info(f"P&L Total:          $  {total_pnl:>8.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"Max Drawdown:                {max_drawdown:>6.2f}%")
        logger.info("-"*70)
        logger.info(f"Trades Totales:                    {total_trades:>4}")
        logger.info(f"Ganadores:                         {len(winners):>4}")
        logger.info(f"Perdedores:                        {len(losers):>4}")
        logger.info(f"Win Rate:                      {win_rate:>6.2f}%")
        logger.info("-"*70)
        logger.info(f"Ganancia Promedio:  $  {avg_win:>8.2f}")
        logger.info(f"Pérdida Promedio:   $  {avg_loss:>8.2f}")
        logger.info(f"Mejor Trade:        $  {best_trade:>8.2f}")
        logger.info(f"Peor Trade:         $  {worst_trade:>8.2f}")
        logger.info(f"Profit Factor:               {profit_factor:>8.2f}")
        logger.info("="*70)


def get_binance_usdt_balance(api_key, api_secret):
    client = Client(api_key, api_secret)
    account = client.get_account()
    for balance in account['balances']:
        if balance['asset'] == 'USDT':
            return float(balance['free'])
    return 0.0


def main():
    parser = argparse.ArgumentParser(description='Botardo v1.0 - Mean Reversion Strategy')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--capital', type=float, default=None, help='Initial capital (overridden if --use-binance-balance)')
    parser.add_argument('--risk', type=float, default=0.04, help='Risk per trade (0.04 = 4%)')
    parser.add_argument('--leverage', type=int, default=5, help='Leverage multiplier')
    parser.add_argument('--start', default='2024-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default='2024-12-31', help='End date YYYY-MM-DD')
    parser.add_argument('--assets', type=str, default=None, help='Comma-separated list of assets (ej: BTCUSDT,SOLUSDT,ADAUSDT)')
    parser.add_argument('--use-binance-balance', action='store_true', help='Use real Binance USDT balance as capital')
    parser.add_argument('--binance-api-key', type=str, default=None, help='Binance API key')
    parser.add_argument('--binance-api-secret', type=str, default=None, help='Binance API secret')
    args = parser.parse_args()

    # Obtener capital real si se solicita
    if args.use_binance_balance:
        api_key = args.binance_api_key or os.getenv('BINANCE_API_KEY')
        api_secret = args.binance_api_secret or os.getenv('BINANCE_API_SECRET')
        if not api_key or not api_secret:
            raise ValueError('Debe proveer --binance-api-key y --binance-api-secret o variables de entorno BINANCE_API_KEY/BINANCE_API_SECRET')
        capital_total = get_binance_usdt_balance(api_key, api_secret)
        logger.info(f"Capital real en Binance: ${capital_total:.2f}")
    else:
        if args.capital is None:
            raise ValueError('Debe especificar --capital o --use-binance-balance')
        capital_total = args.capital

    # Determinar activos y capital por activo
    if args.assets:
        asset_list = [a.strip() for a in args.assets.split(',') if a.strip()]
        n_assets = len(asset_list)
        if n_assets == 0:
            raise ValueError("Debe especificar al menos un activo en --assets")
        capital_per_asset = capital_total / n_assets
        logger.info(f"Multi-asset: {asset_list} | Capital por activo: ${capital_per_asset:.2f}")
    else:
        asset_list = [None]
        capital_per_asset = capital_total

    # Cargar datos y ejecutar para cada activo
    for i, asset in enumerate(asset_list):
        logger.info(f"\n📂 Cargando datos... {asset if asset else ''}")
        data_path = args.data.replace('SYMBOL', asset) if asset else args.data
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        df = df[(df['timestamp'] >= args.start) & (df['timestamp'] <= args.end)]
        logger.info(f"✅ Cargados {len(df)} registros")
        logger.info(f"   Timeframe: 15 minutos")
        logger.info(f"   Período: {args.start} - {args.end}")
        bot = Botardo(capital=capital_per_asset, risk_per_trade=args.risk, leverage=args.leverage)
        trades_df = bot.run_backtest(df)
        bot.print_results()
        if not trades_df.empty:
            out_name = f'botardo_trades_{asset}.csv' if asset else 'botardo_trades.csv'
            trades_df.to_csv(out_name, index=False)
            logger.info(f"\n✅ Trades guardados: {out_name}")


if __name__ == '__main__':
    main()

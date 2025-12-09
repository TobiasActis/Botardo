"""
🤖 SUPERBOT BACKTEST 🤖
Backtest del sistema unificado: PO3 Primary + SMC Confirm + Wyckoff Filter
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from loguru import logger

from unified_strategy import UnifiedStrategy
from backtest_wyckoff import BacktestEngine


class SuperbotBacktest(BacktestEngine):
    """Backtest específico para la estrategia unificada"""
    
    def __init__(
        self,
        initial_balance: float = 20.0,
        risk_per_trade: float = 0.02,
        max_leverage: int = 5,
        commission: float = 0.0004,
        use_po3: bool = True,
        use_smc: bool = True,
        use_wyckoff: bool = False,
        po3_min_rr: float = 3.0,
        smc_standalone: int = 7,
        max_loss_per_trade: float = None
    ):
        super().__init__(initial_balance, risk_per_trade, max_leverage, commission)
        self.max_loss_per_trade = max_loss_per_trade or (initial_balance * 0.30)  # 30% default
        
        self.strategy = UnifiedStrategy(
            use_po3=use_po3,
            use_smc=use_smc,
            use_wyckoff=use_wyckoff,
            po3_min_rr=po3_min_rr,
            smc_standalone_threshold=smc_standalone
        )
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, max_loss_override: float = None) -> tuple[float, int]:
        """Override con leverage dinámico según calidad del setup"""
        # Obtener info de la última señal procesada
        last_signal = getattr(self, '_last_signal_info', {})
        
        # Determinar leverage dinámico según contexto
        confluence_score = last_signal.get('confluence_score', 0)
        rr_ratio = last_signal.get('rr_ratio', 0)
        signal_type = last_signal.get('signal_source', '')
        
        # Lógica de leverage dinámico
        if confluence_score >= 13:  # PO3 (10) + SMC (3+)
            dynamic_leverage = 75  # Máxima confianza
        elif signal_type == 'po3_primary' and rr_ratio >= 5.0:
            dynamic_leverage = 60  # PO3 con RR excelente
        elif signal_type == 'po3_primary' and rr_ratio >= 4.0:
            dynamic_leverage = 50  # PO3 con RR muy bueno
        elif signal_type == 'po3_primary':
            dynamic_leverage = 30  # PO3 standard
        elif signal_type == 'smc_standalone':
            dynamic_leverage = 10  # SMC solo
        else:
            dynamic_leverage = 5  # Default conservador
        
        # Limitar al máximo configurado
        dynamic_leverage = min(dynamic_leverage, self.max_leverage)
        
        # PROTECCIÓN: Si balance negativo, NO operar
        if self.balance <= 0:
            return 0, 1
        
        # Calcular posición con leverage dinámico
        # IMPORTANTE: risk_amount basado en balance INICIAL, no actual
        initial_capital = self.initial_balance
        max_loss = max_loss_override if max_loss_override else (initial_capital * self.risk_per_trade)
        risk_amount = min(initial_capital * self.risk_per_trade, max_loss)
        stop_distance = abs(entry_price - stop_loss) / entry_price
        
        # Evitar stop_distance muy pequeño (protección)
        if stop_distance < 0.001:  # Menos de 0.1%
            return 0, 1
        
        # Usar leverage dinámico en vez de calculado
        leverage = dynamic_leverage
        
        # Valor máximo de posición basado en balance ACTUAL
        max_position_value = max(self.balance, 0) * leverage
        risk_based_position = risk_amount / stop_distance
        position_value = min(risk_based_position, max_position_value)
        
        quantity = position_value / entry_price
        
        return quantity, leverage
    
    def run_backtest(
        self,
        data_1h: pd.DataFrame,
        data_4h: pd.DataFrame,
        data_1d: pd.DataFrame,
        data_15m: pd.DataFrame = None,
        start_date: str = None,
        end_date: str = None
    ):
        """Ejecuta el backtest con la estrategia unificada"""
        
        logger.info("=" * 60)
        logger.info("🤖 SUPERBOT BACKTEST 🤖")
        logger.info("=" * 60)
        logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
        logger.info(f"Risk per Trade: {self.risk_per_trade * 100}%")
        logger.info(f"Max Leverage: {self.max_leverage}x")
        logger.info(f"PO3 Min RR: {self.strategy.po3_min_rr}:1")
        logger.info(f"SMC Standalone: {self.strategy.smc_standalone_threshold}+ pts")
        logger.info("=" * 60)
        
        # Filtrar por fechas
        if start_date:
            start = pd.Timestamp(start_date)
            data_1h = data_1h[data_1h.index >= start]
            data_4h = data_4h[data_4h.index >= start]
            data_1d = data_1d[data_1d.index >= start]
            if data_15m is not None:
                data_15m = data_15m[data_15m.index >= start]
        
        if end_date:
            end = pd.Timestamp(end_date)
            data_1h = data_1h[data_1h.index <= end]
            data_4h = data_4h[data_4h.index <= end]
            data_1d = data_1d[data_1d.index <= end]
            if data_15m is not None:
                data_15m = data_15m[data_15m.index <= end]
        
        logger.info(f"Backtest period: {data_1h.index[0]} to {data_1h.index[-1]}")
        logger.info(f"Total 1h bars: {len(data_1h)}")
        
        # Determinar timeframe de ejecución
        if data_15m is not None and len(data_15m) > 0:
            execution_tf = '15m'
            execution_data = data_15m
        else:
            execution_tf = '1h'
            execution_data = data_1h
        
        logger.info(f"Execution timeframe: {execution_tf}")
        logger.info("Starting simulation...")
        logger.info("-" * 60)
        
        # Iterar por las velas de ejecución
        total_bars = len(execution_data)
        
        for i, (timestamp, row) in enumerate(execution_data.iterrows()):
            
            # Progress cada 1000 barras
            if i % 1000 == 0:
                progress = (i / total_bars) * 100
                logger.info(f"Progress: {progress:.1f}% ({i}/{total_bars}) - Balance: ${self.balance:.2f}")
            
            # Verificar posición existente
            if self.current_position is not None:
                self.check_position(row)
            
            # Si ya hay posición, skip
            if self.current_position is not None:
                continue
            
            # Preparar data dict para la estrategia
            data_dict = {
                '1h': data_1h[data_1h.index <= timestamp],
                '4h': data_4h[data_4h.index <= timestamp],
                '1d': data_1d[data_1d.index <= timestamp]
            }
            
            if data_15m is not None:
                data_dict['15m'] = data_15m[data_15m.index <= timestamp]
            
            # Pedir señal a la estrategia
            signal = self.strategy.get_trading_signal(
                data_dict=data_dict,
                current_time=timestamp,
                execution_tf=execution_tf
            )
            
            if signal is None:
                continue
            
            # Filtrar trades con profit potencial muy bajo
            entry = signal['entry']
            tp = signal['take_profit']
            profit_pct = abs((tp - entry) / entry) * 100
            
            # Requerir mínimo 1% de movimiento (con leverage dinámico 10x-75x)
            if profit_pct < 1.0:
                continue
            
            # Tenemos señal!
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"SIGNAL #{len(self.trades) + 1} at {timestamp}")
            logger.info("=" * 60)
            
            # Guardar info de señal para leverage dinámico
            self._last_signal_info = {
                'confluence_score': signal.get('confluence_score', 0),
                'rr_ratio': signal.get('rr_ratio', 0),
                'signal_source': 'po3_primary' if signal.get('confluence_score', 0) >= 10 else 'smc_standalone'
            }
            
            # Abrir posición
            self.open_position(
                timestamp=timestamp,
                side='long' if signal['type'] == 'LONG' else 'short',
                entry_price=signal['entry'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                signal_info=signal
            )
        
        # Cerrar posición final si existe
        if self.current_position is not None:
            last_row = execution_data.iloc[-1]
            self.close_position(
                timestamp=execution_data.index[-1],
                close_price=last_row['close'],
                reason='backtest_end'
            )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🏁 BACKTEST COMPLETED")
        logger.info("=" * 60)
        
        return {
            'final_balance': self.balance,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Superbot Backtest')
    parser.add_argument('--data_1m', required=True, help='Path to 1-minute data CSV')
    parser.add_argument('--start', default='2025-11-01', help='Start date')
    parser.add_argument('--end', default='2025-11-30', help='End date')
    parser.add_argument('--initial_capital', type=float, default=20, help='Initial capital in USD')
    parser.add_argument('--risk_per_trade', type=float, default=0.02, help='Risk per trade (0.02 = 2%)')
    parser.add_argument('--leverage', type=int, default=10, help='Max leverage (default: 10x)')
    parser.add_argument('--po3_min_rr', type=float, default=3.0, help='PO3 minimum RR ratio (default: 3.0)')
    parser.add_argument('--smc_standalone', type=int, default=8, help='SMC standalone threshold (default: 8)')
    parser.add_argument('--max_loss', type=float, default=None, help='Max loss per trade in USD (None = unlimited)')
    parser.add_argument('--no_po3', action='store_true', help='Disable PO3')
    parser.add_argument('--no_smc', action='store_true', help='Disable SMC')
    parser.add_argument('--wyckoff', action='store_true', help='Enable Wyckoff filter')
    parser.add_argument('--out', default='superbot_backtest', help='Output file prefix')
    
    args = parser.parse_args()
    
    # Cargar datos
    logger.info(f"Loading data from {args.data_1m}...")
    df_1m = pd.read_csv(args.data_1m)
    df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
    df_1m.set_index('timestamp', inplace=True)
    logger.info(f"Loaded {len(df_1m)} 1-minute bars")
    
    # Resamplear
    logger.info("Resampling to higher timeframes...")
    df_15m = df_1m.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_1h = df_1m.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_4h = df_1m.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_1d = df_1m.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    logger.info(f"Resampled: {len(df_15m)} 15m, {len(df_1h)} 1h, {len(df_4h)} 4h, {len(df_1d)} 1d bars")
    
    # Crear backtest
    backtest = SuperbotBacktest(
        initial_balance=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        max_leverage=args.leverage,
        use_po3=not args.no_po3,
        use_smc=not args.no_smc,
        use_wyckoff=args.wyckoff,
        po3_min_rr=args.po3_min_rr,
        smc_standalone=args.smc_standalone,
        max_loss_per_trade=args.max_loss
    )
    
    # Ejecutar backtest con manejo de interrupción
    try:
        results = backtest.run_backtest(
            data_1h=df_1h,
            data_4h=df_4h,
            data_1d=df_1d,
            data_15m=df_15m,
            start_date=args.start,
            end_date=args.end
        )
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Backtest interrupted by user")
        logger.info("Generating statistics for completed trades...")
    
    # Mostrar resumen (siempre, incluso si fue interrumpido)
    backtest.print_summary()
    
    # Guardar resultados
    if backtest.trades:
        trades_df = pd.DataFrame(backtest.trades)
        trades_df.to_csv(f"{args.out}_trades.csv", index=False)
        logger.info(f"✅ Trade log saved to {args.out}_trades.csv")
        
        # Graficar si hay trades
        try:
            backtest.plot_results(f"{args.out}.png")
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")

"""
Backtest de Estrategia Híbrida (Wyckoff + SMC)
"""

import pandas as pd
import numpy as np
import argparse
from loguru import logger
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from hybrid_strategy import HybridStrategy
from backtest_wyckoff import BacktestEngine


class HybridBacktest(BacktestEngine):
    """
    Motor de backtest para estrategia híbrida
    Extiende BacktestEngine pero usa HybridStrategy
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        max_leverage: int = 5,
        commission: float = 0.0004,
        
        # Parámetros de estrategia
        min_confluence_score: int = 5,
        require_wyckoff_alignment: bool = False  # Más flexible por defecto
    ):
        super().__init__(initial_balance, risk_per_trade, max_leverage, commission)
        
        self.strategy = HybridStrategy(
            wyckoff_timeframes=['1h', '4h', '1d'],
            min_confluence_score=min_confluence_score,
            require_wyckoff_alignment=require_wyckoff_alignment
        )
        
        self.min_confluence_score = min_confluence_score
        
    def run_backtest(
        self,
        data_1h: pd.DataFrame,
        data_4h: pd.DataFrame,
        data_1d: pd.DataFrame,
        data_15m: pd.DataFrame = None,  # Timeframe de ejecución
        start_date: str = None,
        end_date: str = None
    ):
        """
        Ejecuta backtest con estrategia híbrida
        """
        logger.info("="*60)
        logger.info("HYBRID BACKTEST (Wyckoff + SMC)")
        logger.info("="*60)
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Risk per Trade: {self.risk_per_trade * 100}%")
        logger.info(f"Max Leverage: {self.max_leverage}x")
        logger.info(f"Min Confluence Score: {self.min_confluence_score}")
        logger.info("="*60)
        
        # Filtrar por fechas
        if start_date:
            data_1h = data_1h[data_1h.index >= start_date]
            data_4h = data_4h[data_4h.index >= start_date]
            data_1d = data_1d[data_1d.index >= start_date]
            if data_15m is not None:
                data_15m = data_15m[data_15m.index >= start_date]
        
        if end_date:
            data_1h = data_1h[data_1h.index <= end_date]
            data_4h = data_4h[data_4h.index <= end_date]
            data_1d = data_1d[data_1d.index <= end_date]
            if data_15m is not None:
                data_15m = data_15m[data_15m.index <= end_date]
        
        logger.info(f"Backtest period: {data_1h.index[0]} to {data_1h.index[-1]}")
        logger.info(f"Total 1h bars: {len(data_1h)}")
        
        # Usar 15m si está disponible, sino 1h
        execution_tf = '15m' if data_15m is not None else '1h'
        execution_data = data_15m if data_15m is not None else data_1h
        
        logger.info(f"Execution timeframe: {execution_tf}")
        logger.info("Starting simulation...")
        logger.info("-"*60)
        
        # Iterar sobre cada barra
        total_bars = len(execution_data)
        signal_count = 0
        
        for i, (timestamp, row) in enumerate(execution_data.iterrows()):
            
            # Progress cada 1000 barras
            if i % 1000 == 0:
                progress = (i / total_bars) * 100
                logger.info(f"Progress: {progress:.1f}% ({i}/{total_bars}) - Balance: ${self.balance:,.2f}")
            
            # Preparar datos multi-timeframe
            data_dict = {
                '1h': data_1h[:timestamp],
                '4h': data_4h[:timestamp],
                '1d': data_1d[:timestamp],
                execution_tf: execution_data[:timestamp]
            }
            
            # Asegurar que hay suficientes datos
            if len(data_dict['1h']) < 200:
                continue
            
            # 1. Verificar posición abierta
            if self.current_position is not None:
                position_closed = self.check_position(row)
                if position_closed:
                    logger.info(f"Position closed at {timestamp}")
            
            # 2. Buscar nuevas señales (solo si no hay posición abierta)
            if self.current_position is None:
                signal = self.strategy.get_trading_signal(
                    data_dict=data_dict,
                    current_time=timestamp,
                    execution_tf=execution_tf
                )
                
                if signal:
                    signal_count += 1
                    logger.info(f"\n{'='*60}")
                    logger.info(f"SIGNAL #{signal_count} at {timestamp}")
                    logger.info(f"{'='*60}")
                    
                    # Abrir posición
                    self.open_position(
                        timestamp=timestamp,
                        side=signal['signal'],
                        entry_price=signal['entry_price'],
                        stop_loss=signal['stop_loss'],
                        take_profit=signal['take_profit'],
                        signal_info=signal
                    )
        
        # Cerrar posición final si quedó abierta
        if self.current_position is not None:
            logger.warning("Closing final open position...")
            self.close_position(
                timestamp=execution_data.index[-1],
                exit_price=execution_data.iloc[-1]['close'],
                reason="End of backtest"
            )
        
        logger.info("\n" + "="*60)
        logger.info("BACKTEST COMPLETED")
        logger.info("="*60)
        logger.info(f"Total signals generated: {signal_count}")
        logger.info(f"Total trades executed: {self.total_trades}")
        
        return {
            'signals_generated': signal_count,
            'trades_executed': self.total_trades
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Backtest (Wyckoff + SMC)")
    parser.add_argument("--data_1m", type=str, required=True, help="Path to 1m OHLCV CSV")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-11-30", help="End date")
    parser.add_argument("--initial_capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--leverage", type=int, default=5, help="Max leverage")
    parser.add_argument("--risk_per_trade", type=float, default=0.02, help="Risk per trade")
    parser.add_argument("--min_confluence", type=int, default=5, help="Min confluence score")
    parser.add_argument("--wyckoff_align", action='store_true', help="Require Wyckoff alignment")
    parser.add_argument("--out", type=str, default="hybrid_backtest_results", help="Output prefix")
    
    args = parser.parse_args()
    
    # Cargar datos
    logger.info(f"Loading data from {args.data_1m}...")
    df_1m = pd.read_csv(args.data_1m, parse_dates=["timestamp"], index_col="timestamp")
    df_1m = df_1m.loc[args.start:args.end]
    logger.info(f"Loaded {len(df_1m)} 1-minute bars")
    
    # Resample a timeframes necesarios
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
    
    # Crear motor de backtest
    backtest = HybridBacktest(
        initial_balance=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        max_leverage=args.leverage,
        min_confluence_score=args.min_confluence,
        require_wyckoff_alignment=args.wyckoff_align
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
    else:
        logger.warning("⚠️  No trades executed, no files generated")

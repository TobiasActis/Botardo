"""
Backtest Engine for Wyckoff Multi-Timeframe Strategy
Motor de backtesting completo con análisis de performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from multi_tf_wyckoff_rules import MultiTimeframeWyckoff, WyckoffPhase

# Configurar estilo de gráficos
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 10)


class BacktestEngine:
    """
    Motor de backtesting para estrategia Wyckoff
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02,
        max_leverage: int = 5,
        commission: float = 0.0004  # 0.04% por lado
    ):
        """
        Args:
            initial_balance: Balance inicial en USDT
            risk_per_trade: % de balance arriesgado por trade
            max_leverage: Apalancamiento máximo
            commission: Comisión por trade (maker/taker)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self.commission = commission
        
        # Tracking de trades
        self.trades: List[Dict] = []
        self.open_position: Optional[Dict] = None
        self.equity_curve: List[float] = [initial_balance]
        
        # Métricas
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float
    ) -> tuple[float, int]:
        """Calcula tamaño de posición y leverage"""
        risk_amount = self.balance * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss) / entry_price
        
        # Leverage óptimo
        leverage = min(int(1 / stop_distance), self.max_leverage)
        leverage = max(1, leverage)
        
        # Cantidad en USDT
        position_value = risk_amount / stop_distance
        quantity = position_value / entry_price
        
        return quantity, leverage
    
    def open_position(
        self,
        timestamp: pd.Timestamp,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_info: Dict
    ):
        """Abre una nueva posición"""
        if self.open_position is not None:
            logger.warning("Position already open, skipping new signal")
            return
        
        quantity, leverage = self.calculate_position_size(entry_price, stop_loss)
        position_value = quantity * entry_price
        
        # Calcular comisión de entrada
        entry_commission = position_value * self.commission
        
        self.open_position = {
            'entry_time': timestamp,
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_value': position_value,
            'entry_commission': entry_commission,
            'signal_info': signal_info
        }
        
        logger.info(
            f"Position opened: {side} {quantity:.4f} @ ${entry_price:.2f} "
            f"(Leverage: {leverage}x, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})"
        )
    
    def check_position(self, current_bar: pd.Series) -> bool:
        """
        Verifica si la posición debe cerrarse
        Returns True si se cerró la posición
        """
        if self.open_position is None:
            return False
        
        pos = self.open_position
        current_price = current_bar['close']
        high = current_bar['high']
        low = current_bar['low']
        
        # Variables para tracking
        exit_price = None
        exit_reason = None
        
        if pos['side'] == 'LONG':
            # Check stop loss
            if low <= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_reason = 'stop_loss'
            # Check take profit
            elif high >= pos['take_profit']:
                exit_price = pos['take_profit']
                exit_reason = 'take_profit'
        
        else:  # SHORT
            # Check stop loss
            if high >= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_reason = 'stop_loss'
            # Check take profit
            elif low <= pos['take_profit']:
                exit_price = pos['take_profit']
                exit_reason = 'take_profit'
        
        if exit_price:
            self.close_position(
                timestamp=current_bar.name,
                exit_price=exit_price,
                exit_reason=exit_reason
            )
            return True
        
        return False
    
    def close_position(
        self,
        timestamp: pd.Timestamp,
        exit_price: float,
        exit_reason: str
    ):
        """Cierra la posición actual"""
        if self.open_position is None:
            return
        
        pos = self.open_position
        
        # Calcular P&L
        if pos['side'] == 'LONG':
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        else:  # SHORT
            pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
        
        # Aplicar leverage
        pnl_pct = pnl_pct * pos['leverage']
        
        # Calcular P&L en USDT
        position_value = pos['position_value']
        pnl_usdt = position_value * pnl_pct
        
        # Restar comisiones
        exit_commission = position_value * self.commission
        total_commission = pos['entry_commission'] + exit_commission
        pnl_usdt -= total_commission
        
        # Actualizar balance
        self.balance += pnl_usdt
        self.equity_curve.append(self.balance)
        
        # Registrar trade
        trade_duration = timestamp - pos['entry_time']
        
        trade_record = {
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'duration': trade_duration,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'quantity': pos['quantity'],
            'leverage': pos['leverage'],
            'pnl_pct': pnl_pct * 100,
            'pnl_usdt': pnl_usdt,
            'exit_reason': exit_reason,
            'balance_after': self.balance,
            'signal_info': pos['signal_info']
        }
        
        self.trades.append(trade_record)
        self.total_trades += 1
        
        if pnl_usdt > 0:
            self.winning_trades += 1
            logger.info(f"✅ Trade closed: +${pnl_usdt:.2f} ({pnl_pct*100:.2f}%) - {exit_reason}")
        else:
            self.losing_trades += 1
            logger.info(f"❌ Trade closed: ${pnl_usdt:.2f} ({pnl_pct*100:.2f}%) - {exit_reason}")
        
        # Limpiar posición
        self.open_position = None
    
    def run_backtest(
        self,
        data_1h: pd.DataFrame,
        data_4h: pd.DataFrame,
        data_1d: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Ejecuta el backtest completo
        
        Args:
            data_1h, data_4h, data_1d: DataFrames con datos OHLCV
            start_date: Fecha de inicio (formato 'YYYY-MM-DD')
            end_date: Fecha de fin (formato 'YYYY-MM-DD')
        """
        logger.info("Starting backtest...")
        
        # Filtrar por fechas si se especificaron
        if start_date:
            data_1h = data_1h[data_1h.index >= start_date]
        if end_date:
            data_1h = data_1h[data_1h.index <= end_date]
        
        # Crear instancia de Wyckoff
        wyckoff = MultiTimeframeWyckoff(timeframes=['1h', '4h', '1d'])
        wyckoff.load_data('1h', data_1h)
        wyckoff.load_data('4h', data_4h)
        wyckoff.load_data('1d', data_1d)
        
        # Iterar sobre cada barra de 1h
        for i in range(100, len(data_1h)):  # Empezar después de periodo de warmup
            current_bar = data_1h.iloc[i]
            current_time = current_bar.name
            
            # Actualizar datos de Wyckoff con ventana móvil
            wyckoff.load_data('1h', data_1h.iloc[:i+1].tail(200))
            wyckoff.load_data('4h', data_4h[data_4h.index <= current_time].tail(100))
            wyckoff.load_data('1d', data_1d[data_1d.index <= current_time].tail(50))
            
            # Verificar posición abierta
            if self.open_position:
                self.check_position(current_bar)
            
            # Si no hay posición, buscar señal
            if self.open_position is None:
                # Analizar cada 4 horas para no saturar
                if i % 4 == 0:
                    wyckoff.analyze_all_timeframes()
                    signal = wyckoff.get_trading_signal()
                    
                    if signal:
                        self.open_position(
                            timestamp=current_time,
                            side=signal['type'],
                            entry_price=signal['entry'],
                            stop_loss=signal['stop_loss'],
                            take_profit=signal['take_profit'],
                            signal_info=signal
                        )
        
        # Cerrar posición abierta al final
        if self.open_position:
            last_bar = data_1h.iloc[-1]
            self.close_position(
                timestamp=last_bar.name,
                exit_price=last_bar['close'],
                exit_reason='backtest_end'
            )
        
        logger.info("Backtest completed!")
        self.print_summary()
    
    def print_summary(self):
        """Imprime resumen de resultados"""
        if self.total_trades == 0:
            logger.warning("No trades executed")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        win_rate = (self.winning_trades / self.total_trades) * 100
        total_pnl = self.balance - self.initial_balance
        total_return = (total_pnl / self.initial_balance) * 100
        
        avg_win = trades_df[trades_df['pnl_usdt'] > 0]['pnl_usdt'].mean() if self.winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_usdt'] < 0]['pnl_usdt'].mean() if self.losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if avg_loss != 0 else 0
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print(f"Total P&L: ${total_pnl:,.2f} ({total_return:.2f}%)")
        print("-"*60)
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print("-"*60)
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print("-"*60)
        print(f"Max Drawdown: {self.calculate_max_drawdown():.2f}%")
        print(f"Sharpe Ratio: {self.calculate_sharpe_ratio():.2f}")
        print("="*60 + "\n")
    
    def calculate_max_drawdown(self) -> float:
        """Calcula el máximo drawdown"""
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        return abs(drawdown.min())
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calcula el Sharpe Ratio"""
        if len(self.trades) < 2:
            return 0.0
        
        trades_df = pd.DataFrame(self.trades)
        returns = trades_df['pnl_pct'].values / 100
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if len(excess_returns) == 0 or excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    
    def plot_results(self, save_path: str = 'backtest_results.png'):
        """Genera gráficos de resultados"""
        if self.total_trades == 0:
            logger.warning("No trades to plot")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Backtest Results - Wyckoff Strategy', fontsize=16, fontweight='bold')
        
        trades_df = pd.DataFrame(self.trades)
        
        # 1. Equity Curve
        axes[0, 0].plot(self.equity_curve, linewidth=2, color='#2E86AB')
        axes[0, 0].axhline(y=self.initial_balance, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Equity Curve', fontweight='bold')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Balance (USDT)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Trade P&L Distribution
        colors = ['#06D6A0' if x > 0 else '#EF476F' for x in trades_df['pnl_usdt']]
        axes[0, 1].bar(range(len(trades_df)), trades_df['pnl_usdt'], color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[0, 1].set_title('Trade P&L Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('P&L (USDT)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative Returns
        cumulative_returns = (trades_df['pnl_usdt'].cumsum() / self.initial_balance) * 100
        axes[1, 0].plot(cumulative_returns, linewidth=2, color='#F77F00')
        axes[1, 0].fill_between(range(len(cumulative_returns)), 0, cumulative_returns, alpha=0.3, color='#F77F00')
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Cumulative Returns (%)', fontweight='bold')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Win/Loss Pie Chart
        win_loss_data = [self.winning_trades, self.losing_trades]
        colors_pie = ['#06D6A0', '#EF476F']
        axes[1, 1].pie(win_loss_data, labels=['Wins', 'Losses'], autopct='%1.1f%%',
                       colors=colors_pie, startangle=90)
        axes[1, 1].set_title('Win/Loss Ratio', fontweight='bold')
        
        # 5. Trade Duration Distribution
        durations_hours = [td.total_seconds() / 3600 for td in trades_df['duration']]
        axes[2, 0].hist(durations_hours, bins=20, color='#7209B7', alpha=0.7, edgecolor='black')
        axes[2, 0].set_title('Trade Duration Distribution', fontweight='bold')
        axes[2, 0].set_xlabel('Duration (hours)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].grid(True, alpha=0.3, axis='y')
        
        # 6. Monthly Returns
        trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
        monthly_returns = trades_df.groupby('month')['pnl_usdt'].sum()
        colors_monthly = ['#06D6A0' if x > 0 else '#EF476F' for x in monthly_returns]
        axes[2, 1].bar(range(len(monthly_returns)), monthly_returns, color=colors_monthly, alpha=0.7)
        axes[2, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[2, 1].set_title('Monthly Returns', fontweight='bold')
        axes[2, 1].set_xlabel('Month')
        axes[2, 1].set_ylabel('P&L (USDT)')
        axes[2, 1].set_xticks(range(len(monthly_returns)))
        axes[2, 1].set_xticklabels([str(m) for m in monthly_returns.index], rotation=45)
        axes[2, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Results plot saved to {save_path}")
        plt.show()


if __name__ == "__main__":
    # Ejemplo de uso con datos sintéticos
    logger.info("Backtest Engine Ready")
    
    # En producción, cargar datos reales con download_btc_futures_1m.py
    print("\nPara ejecutar backtest completo:")
    print("1. Ejecuta: python download_btc_futures_1m.py")
    print("2. Luego ejecuta este script con datos reales\n")

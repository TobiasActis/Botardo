"""
Futures Executor with Liquidation Protection
Ejecuta órdenes en Binance Futures con gestión automática de liquidación y riesgo
"""

import os
import time
from typing import Dict, Optional, Tuple
from decimal import Decimal, ROUND_DOWN
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class FuturesExecutor:
    """
    Executor de órdenes para Binance Futures con protección de liquidación
    """
    
    def __init__(self, testnet: bool = True):
        """
        Args:
            testnet: True para usar testnet, False para mainnet
        """
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set")
        
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.testnet = testnet
        
        if testnet:
            self.client.API_URL = 'https://testnet.binancefuture.com'
            logger.info("Using Binance Futures TESTNET")
        else:
            logger.warning("Using Binance Futures MAINNET - Real money!")
        
        # Cache de información del símbolo
        self.symbol_info_cache: Dict[str, Dict] = {}
        
    def get_symbol_info(self, symbol: str) -> Dict:
        """Obtiene información del símbolo (filters, precision, etc)"""
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        try:
            exchange_info = self.client.futures_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    self.symbol_info_cache[symbol] = s
                    return s
            
            raise ValueError(f"Symbol {symbol} not found")
        except BinanceAPIException as e:
            logger.error(f"Error getting symbol info: {e}")
            raise
    
    def get_step_size(self, symbol: str) -> float:
        """Obtiene el step size para cantidad mínima"""
        info = self.get_symbol_info(symbol)
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                return float(f['stepSize'])
        return 0.001
    
    def get_price_precision(self, symbol: str) -> int:
        """Obtiene la precisión de precio"""
        info = self.get_symbol_info(symbol)
        return info['pricePrecision']
    
    def get_quantity_precision(self, symbol: str) -> int:
        """Obtiene la precisión de cantidad"""
        info = self.get_symbol_info(symbol)
        return info['quantityPrecision']
    
    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Redondea cantidad según step size del símbolo"""
        step_size = self.get_step_size(symbol)
        precision = self.get_quantity_precision(symbol)
        
        quantity_decimal = Decimal(str(quantity))
        step_decimal = Decimal(str(step_size))
        
        rounded = (quantity_decimal // step_decimal) * step_decimal
        return float(rounded.quantize(Decimal(10) ** -precision, rounding=ROUND_DOWN))
    
    def round_price(self, symbol: str, price: float) -> float:
        """Redondea precio según precisión del símbolo"""
        precision = self.get_price_precision(symbol)
        return round(price, precision)
    
    def get_account_balance(self) -> float:
        """Obtiene balance disponible en USDT"""
        try:
            account = self.client.futures_account()
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['availableBalance'])
            return 0.0
        except BinanceAPIException as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0
    
    def get_current_price(self, symbol: str) -> float:
        """Obtiene precio actual del mercado"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error getting current price: {e}")
            raise
    
    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: int,
        side: str,
        margin_ratio: float = 0.01
    ) -> float:
        """
        Calcula precio de liquidación aproximado
        
        Args:
            entry_price: Precio de entrada
            leverage: Apalancamiento
            side: 'LONG' o 'SHORT'
            margin_ratio: Ratio de mantenimiento (default 1%)
        
        Returns:
            Precio de liquidación estimado
        """
        if side == 'LONG':
            liq_price = entry_price * (1 - (1 / leverage) + margin_ratio)
        else:  # SHORT
            liq_price = entry_price * (1 + (1 / leverage) - margin_ratio)
        
        return liq_price
    
    def calculate_position_size(
        self,
        symbol: str,
        risk_usdt: float,
        entry_price: float,
        stop_loss: float,
        max_leverage: int = 5
    ) -> Tuple[float, int]:
        """
        Calcula tamaño de posición óptimo basado en riesgo
        
        Args:
            symbol: Par de trading
            risk_usdt: Cantidad en USDT que se arriesga
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss
            max_leverage: Apalancamiento máximo permitido
        
        Returns:
            Tuple (quantity, leverage)
        """
        # Calcular distancia al stop loss en %
        stop_distance = abs(entry_price - stop_loss) / entry_price
        
        # Calcular leverage óptimo (mínimo entre calculado y máximo)
        calculated_leverage = min(int(1 / stop_distance), max_leverage)
        leverage = max(1, calculated_leverage)
        
        # Calcular cantidad en USDT
        position_value_usdt = risk_usdt / stop_distance
        
        # Aplicar leverage
        margin_required = position_value_usdt / leverage
        
        # Calcular quantity en unidades del asset
        quantity = position_value_usdt / entry_price
        
        # Redondear según reglas del exchange
        quantity = self.round_quantity(symbol, quantity)
        
        logger.info(f"Position sizing: {quantity} units, leverage {leverage}x, margin ${margin_required:.2f}")
        
        return quantity, leverage
    
    def set_leverage(self, symbol: str, leverage: int):
        """Establece apalancamiento para el símbolo"""
        try:
            result = self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            logger.info(f"Leverage set to {leverage}x for {symbol}")
            return result
        except BinanceAPIException as e:
            logger.error(f"Error setting leverage: {e}")
            raise
    
    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Abre posición con protección de liquidación
        
        Args:
            symbol: Par de trading (ej: 'BTCUSDT')
            side: 'LONG' o 'SHORT'
            quantity: Cantidad a tradear
            leverage: Apalancamiento
            stop_loss: Precio de stop loss (opcional)
            take_profit: Precio de take profit (opcional)
        
        Returns:
            Dict con información de la orden
        """
        # Validaciones
        if side not in ['LONG', 'SHORT']:
            raise ValueError("side must be 'LONG' or 'SHORT'")
        
        if leverage < 1 or leverage > 125:
            raise ValueError("leverage must be between 1 and 125")
        
        # Establecer leverage
        self.set_leverage(symbol, leverage)
        
        # Determinar side de Binance
        binance_side = 'BUY' if side == 'LONG' else 'SELL'
        
        # Redondear quantity
        quantity = self.round_quantity(symbol, quantity)
        
        try:
            # Abrir posición de mercado
            order = self.client.futures_create_order(
                symbol=symbol,
                side=binance_side,
                type='MARKET',
                quantity=quantity
            )
            
            logger.info(f"Position opened: {side} {quantity} {symbol} at leverage {leverage}x")
            
            # Calcular precio de liquidación
            entry_price = float(order['avgPrice']) if 'avgPrice' in order else self.get_current_price(symbol)
            liq_price = self.calculate_liquidation_price(entry_price, leverage, side)
            
            logger.warning(f"Liquidation price: ${liq_price:.2f}")
            
            # Colocar stop loss si se especificó
            if stop_loss:
                self.place_stop_loss(symbol, side, quantity, stop_loss)
            
            # Colocar take profit si se especificó
            if take_profit:
                self.place_take_profit(symbol, side, quantity, take_profit)
            
            return {
                'order': order,
                'entry_price': entry_price,
                'liquidation_price': liq_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except BinanceAPIException as e:
            logger.error(f"Error opening position: {e}")
            raise
    
    def place_stop_loss(self, symbol: str, side: str, quantity: float, stop_price: float):
        """Coloca orden de stop loss"""
        # Side opuesto para cerrar posición
        close_side = 'SELL' if side == 'LONG' else 'BUY'
        stop_price = self.round_price(symbol, stop_price)
        
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type='STOP_MARKET',
                stopPrice=stop_price,
                quantity=quantity,
                closePosition=True
            )
            logger.info(f"Stop loss placed at ${stop_price}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing stop loss: {e}")
            raise
    
    def place_take_profit(self, symbol: str, side: str, quantity: float, tp_price: float):
        """Coloca orden de take profit"""
        # Side opuesto para cerrar posición
        close_side = 'SELL' if side == 'LONG' else 'BUY'
        tp_price = self.round_price(symbol, tp_price)
        
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=tp_price,
                quantity=quantity,
                closePosition=True
            )
            logger.info(f"Take profit placed at ${tp_price}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing take profit: {e}")
            raise
    
    def get_open_positions(self) -> list:
        """Obtiene posiciones abiertas"""
        try:
            positions = self.client.futures_position_information()
            return [p for p in positions if float(p['positionAmt']) != 0]
        except BinanceAPIException as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def close_all_positions(self, symbol: str = None):
        """Cierra todas las posiciones (opcionalmente filtrado por símbolo)"""
        positions = self.get_open_positions()
        
        for pos in positions:
            if symbol and pos['symbol'] != symbol:
                continue
            
            pos_symbol = pos['symbol']
            pos_amt = float(pos['positionAmt'])
            
            if pos_amt == 0:
                continue
            
            side = 'SELL' if pos_amt > 0 else 'BUY'
            quantity = abs(pos_amt)
            
            try:
                order = self.client.futures_create_order(
                    symbol=pos_symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
                logger.info(f"Closed position: {pos_symbol}")
            except BinanceAPIException as e:
                logger.error(f"Error closing position {pos_symbol}: {e}")


if __name__ == "__main__":
    # Ejemplo de uso
    logger.info("Futures Executor initialized")
    
    # Crear executor (testnet por defecto)
    executor = FuturesExecutor(testnet=True)
    
    # Verificar balance
    balance = executor.get_account_balance()
    logger.info(f"Available balance: ${balance:.2f}")
    
    # Ejemplo: Calcular posición
    symbol = "BTCUSDT"
    risk_usdt = 100  # Arriesgar $100
    entry_price = 43000
    stop_loss = 42500
    
    qty, lev = executor.calculate_position_size(
        symbol=symbol,
        risk_usdt=risk_usdt,
        entry_price=entry_price,
        stop_loss=stop_loss,
        max_leverage=5
    )
    
    logger.info(f"Calculated position: {qty} BTC at {lev}x leverage")
    
    # Descomentar para abrir posición real (TESTNET)
    # result = executor.open_position(
    #     symbol=symbol,
    #     side='LONG',
    #     quantity=qty,
    #     leverage=lev,
    #     stop_loss=stop_loss,
    #     take_profit=44000
    # )

"""
Smart Money Concepts (SMC) Engine
Detección de Order Blocks, FVGs, BOS/CHoCH, Liquidity Sweeps
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
from loguru import logger


class StructureType(Enum):
    """Tipo de estructura de mercado"""
    BOS_BULLISH = "BOS_BULL"      # Break of Structure alcista
    BOS_BEARISH = "BOS_BEAR"      # Break of Structure bajista
    CHOCH_BULLISH = "CHOCH_BULL"  # Change of Character alcista
    CHOCH_BEARISH = "CHOCH_BEAR"  # Change of Character bajista


class OrderBlockType(Enum):
    """Tipo de Order Block"""
    BULLISH = "BULL_OB"   # OB de demanda (compra)
    BEARISH = "BEAR_OB"   # OB de oferta (venta)


class SMCEngine:
    """
    Motor de análisis Smart Money Concepts
    """
    
    def __init__(
        self,
        swing_length: int = 5,
        fvg_threshold: float = 0.001,  # 0.1% mínimo para considerar FVG
        ob_lookback: int = 20,
        liquidity_lookback: int = 50
    ):
        """
        Args:
            swing_length: Períodos para identificar swing highs/lows
            fvg_threshold: Umbral mínimo para FVG (% del precio)
            ob_lookback: Cuántas velas mirar atrás para OBs
            liquidity_lookback: Cuántas velas para zonas de liquidez
        """
        self.swing_length = swing_length
        self.fvg_threshold = fvg_threshold
        self.ob_lookback = ob_lookback
        self.liquidity_lookback = liquidity_lookback
        
    def detect_swing_points(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detecta swing highs y swing lows
        
        Returns:
            (swing_highs, swing_lows): Series booleanas
        """
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        
        swing_highs = pd.Series(False, index=df.index)
        swing_lows = pd.Series(False, index=df.index)
        
        for i in range(self.swing_length, n - self.swing_length):
            # Swing high: máximo local
            if highs[i] == max(highs[i - self.swing_length:i + self.swing_length + 1]):
                swing_highs.iloc[i] = True
            
            # Swing low: mínimo local
            if lows[i] == min(lows[i - self.swing_length:i + self.swing_length + 1]):
                swing_lows.iloc[i] = True
        
        return swing_highs, swing_lows
    
    def detect_market_structure(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detecta BOS (Break of Structure) y CHoCH (Change of Character)
        
        Returns:
            Lista de estructuras detectadas
        """
        swing_highs, swing_lows = self.detect_swing_points(df)
        structures = []
        
        # Obtener índices de swings
        high_indices = df.index[swing_highs].tolist()
        low_indices = df.index[swing_lows].tolist()
        
        if len(high_indices) < 2 or len(low_indices) < 2:
            return structures
        
        # Variables de estado
        last_high = None
        last_low = None
        trend = None  # 'bull' o 'bear'
        
        # Combinar y ordenar todos los swings
        all_swings = []
        for idx in high_indices:
            all_swings.append({'time': idx, 'type': 'high', 'price': df.loc[idx, 'high']})
        for idx in low_indices:
            all_swings.append({'time': idx, 'type': 'low', 'price': df.loc[idx, 'low']})
        
        all_swings.sort(key=lambda x: x['time'])
        
        for swing in all_swings:
            if swing['type'] == 'high':
                if last_high is not None:
                    # Higher High (HH)
                    if swing['price'] > last_high['price']:
                        if trend == 'bear':
                            # CHoCH: cambio de bajista a alcista
                            structures.append({
                                'time': swing['time'],
                                'type': StructureType.CHOCH_BULLISH,
                                'price': swing['price'],
                                'previous_price': last_high['price']
                            })
                            trend = 'bull'
                        elif trend == 'bull':
                            # BOS alcista
                            structures.append({
                                'time': swing['time'],
                                'type': StructureType.BOS_BULLISH,
                                'price': swing['price'],
                                'previous_price': last_high['price']
                            })
                    # Lower High (LH)
                    elif swing['price'] < last_high['price'] and trend == 'bull':
                        # Posible debilidad
                        pass
                
                last_high = swing
            
            elif swing['type'] == 'low':
                if last_low is not None:
                    # Lower Low (LL)
                    if swing['price'] < last_low['price']:
                        if trend == 'bull':
                            # CHoCH: cambio de alcista a bajista
                            structures.append({
                                'time': swing['time'],
                                'type': StructureType.CHOCH_BEARISH,
                                'price': swing['price'],
                                'previous_price': last_low['price']
                            })
                            trend = 'bear'
                        elif trend == 'bear':
                            # BOS bajista
                            structures.append({
                                'time': swing['time'],
                                'type': StructureType.BOS_BEARISH,
                                'price': swing['price'],
                                'previous_price': last_low['price']
                            })
                    # Higher Low (HL)
                    elif swing['price'] > last_low['price'] and trend == 'bear':
                        # Posible debilidad
                        pass
                
                last_low = swing
            
            # Inicializar tendencia en el primer swing relevante
            if trend is None:
                if swing['type'] == 'high' and last_low is not None:
                    trend = 'bull'
                elif swing['type'] == 'low' and last_high is not None:
                    trend = 'bear'
        
        return structures
    
    def detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detecta Order Blocks (últimas velas antes de movimientos impulsivos)
        
        Returns:
            Lista de Order Blocks detectados
        """
        order_blocks = []
        n = len(df)
        
        for i in range(self.ob_lookback, n - 1):
            current = df.iloc[i]
            next_candle = df.iloc[i + 1]
            
            # Bullish Order Block: última vela bajista antes de impulso alcista
            if (current['close'] < current['open'] and  # Vela bajista
                next_candle['close'] > next_candle['open'] and  # Siguiente alcista
                (next_candle['close'] - next_candle['open']) > 2 * (current['open'] - current['close'])):  # Impulso
                
                order_blocks.append({
                    'time': df.index[i],
                    'type': OrderBlockType.BULLISH,
                    'top': current['high'],
                    'bottom': current['low'],
                    'mitigated': False,
                    'mitigation_time': None
                })
            
            # Bearish Order Block: última vela alcista antes de impulso bajista
            elif (current['close'] > current['open'] and  # Vela alcista
                  next_candle['close'] < next_candle['open'] and  # Siguiente bajista
                  (next_candle['open'] - next_candle['close']) > 2 * (current['close'] - current['open'])):  # Impulso
                
                order_blocks.append({
                    'time': df.index[i],
                    'type': OrderBlockType.BEARISH,
                    'top': current['high'],
                    'bottom': current['low'],
                    'mitigated': False,
                    'mitigation_time': None
                })
        
        # Verificar mitigación de OBs
        for ob in order_blocks:
            ob_idx = df.index.get_loc(ob['time'])
            for i in range(ob_idx + 1, n):
                candle = df.iloc[i]
                
                if ob['type'] == OrderBlockType.BULLISH:
                    # OB alcista se mitiga cuando precio entra en la zona
                    if candle['low'] <= ob['top'] and candle['low'] >= ob['bottom']:
                        ob['mitigated'] = True
                        ob['mitigation_time'] = df.index[i]
                        break
                
                elif ob['type'] == OrderBlockType.BEARISH:
                    # OB bajista se mitiga cuando precio entra en la zona
                    if candle['high'] >= ob['bottom'] and candle['high'] <= ob['top']:
                        ob['mitigated'] = True
                        ob['mitigation_time'] = df.index[i]
                        break
        
        return order_blocks
    
    def detect_fvg(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detecta Fair Value Gaps (imbalances en el precio)
        
        Returns:
            Lista de FVGs detectados
        """
        fvgs = []
        n = len(df)
        
        for i in range(1, n - 1):
            prev = df.iloc[i - 1]
            current = df.iloc[i]
            next_candle = df.iloc[i + 1]
            
            # Bullish FVG: gap entre low de vela 3 y high de vela 1
            bullish_gap = next_candle['low'] - prev['high']
            if bullish_gap > 0 and bullish_gap / current['close'] >= self.fvg_threshold:
                fvgs.append({
                    'time': df.index[i],
                    'type': 'bullish',
                    'top': next_candle['low'],
                    'bottom': prev['high'],
                    'gap_size': bullish_gap,
                    'filled': False,
                    'fill_time': None
                })
            
            # Bearish FVG: gap entre high de vela 3 y low de vela 1
            bearish_gap = prev['low'] - next_candle['high']
            if bearish_gap > 0 and bearish_gap / current['close'] >= self.fvg_threshold:
                fvgs.append({
                    'time': df.index[i],
                    'type': 'bearish',
                    'top': prev['low'],
                    'bottom': next_candle['high'],
                    'gap_size': bearish_gap,
                    'filled': False,
                    'fill_time': None
                })
        
        # Verificar si FVGs se llenaron
        for fvg in fvgs:
            fvg_idx = df.index.get_loc(fvg['time'])
            for i in range(fvg_idx + 2, n):
                candle = df.iloc[i]
                
                if fvg['type'] == 'bullish':
                    # FVG alcista se llena si precio baja al gap
                    if candle['low'] <= fvg['top']:
                        fvg['filled'] = True
                        fvg['fill_time'] = df.index[i]
                        break
                
                elif fvg['type'] == 'bearish':
                    # FVG bajista se llena si precio sube al gap
                    if candle['high'] >= fvg['bottom']:
                        fvg['filled'] = True
                        fvg['fill_time'] = df.index[i]
                        break
        
        return fvgs
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detecta barridas de liquidez (sweeps de highs/lows anteriores)
        
        Returns:
            Lista de liquidity sweeps
        """
        sweeps = []
        swing_highs, swing_lows = self.detect_swing_points(df)
        
        high_indices = df.index[swing_highs].tolist()
        low_indices = df.index[swing_lows].tolist()
        
        n = len(df)
        
        # Detectar sweeps de swing highs (SSL - Sell Side Liquidity)
        for i in range(len(high_indices)):
            swing_time = high_indices[i]
            swing_price = df.loc[swing_time, 'high']
            swing_idx = df.index.get_loc(swing_time)
            
            # Buscar si hay un sweep posterior
            for j in range(swing_idx + 1, min(swing_idx + self.liquidity_lookback, n)):
                candle = df.iloc[j]
                
                # Sweep: precio supera el high pero luego rechaza
                if candle['high'] > swing_price:
                    # Verificar rechazo (close por debajo del high)
                    if candle['close'] < swing_price or (j < n - 1 and df.iloc[j + 1]['close'] < swing_price):
                        sweeps.append({
                            'time': df.index[j],
                            'type': 'buy_side_liquidity_sweep',  # BSL
                            'swept_price': swing_price,
                            'sweep_high': candle['high'],
                            'rejection': True
                        })
                        break
        
        # Detectar sweeps de swing lows (BSL - Buy Side Liquidity)
        for i in range(len(low_indices)):
            swing_time = low_indices[i]
            swing_price = df.loc[swing_time, 'low']
            swing_idx = df.index.get_loc(swing_time)
            
            # Buscar si hay un sweep posterior
            for j in range(swing_idx + 1, min(swing_idx + self.liquidity_lookback, n)):
                candle = df.iloc[j]
                
                # Sweep: precio rompe el low pero luego rechaza
                if candle['low'] < swing_price:
                    # Verificar rechazo (close por encima del low)
                    if candle['close'] > swing_price or (j < n - 1 and df.iloc[j + 1]['close'] > swing_price):
                        sweeps.append({
                            'time': df.index[j],
                            'type': 'sell_side_liquidity_sweep',  # SSL
                            'swept_price': swing_price,
                            'sweep_low': candle['low'],
                            'rejection': True
                        })
                        break
        
        return sweeps
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_rsi_divergence(self, df: pd.DataFrame, rsi_period: int = 14) -> List[Dict]:
        """
        Detecta divergencias entre precio y RSI
        
        Returns:
            Lista de divergencias
        """
        divergences = []
        rsi = self.calculate_rsi(df, rsi_period)
        
        swing_highs, swing_lows = self.detect_swing_points(df)
        
        high_indices = df.index[swing_highs].tolist()
        low_indices = df.index[swing_lows].tolist()
        
        # Divergencia bajista: precio hace HH pero RSI hace LH
        for i in range(1, len(high_indices)):
            prev_idx = high_indices[i - 1]
            curr_idx = high_indices[i]
            
            prev_price = df.loc[prev_idx, 'high']
            curr_price = df.loc[curr_idx, 'high']
            prev_rsi = rsi.loc[prev_idx]
            curr_rsi = rsi.loc[curr_idx]
            
            if curr_price > prev_price and curr_rsi < prev_rsi:
                divergences.append({
                    'time': curr_idx,
                    'type': 'bearish',
                    'price_prev': prev_price,
                    'price_curr': curr_price,
                    'rsi_prev': prev_rsi,
                    'rsi_curr': curr_rsi
                })
        
        # Divergencia alcista: precio hace LL pero RSI hace HL
        for i in range(1, len(low_indices)):
            prev_idx = low_indices[i - 1]
            curr_idx = low_indices[i]
            
            prev_price = df.loc[prev_idx, 'low']
            curr_price = df.loc[curr_idx, 'low']
            prev_rsi = rsi.loc[prev_idx]
            curr_rsi = rsi.loc[curr_idx]
            
            if curr_price < prev_price and curr_rsi > prev_rsi:
                divergences.append({
                    'time': curr_idx,
                    'type': 'bullish',
                    'price_prev': prev_price,
                    'price_curr': curr_price,
                    'rsi_prev': prev_rsi,
                    'rsi_curr': curr_rsi
                })
        
        return divergences
    
    def get_trading_signals(
        self,
        df: pd.DataFrame,
        current_time: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Genera señales de trading basadas en confluencia SMC
        
        Returns:
            Señal con confluencia de múltiples factores SMC
        """
        # Analizar todos los componentes
        structures = self.detect_market_structure(df)
        order_blocks = self.detect_order_blocks(df)
        fvgs = self.detect_fvg(df)
        sweeps = self.detect_liquidity_sweeps(df)
        divergences = self.detect_rsi_divergence(df)
        rsi = self.calculate_rsi(df)
        
        # Obtener datos actuales
        current_idx = df.index.get_loc(current_time)
        current_price = df.loc[current_time, 'close']
        current_rsi = rsi.loc[current_time]
        
        signal = None
        confluence_score = 0
        reasons = []
        
        # 1. Verificar estructura reciente (últimos 10 períodos)
        recent_structures = [s for s in structures if df.index.get_loc(s['time']) >= current_idx - 10]
        
        # 2. Verificar OBs no mitigados cercanos
        active_obs = [ob for ob in order_blocks if not ob['mitigated']]
        nearby_bull_ob = any(
            ob['type'] == OrderBlockType.BULLISH and 
            current_price >= ob['bottom'] * 0.99 and 
            current_price <= ob['top'] * 1.01
            for ob in active_obs
        )
        nearby_bear_ob = any(
            ob['type'] == OrderBlockType.BEARISH and 
            current_price >= ob['bottom'] * 0.99 and 
            current_price <= ob['top'] * 1.01
            for ob in active_obs
        )
        
        # 3. Verificar FVGs sin llenar
        active_fvgs = [fvg for fvg in fvgs if not fvg['filled']]
        in_bull_fvg = any(
            fvg['type'] == 'bullish' and 
            current_price >= fvg['bottom'] and 
            current_price <= fvg['top']
            for fvg in active_fvgs
        )
        in_bear_fvg = any(
            fvg['type'] == 'bearish' and 
            current_price >= fvg['bottom'] and 
            current_price <= fvg['top']
            for fvg in active_fvgs
        )
        
        # 4. Verificar sweeps recientes
        recent_sweeps = [s for s in sweeps if df.index.get_loc(s['time']) >= current_idx - 5]
        ssl_sweep = any(s['type'] == 'sell_side_liquidity_sweep' for s in recent_sweeps)
        bsl_sweep = any(s['type'] == 'buy_side_liquidity_sweep' for s in recent_sweeps)
        
        # 5. Verificar divergencias recientes
        recent_divergences = [d for d in divergences if df.index.get_loc(d['time']) >= current_idx - 10]
        bull_div = any(d['type'] == 'bullish' for d in recent_divergences)
        bear_div = any(d['type'] == 'bearish' for d in recent_divergences)
        
        # SEÑAL ALCISTA
        if (nearby_bull_ob or in_bull_fvg or ssl_sweep) and current_rsi < 50:
            signal = 'long'
            
            if nearby_bull_ob:
                confluence_score += 3
                reasons.append("Bullish Order Block")
            if in_bull_fvg:
                confluence_score += 2
                reasons.append("In Bullish FVG")
            if ssl_sweep:
                confluence_score += 3
                reasons.append("SSL Sweep (reversal)")
            if bull_div:
                confluence_score += 2
                reasons.append("Bullish RSI Divergence")
            if current_rsi < 30:
                confluence_score += 2
                reasons.append("RSI Oversold")
            if recent_structures and recent_structures[-1]['type'] in [StructureType.BOS_BULLISH, StructureType.CHOCH_BULLISH]:
                confluence_score += 2
                reasons.append("Bullish Structure")
        
        # SEÑAL BAJISTA
        elif (nearby_bear_ob or in_bear_fvg or bsl_sweep) and current_rsi > 50:
            signal = 'short'
            
            if nearby_bear_ob:
                confluence_score += 3
                reasons.append("Bearish Order Block")
            if in_bear_fvg:
                confluence_score += 2
                reasons.append("In Bearish FVG")
            if bsl_sweep:
                confluence_score += 3
                reasons.append("BSL Sweep (reversal)")
            if bear_div:
                confluence_score += 2
                reasons.append("Bearish RSI Divergence")
            if current_rsi > 70:
                confluence_score += 2
                reasons.append("RSI Overbought")
            if recent_structures and recent_structures[-1]['type'] in [StructureType.BOS_BEARISH, StructureType.CHOCH_BEARISH]:
                confluence_score += 2
                reasons.append("Bearish Structure")
        
        # Solo retornar señal si hay suficiente confluencia (mínimo 5 puntos)
        if signal and confluence_score >= 5:
            # Calcular stop loss y take profit basados en estructura
            atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            atr_value = atr.iloc[-1] * 0.5
            
            if signal == 'long':
                stop_loss = current_price - atr_value
                take_profit = current_price + (atr_value * 2)  # R:R 1:2
            else:
                stop_loss = current_price + atr_value
                take_profit = current_price - (atr_value * 2)
            
            return {
                'signal': signal,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confluence_score': confluence_score,
                'reasons': reasons,
                'rsi': current_rsi,
                'timestamp': current_time
            }
        
        return None


if __name__ == "__main__":
    logger.info("SMC Engine loaded successfully")

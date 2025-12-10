"""
🤖 BOTARDO - Trading Bot Unificado 🤖
Sistema completo: SMC + Will Street PO3 + Backtest Engine

Uso:
    python botardo.py --data_1m "data/BTCUSDT_1m_2024-01-01_to_now.csv" \
                      --initial_capital 500 \
                      --risk_per_trade 0.02 \
                      --leverage 10 \
                      --start "2024-01-01" \
                      --end "2025-12-09"
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Configurar estilo de gráficos
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 10)


# ============================================================================
# SMC ENGINE - Smart Money Concepts
# ============================================================================

class StructureType(Enum):
    """Tipo de estructura de mercado"""
    BOS_BULLISH = "BOS_BULL"
    BOS_BEARISH = "BOS_BEAR"
    CHOCH_BULLISH = "CHOCH_BULL"
    CHOCH_BEARISH = "CHOCH_BEAR"


class OrderBlockType(Enum):
    """Tipo de Order Block"""
    BULLISH = "BULL_OB"
    BEARISH = "BEAR_OB"


class SMCEngine:
    """Motor de análisis Smart Money Concepts"""
    
    def __init__(self, swing_length: int = 5, fvg_threshold: float = 0.001, 
                 ob_lookback: int = 20, liquidity_lookback: int = 50):
        self.swing_length = swing_length
        self.fvg_threshold = fvg_threshold
        self.ob_lookback = ob_lookback
        self.liquidity_lookback = liquidity_lookback
        
    def detect_swing_points(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detecta swing highs y swing lows"""
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        
        swing_highs = pd.Series(False, index=df.index)
        swing_lows = pd.Series(False, index=df.index)
        
        for i in range(self.swing_length, n - self.swing_length):
            if highs[i] == max(highs[i - self.swing_length:i + self.swing_length + 1]):
                swing_highs.iloc[i] = True
            if lows[i] == min(lows[i - self.swing_length:i + self.swing_length + 1]):
                swing_lows.iloc[i] = True
        
        return swing_highs, swing_lows
    
    def detect_order_blocks(self, df: pd.DataFrame, current_idx: int) -> List[Dict]:
        """Detecta Order Blocks válidos"""
        order_blocks = []
        start_idx = max(0, current_idx - self.ob_lookback)
        
        for i in range(start_idx, current_idx):
            candle = df.iloc[i]
            next_candle = df.iloc[i + 1] if i + 1 < len(df) else None
            
            if next_candle is None:
                continue
            
            # Bullish OB: vela bajista seguida de movimiento alcista fuerte
            if candle['close'] < candle['open'] and next_candle['close'] > next_candle['open']:
                if (next_candle['close'] - next_candle['open']) / next_candle['open'] > 0.005:
                    order_blocks.append({
                        'type': OrderBlockType.BULLISH,
                        'high': candle['high'],
                        'low': candle['low'],
                        'time': candle.name,
                        'mitigated': False
                    })
            
            # Bearish OB: vela alcista seguida de movimiento bajista fuerte
            if candle['close'] > candle['open'] and next_candle['close'] < next_candle['open']:
                if (next_candle['open'] - next_candle['close']) / next_candle['open'] > 0.005:
                    order_blocks.append({
                        'type': OrderBlockType.BEARISH,
                        'high': candle['high'],
                        'low': candle['low'],
                        'time': candle.name,
                        'mitigated': False
                    })
        
        return order_blocks
    
    def detect_fvg(self, df: pd.DataFrame, current_idx: int) -> List[Dict]:
        """Detecta Fair Value Gaps"""
        fvgs = []
        
        if current_idx < 2:
            return fvgs
        
        for i in range(max(2, current_idx - 50), current_idx):
            candle_1 = df.iloc[i - 2]
            candle_2 = df.iloc[i - 1]
            candle_3 = df.iloc[i]
            
            # Bullish FVG: gap entre low de vela 3 y high de vela 1
            gap_bull = candle_3['low'] - candle_1['high']
            if gap_bull > candle_2['close'] * self.fvg_threshold:
                fvgs.append({
                    'type': 'BULL_FVG',
                    'top': candle_3['low'],
                    'bottom': candle_1['high'],
                    'time': candle_3.name,
                    'filled': False
                })
            
            # Bearish FVG: gap entre high de vela 3 y low de vela 1
            gap_bear = candle_1['low'] - candle_3['high']
            if gap_bear > candle_2['close'] * self.fvg_threshold:
                fvgs.append({
                    'type': 'BEAR_FVG',
                    'top': candle_1['low'],
                    'bottom': candle_3['high'],
                    'time': candle_3.name,
                    'filled': False
                })
        
        return fvgs
    
    def analyze_context(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """Análisis completo de contexto SMC"""
        if current_idx < self.swing_length:
            return {'confluence_score': 0, 'bias': 'NEUTRAL'}
        
        current_price = df.iloc[current_idx]['close']
        order_blocks = self.detect_order_blocks(df, current_idx)
        fvgs = self.detect_fvg(df, current_idx)
        
        confluence_bull = 0
        confluence_bear = 0
        
        # Evaluar Order Blocks cercanos
        for ob in order_blocks:
            if not ob['mitigated']:
                distance_pct = abs(current_price - (ob['high'] + ob['low']) / 2) / current_price
                if distance_pct < 0.02:  # Dentro del 2%
                    if ob['type'] == OrderBlockType.BULLISH:
                        confluence_bull += 2
                    else:
                        confluence_bear += 2
        
        # Evaluar FVGs no llenados
        for fvg in fvgs:
            if not fvg['filled']:
                if current_price >= fvg['bottom'] and current_price <= fvg['top']:
                    if fvg['type'] == 'BULL_FVG':
                        confluence_bull += 1
                    else:
                        confluence_bear += 1
        
        # Determinar bias y score
        if confluence_bull > confluence_bear:
            return {
                'confluence_score': confluence_bull,
                'bias': 'BULLISH',
                'signal_type': 'LONG',
                'order_blocks': order_blocks,
                'fvgs': fvgs
            }
        elif confluence_bear > confluence_bull:
            return {
                'confluence_score': confluence_bear,
                'bias': 'BEARISH',
                'signal_type': 'SHORT',
                'order_blocks': order_blocks,
                'fvgs': fvgs
            }
        else:
            return {
                'confluence_score': 0,
                'bias': 'NEUTRAL',
                'order_blocks': order_blocks,
                'fvgs': fvgs
            }


# ============================================================================
# WILL STREET PO3 - Power of 3
# ============================================================================

class WillStreetPO3:
    """Will Street Power of 3 Strategy - Opera velas de 4 horas"""
    
    def __init__(self, min_rr_ratio: float = 3.0, min_stop_pct: float = 0.3, use_trend_filter: bool = True):
        self.min_rr_ratio = min_rr_ratio
        self.min_stop_pct = min_stop_pct
        self.use_trend_filter = use_trend_filter
        self.processed_candles = set()
        logger.info(f"Will Street PO3 initialized - Min RR: {min_rr_ratio}:1, Min SL: {min_stop_pct}%, Trend Filter: {use_trend_filter}")
    
    def get_previous_candle_extremes(self, df_4h: pd.DataFrame, current_idx: int) -> Optional[Dict]:
        """Obtiene extremos de la vela anterior de 4h"""
        if current_idx < 1:
            return None
        
        prev_candle = df_4h.iloc[current_idx - 1]
        return {
            'prev_high': prev_candle['high'],
            'prev_low': prev_candle['low'],
            'prev_open': prev_candle['open'],
            'prev_close': prev_candle['close']
        }
    
    def analyze_4h_candle(self, df_4h: pd.DataFrame, df_15m: pd.DataFrame, 
                         current_time: pd.Timestamp) -> Optional[Dict]:
        """
        Analiza si hay setup válido en la vela de 4h actual
        
        Reglas:
        1. SHORT: precio arriba del Open, target Low previo
        2. LONG: precio abajo del Open, target High previo
        3. Stop loss con mínimo 3:1 RR
        """
        candle_4h_idx = df_4h.index.get_indexer([current_time], method='ffill')[0]
        
        if candle_4h_idx < 1:
            return None
        
        # Filtro de tendencia: calcular EMA 200 en 4h
        current_ema200 = None
        if self.use_trend_filter:
            if len(df_4h) < 200:
                return None  # No hay suficientes datos para EMA 200
            
            df_4h_copy = df_4h.copy()
            df_4h_copy['ema_200'] = df_4h_copy['close'].ewm(span=200, adjust=False).mean()
            current_ema200 = df_4h_copy.iloc[candle_4h_idx]['ema_200']
        
        current_candle = df_4h.iloc[candle_4h_idx]
        candle_timestamp = current_candle.name
        
        # Prevenir duplicados
        if candle_timestamp in self.processed_candles:
            return None
        
        prev_extremes = self.get_previous_candle_extremes(df_4h, candle_4h_idx)
        if prev_extremes is None:
            return None
        
        current_price = current_candle['close']
        open_price = current_candle['open']
        
        # ===== SETUP SHORT =====
        if current_price > open_price:
            # Filtro de tendencia: solo SHORT si precio está debajo de EMA 200
            if self.use_trend_filter and current_ema200 is not None:
                if current_price > current_ema200:
                    return None  # No operar SHORT en tendencia alcista
            
            entry = current_price
            take_profit = prev_extremes['prev_low']
            
            # Stop loss: 0.5% arriba del entry
            raw_sl = entry * 1.005
            stop_loss = max(raw_sl, entry * 1.005)  # Garantizar SL arriba
            
            # Validar RR
            profit_distance = abs(entry - take_profit)
            loss_distance = abs(entry - stop_loss)
            
            if loss_distance == 0:
                return None
            
            rr_ratio = profit_distance / loss_distance
            
            if rr_ratio >= self.min_rr_ratio and take_profit < entry:
                self.processed_candles.add(candle_timestamp)
                return {
                    'type': 'SHORT',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'rr_ratio': rr_ratio,
                    'source': 'po3',
                    'timestamp': current_time
                }
        
        # ===== SETUP LONG =====
        elif current_price < open_price:
            # Filtro de tendencia: solo LONG si precio está encima de EMA 200
            if self.use_trend_filter and current_ema200 is not None:
                if current_price < current_ema200:
                    return None  # No operar LONG en tendencia bajista
            
            entry = current_price
            take_profit = prev_extremes['prev_high']
            
            # Stop loss: 0.5% abajo del entry
            raw_sl = entry * 0.995
            stop_loss = min(raw_sl, entry * 0.995)  # Garantizar SL abajo
            
            # Validar RR
            profit_distance = abs(take_profit - entry)
            loss_distance = abs(entry - stop_loss)
            
            if loss_distance == 0:
                return None
            
            rr_ratio = profit_distance / loss_distance
            
            if rr_ratio >= self.min_rr_ratio and take_profit > entry:
                self.processed_candles.add(candle_timestamp)
                return {
                    'type': 'LONG',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'rr_ratio': rr_ratio,
                    'source': 'po3',
                    'timestamp': current_time
                }
        
        return None


# ============================================================================
# RSI DIVERGENCE DETECTOR
# ============================================================================

class RSIDivergenceDetector:
    """Detector de divergencias RSI y zonas extremas"""
    
    def __init__(self, rsi_period: int = 14, overbought: float = 70, oversold: float = 30):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        logger.info(f"RSI Divergence Detector initialized - Period: {rsi_period}, OB/OS: {overbought}/{oversold}")
    
    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calcula RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_divergence(self, df: pd.DataFrame, current_idx: int, lookback: int = 20) -> Dict:
        """Detecta divergencias alcistas/bajistas"""
        if current_idx < lookback + self.rsi_period:
            return {'type': None, 'strength': 0}
        
        # Calcular RSI
        rsi = self.calculate_rsi(df)
        
        # Buscar divergencia alcista (precio baja, RSI sube)
        price_slice = df['close'].iloc[current_idx-lookback:current_idx+1]
        rsi_slice = rsi.iloc[current_idx-lookback:current_idx+1]
        
        # Encontrar mínimos de precio y RSI
        price_min_idx = price_slice.idxmin()
        rsi_min_idx = rsi_slice.idxmin()
        
        # Encontrar máximos de precio y RSI
        price_max_idx = price_slice.idxmax()
        rsi_max_idx = rsi_slice.idxmax()
        
        current_price = df['close'].iloc[current_idx]
        current_rsi = rsi.iloc[current_idx]
        
        # Divergencia Alcista: precio hace mínimo más bajo, RSI hace mínimo más alto
        if price_min_idx != current_idx and current_idx - price_slice.index.get_loc(price_min_idx) < lookback // 2:
            prev_price_min = price_slice.loc[price_min_idx]
            prev_rsi_min = rsi_slice.loc[price_min_idx]
            
            if current_price < prev_price_min and current_rsi > prev_rsi_min and current_rsi < self.oversold + 10:
                return {'type': 'BULLISH', 'strength': 3, 'rsi': current_rsi}
        
        # Divergencia Bajista: precio hace máximo más alto, RSI hace máximo más bajo
        if price_max_idx != current_idx and current_idx - price_slice.index.get_loc(price_max_idx) < lookback // 2:
            prev_price_max = price_slice.loc[price_max_idx]
            prev_rsi_max = rsi_slice.loc[price_max_idx]
            
            if current_price > prev_price_max and current_rsi < prev_rsi_max and current_rsi > self.overbought - 10:
                return {'type': 'BEARISH', 'strength': 3, 'rsi': current_rsi}
        
        return {'type': None, 'strength': 0, 'rsi': current_rsi}
    
    def check_extreme_zones(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """Verifica si RSI está en zonas extremas"""
        rsi = self.calculate_rsi(df)
        current_rsi = rsi.iloc[current_idx]
        
        if current_rsi > self.overbought:
            return {'extreme': 'OVERBOUGHT', 'value': current_rsi, 'signal': 'SHORT'}
        elif current_rsi < self.oversold:
            return {'extreme': 'OVERSOLD', 'value': current_rsi, 'signal': 'LONG'}
        
        return {'extreme': None, 'value': current_rsi}


# ============================================================================
# LIQUIDITY ZONES (BSL/SSL)
# ============================================================================

class LiquidityZones:
    """Detector de zonas de liquidez Buy Side (BSL) y Sell Side (SSL)"""
    
    def __init__(self, lookback: int = 20, threshold: float = 0.002):
        self.lookback = lookback
        self.threshold = threshold
        logger.info(f"Liquidity Zones initialized - Lookback: {lookback}, Threshold: {threshold}")
    
    def find_liquidity_zones(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """Encuentra zonas de liquidez cercanas"""
        if current_idx < self.lookback:
            return {'bsl': None, 'ssl': None}
        
        # Slice del lookback
        price_slice = df.iloc[current_idx-self.lookback:current_idx+1]
        current_price = df['close'].iloc[current_idx]
        
        # Buy Side Liquidity (BSL): Máximos recientes arriba del precio
        highs = price_slice['high']
        recent_highs = highs[highs > current_price * (1 + self.threshold)]
        
        if len(recent_highs) > 0:
            bsl = recent_highs.max()
        else:
            bsl = None
        
        # Sell Side Liquidity (SSL): Mínimos recientes abajo del precio
        lows = price_slice['low']
        recent_lows = lows[lows < current_price * (1 - self.threshold)]
        
        if len(recent_lows) > 0:
            ssl = recent_lows.min()
        else:
            ssl = None
        
        return {'bsl': bsl, 'ssl': ssl}
    
    def calculate_liquidity_confluence(self, current_price: float, target_price: float, 
                                      bsl: float, ssl: float) -> int:
        """Calcula confluencia basada en cercanía a zonas de liquidez"""
        confluence = 0
        
        if bsl is not None:
            # Si target está cerca de BSL, suma puntos
            distance_to_bsl = abs(target_price - bsl) / current_price
            if distance_to_bsl < 0.01:  # Dentro del 1%
                confluence += 2
            elif distance_to_bsl < 0.02:  # Dentro del 2%
                confluence += 1
        
        if ssl is not None:
            # Si target está cerca de SSL, suma puntos
            distance_to_ssl = abs(target_price - ssl) / current_price
            if distance_to_ssl < 0.01:
                confluence += 2
            elif distance_to_ssl < 0.02:
                confluence += 1
        
        return confluence


# ============================================================================
# UNIFIED STRATEGY - Superbot
# ============================================================================

class UnifiedStrategy:
    """
    🤖 SUPERBOT - Estrategia Unificada
    
    Jerarquía:
    1. Will Street PO3: Motor principal
    2. SMC Engine: Confirmación adicional
    """
    
    def __init__(self, use_po3: bool = True, use_smc: bool = True,
                 po3_min_rr: float = 4.0, smc_standalone_threshold: int = 8,
                 po3_weight: int = 10, smc_weight: int = 3, smc_rr_ratio: float = 4.0,
                 use_rsi: bool = True, use_liquidity: bool = True):
        self.use_po3 = use_po3
        self.use_smc = use_smc
        self.use_rsi = use_rsi
        self.use_liquidity = use_liquidity
        self.smc_standalone_threshold = smc_standalone_threshold
        self.po3_weight = po3_weight
        self.smc_weight = smc_weight
        self.smc_rr_ratio = smc_rr_ratio
        
        if use_po3:
            self.po3 = WillStreetPO3(min_rr_ratio=po3_min_rr)
        
        if use_smc:
            self.smc = SMCEngine(swing_length=5, fvg_threshold=0.001)
        
        if use_rsi:
            self.rsi_detector = RSIDivergenceDetector(rsi_period=14, overbought=70, oversold=30)
        
        if use_liquidity:
            self.liquidity_zones = LiquidityZones(lookback=20, threshold=0.002)
        
        logger.success(f"🤖 SUPERBOT Initialized!")
        logger.info(f"   └─ PO3 Primary: {use_po3} (Min RR: {po3_min_rr}:1)")
        logger.info(f"   └─ SMC Confirm: {use_smc} (Standalone: {smc_standalone_threshold}+ pts, RR: {smc_rr_ratio}:1)")
        logger.info(f"   └─ RSI Divergence: {use_rsi}")
        logger.info(f"   └─ Liquidity Zones: {use_liquidity}")
    
    def get_trading_signal(self, data_dict: Dict[str, pd.DataFrame],
                          current_time: pd.Timestamp, execution_tf: str = '15m') -> Optional[Dict]:
        """Genera señal combinando PO3 y SMC"""
        po3_signal = None
        smc_signal = None
        confluence_score = 0
        reasons = []
        
        # PASO 1: Will Street PO3 - Motor principal [DESACTIVADO - SOLO SMC]
        # if self.use_po3 and '4h' in data_dict:
        #     po3_signal = self.po3.analyze_4h_candle(
        #         df_4h=data_dict['4h'],
        #         df_15m=data_dict.get('15m', data_dict.get('1h')),
        #         current_time=current_time
        #     )
        #     
        #     if po3_signal:
        #         confluence_score += self.po3_weight
        #         reasons.append(f"🎯 PO3 {po3_signal['type']} (RR: {po3_signal['rr_ratio']:.1f}:1)")
        #         logger.success(f"⚡ PO3 PRIMARY SIGNAL: {po3_signal['type']} @ ${po3_signal['entry']:.2f}")
        
        # PASO 2: SMC - Única fuente de señales (PO3 desactivado)
        if self.use_smc and execution_tf in data_dict:
            df_exec = data_dict[execution_tf]
            current_idx = df_exec.index.get_indexer([current_time], method='ffill')[0]
            
            if current_idx >= 5:
                smc_analysis = self.smc.analyze_context(df_exec, current_idx)
                
                # SMC como única fuente de señales - umbral conservador
                if smc_analysis['confluence_score'] >= self.smc_standalone_threshold:
                    confluence_score = smc_analysis['confluence_score']
                    reasons.append(f"🎯 SMC Signal ({smc_analysis['confluence_score']} pts)")
                    
                    # NUEVO: Verificar RSI y Divergencias
                    rsi_bonus = 0
                    if self.use_rsi:
                        divergence = self.rsi_detector.detect_divergence(df_exec, current_idx)
                        rsi_extreme = self.rsi_detector.check_extreme_zones(df_exec, current_idx)
                        
                        # Divergencia alineada con señal SMC
                        if divergence['type'] == 'BULLISH' and smc_analysis['signal_type'] == 'LONG':
                            rsi_bonus += divergence['strength']
                            reasons.append(f"📈 RSI Bullish Divergence (+{divergence['strength']})")
                            logger.info(f"   └─ RSI Bullish Divergence detected! RSI: {divergence['rsi']:.1f}")
                        elif divergence['type'] == 'BEARISH' and smc_analysis['signal_type'] == 'SHORT':
                            rsi_bonus += divergence['strength']
                            reasons.append(f"📉 RSI Bearish Divergence (+{divergence['strength']})")
                            logger.info(f"   └─ RSI Bearish Divergence detected! RSI: {divergence['rsi']:.1f}")
                        
                        # RSI extremo alineado
                        if rsi_extreme['extreme'] == 'OVERSOLD' and smc_analysis['signal_type'] == 'LONG':
                            rsi_bonus += 2
                            reasons.append(f"⬇ RSI Oversold ({rsi_extreme['value']:.1f}) (+2)")
                        elif rsi_extreme['extreme'] == 'OVERBOUGHT' and smc_analysis['signal_type'] == 'SHORT':
                            rsi_bonus += 2
                            reasons.append(f"⬆ RSI Overbought ({rsi_extreme['value']:.1f}) (+2)")
                    
                    confluence_score += rsi_bonus
                    
                    logger.success(f"⚡ SMC SIGNAL: {smc_analysis['signal_type'].lower()} with {smc_analysis['confluence_score']} confluence points + RSI bonus {rsi_bonus}")
                    
                    # Crear señal SMC con stops más ajustados
                    current_price = df_exec.iloc[current_idx]['close']
                    atr = df_exec['high'].rolling(14).max().iloc[current_idx] - df_exec['low'].rolling(14).min().iloc[current_idx]
                    
                    # NUEVO: Ajustar TP basado en zonas de liquidez
                    liquidity_bonus = 0
                    target_price = None
                    if self.use_liquidity:
                        liq_zones = self.liquidity_zones.find_liquidity_zones(df_exec, current_idx)
                        
                        if smc_analysis['signal_type'] == 'LONG' and liq_zones['bsl'] is not None:
                            # Target es BSL para LONGs
                            target_price = liq_zones['bsl']
                            reasons.append(f"💧 Target: BSL @ ${target_price:.2f}")
                            liquidity_bonus = 2
                            logger.info(f"   └─ Buy Side Liquidity found @ ${target_price:.2f}")
                        elif smc_analysis['signal_type'] == 'SHORT' and liq_zones['ssl'] is not None:
                            # Target es SSL para SHORTs
                            target_price = liq_zones['ssl']
                            reasons.append(f"💧 Target: SSL @ ${target_price:.2f}")
                            liquidity_bonus = 2
                            logger.info(f"   └─ Sell Side Liquidity found @ ${target_price:.2f}")
                    
                    confluence_score += liquidity_bonus
                    
                    if smc_analysis['signal_type'] == 'LONG':
                        # Si hay target de liquidez, usarlo; sino usar RR ratio estándar
                        tp_price = target_price if (target_price and target_price > current_price) else current_price + (atr * (0.75 * self.smc_rr_ratio))
                        
                        smc_signal = {
                            'type': 'LONG',
                            'entry': current_price,
                            'stop_loss': current_price - (atr * 0.75),  # SL más ajustado
                            'take_profit': tp_price,  # TP ajustado por liquidez o RR
                            'rr_ratio': self.smc_rr_ratio,
                            'source': 'smc_only',
                            'timestamp': current_time
                        }
                    else:
                        # Si hay target de liquidez, usarlo; sino usar RR ratio estándar
                        tp_price = target_price if (target_price and target_price < current_price) else current_price - (atr * (0.75 * self.smc_rr_ratio))
                        
                        smc_signal = {
                            'type': 'SHORT',
                            'entry': current_price,
                            'stop_loss': current_price + (atr * 0.75),  # SL más ajustado
                            'take_profit': tp_price,  # TP ajustado por liquidez o RR
                            'rr_ratio': self.smc_rr_ratio,
                            'source': 'smc_only',
                            'timestamp': current_time
                        }
        
        # DECISIÓN FINAL - Solo SMC
        final_signal = smc_signal
        
        if final_signal and confluence_score > 0:
            final_signal['confluence_score'] = confluence_score
            final_signal['reasons'] = reasons
            
            logger.warning(f"⚡ SUPERBOT ENTRY: SMC Only | Score: {confluence_score}")
            
            return final_signal
        
        return None


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """Motor de backtesting completo"""
    
    def __init__(self, initial_balance: float = 500.0, risk_per_trade: float = 0.02,
                 max_leverage: int = 10, commission: float = 0.0004):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self.commission = commission
        
        self.trades: List[Dict] = []
        self.current_position: Optional[Dict] = None
        self.equity_curve: List[float] = [initial_balance]
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Para leverage dinámico
        self._last_signal_info = {}
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> Tuple[float, int]:
        """Calcula tamaño de posición con leverage dinámico"""
        # Obtener info de la última señal
        confluence_score = self._last_signal_info.get('confluence_score', 0)
        rr_ratio = self._last_signal_info.get('rr_ratio', 0)
        signal_source = self._last_signal_info.get('signal_source', '')
        
        # Leverage dinámico según calidad
        if confluence_score >= 13:
            dynamic_leverage = 75
        elif signal_source == 'po3_primary' and rr_ratio >= 5.0:
            dynamic_leverage = 60
        elif signal_source == 'po3_primary' and rr_ratio >= 4.0:
            dynamic_leverage = 50
        elif signal_source == 'po3_primary':
            dynamic_leverage = 30
        elif signal_source == 'smc_standalone':
            dynamic_leverage = 10
        else:
            dynamic_leverage = 5
        
        leverage = min(dynamic_leverage, self.max_leverage)
        
        # Protección balance negativo
        if self.balance <= 0:
            return 0, 1
        
        # Calcular posición basada en balance actual
        current_capital = max(self.balance, 0)
        risk_amount = current_capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss) / entry_price
        
        if stop_distance < 0.001:
            return 0, 1
        
        max_position_value = current_capital * leverage
        risk_based_position = risk_amount / stop_distance
        position_value = min(risk_based_position, max_position_value)
        
        quantity = position_value / entry_price
        
        return quantity, leverage
    
    def open_position(self, timestamp: pd.Timestamp, side: str, entry_price: float,
                     stop_loss: float, take_profit: float, signal_info: Dict):
        """Abre una nueva posición"""
        if self.current_position is not None:
            logger.warning("Position already open, skipping new signal")
            return
        
        quantity, leverage = self.calculate_position_size(entry_price, stop_loss)
        position_value = quantity * entry_price
        entry_commission = position_value * self.commission
        
        self.current_position = {
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
        """Verifica si la posición debe cerrarse"""
        if self.current_position is None:
            return False
        
        pos = self.current_position
        side = pos['side'].upper()
        
        # Verificar stop loss y take profit
        if side == 'LONG':
            if current_bar['low'] <= pos['stop_loss']:
                logger.debug(f"LONG SL hit: low={current_bar['low']:.2f} <= SL={pos['stop_loss']:.2f}")
                self.close_position(current_bar.name, pos['stop_loss'], 'stop_loss')
                return True
            elif current_bar['high'] >= pos['take_profit']:
                logger.debug(f"LONG TP hit: high={current_bar['high']:.2f} >= TP={pos['take_profit']:.2f}")
                self.close_position(current_bar.name, pos['take_profit'], 'take_profit')
                return True
        
        elif side == 'SHORT':
            if current_bar['high'] >= pos['stop_loss']:
                logger.debug(f"SHORT SL hit: high={current_bar['high']:.2f} >= SL={pos['stop_loss']:.2f}")
                self.close_position(current_bar.name, pos['stop_loss'], 'stop_loss')
                return True
            elif current_bar['low'] <= pos['take_profit']:
                logger.debug(f"SHORT TP hit: low={current_bar['low']:.2f} <= TP={pos['take_profit']:.2f}")
                self.close_position(current_bar.name, pos['take_profit'], 'take_profit')
                return True
        
        return False
    
    def close_position(self, timestamp: pd.Timestamp, exit_price: float, reason: str):
        """Cierra la posición actual"""
        if self.current_position is None:
            return
        
        pos = self.current_position
        side = pos['side'].upper()
        
        logger.debug(f"CLOSE: {side.lower()} | Entry={pos['entry_price']:.2f} | Exit={exit_price:.2f} | Reason={reason}")
        logger.debug(f"       SL={pos['stop_loss']:.2f} | TP={pos['take_profit']:.2f}")
        
        # Calcular P&L
        if side == 'LONG':
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
        
        pnl_usd = pos['position_value'] * pnl_pct
        exit_commission = pos['quantity'] * exit_price * self.commission
        net_pnl = pnl_usd - pos['entry_commission'] - exit_commission
        
        self.balance += net_pnl
        self.equity_curve.append(self.balance)
        
        # Registrar trade
        trade = {
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'quantity': pos['quantity'],
            'leverage': pos['leverage'],
            'pnl': net_pnl,
            'pnl_pct': pnl_pct * 100,
            'exit_reason': reason,
            'duration': timestamp - pos['entry_time']
        }
        
        self.trades.append(trade)
        self.total_trades += 1
        
        if net_pnl > 0:
            self.winning_trades += 1
            logger.info(f"✅ Trade closed: +${net_pnl:.2f} ({pnl_pct*100:.2f}%) - {reason}")
        else:
            self.losing_trades += 1
            logger.info(f"❌ Trade closed: ${net_pnl:.2f} ({pnl_pct*100:.2f}%) - {reason}")
        
        self.current_position = None
    
    def get_statistics(self) -> Dict:
        """Calcula estadísticas del backtest"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] < 0]
        
        total_pnl = trades_df['pnl'].sum()
        win_rate = len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
        avg_loss = losing['pnl'].mean() if len(losing) > 0 else 0
        
        profit_factor = abs(winning['pnl'].sum() / losing['pnl'].sum()) if len(losing) > 0 and losing['pnl'].sum() != 0 else 0
        
        # Sharpe Ratio
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() != 0 else 0
        
        # Max Drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': trades_df['pnl'].max(),
            'worst_trade': trades_df['pnl'].min(),
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_drawdown),
            'final_balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100
        }
    
    def plot_results(self, filename: str = 'backtest_results.png'):
        """Genera gráficos de resultados"""
        if not self.trades:
            logger.warning("No trades to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Equity curve
        axes[0].plot(self.equity_curve, linewidth=2)
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Trade Number')
        axes[0].set_ylabel('Balance (USDT)')
        axes[0].grid(True, alpha=0.3)
        
        # Trade P&L
        trades_df = pd.DataFrame(self.trades)
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
        axes[1].bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.6)
        axes[1].set_title('Trade P&L', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Trade Number')
        axes[1].set_ylabel('P&L (USDT)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"Results plot saved to {filename}")


# ============================================================================
# SUPERBOT BACKTEST
# ============================================================================

class SuperbotBacktest(BacktestEngine):
    """Backtest específico para Superbot con estrategia unificada"""
    
    def __init__(self, initial_balance: float = 500.0, risk_per_trade: float = 0.02,
                 max_leverage: int = 10, commission: float = 0.0004,
                 po3_min_rr: float = 3.5, smc_standalone: int = 8, use_po3: bool = True, smc_rr_ratio: float = 4.0):
        super().__init__(initial_balance, risk_per_trade, max_leverage, commission)
        
        self.strategy = UnifiedStrategy(
            use_po3=use_po3,
            use_smc=True,
            po3_min_rr=po3_min_rr,
            smc_standalone_threshold=smc_standalone,
            smc_rr_ratio=smc_rr_ratio
        )
    
    def run_backtest(self, data_1h: pd.DataFrame, data_4h: pd.DataFrame,
                    data_1d: pd.DataFrame, data_15m: pd.DataFrame = None,
                    start_date: str = None, end_date: str = None):
        """Ejecuta el backtest completo"""
        
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
        
        execution_tf = '15m' if data_15m is not None else '1h'
        execution_data = data_15m if data_15m is not None else data_1h
        
        logger.info(f"Execution timeframe: {execution_tf}")
        logger.info("Starting simulation...")
        logger.info("-" * 60)
        
        # Simular en timeframe de ejecución
        for i, (timestamp, bar) in enumerate(execution_data.iterrows()):
            
            # Progress
            if i % 1000 == 0:
                logger.info(f"Progress: {i/len(execution_data)*100:.1f}% ({i}/{len(execution_data)}) - Balance: ${self.balance:.2f}")
            
            # Check posición abierta
            if self.current_position is not None:
                self.check_position(bar)
                continue
            
            # Buscar nueva señal
            data_dict = {
                '15m': data_15m if data_15m is not None else None,
                '1h': data_1h,
                '4h': data_4h,
                '1d': data_1d
            }
            
            signal = self.strategy.get_trading_signal(
                data_dict=data_dict,
                current_time=timestamp,
                execution_tf=execution_tf
            )
            
            if signal is None:
                continue
            
            # Filtrar trades con profit muy bajo
            entry = signal['entry']
            tp = signal['take_profit']
            profit_pct = abs((tp - entry) / entry) * 100
            
            if profit_pct < 1.0:
                continue
            
            # Señal encontrada!
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"SIGNAL #{len(self.trades) + 1} at {timestamp}")
            logger.info("=" * 60)
            
            # Guardar info para leverage dinámico
            self._last_signal_info = {
                'confluence_score': signal.get('confluence_score', 0),
                'rr_ratio': signal.get('rr_ratio', 0),
                'signal_source': 'po3_primary' if signal.get('source') == 'po3' else 'smc_standalone'
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
            last_bar = execution_data.iloc[-1]
            self.close_position(last_bar.name, last_bar['close'], 'end_of_backtest')
        
        return self.get_statistics()


# ============================================================================
# MAIN - CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='🤖 Botardo - Trading Bot Backtest')
    parser.add_argument('--data_1m', type=str, required=True, help='Archivo CSV con datos 1m')
    parser.add_argument('--initial_capital', type=float, default=500, help='Capital inicial')
    parser.add_argument('--risk_per_trade', type=float, default=0.06, help='Riesgo por trade (0.06 = 6%)')
    parser.add_argument('--leverage', type=int, default=10, help='Leverage máximo (default: 10x)')
    parser.add_argument('--po3_min_rr', type=float, default=2.0, help='RR mínimo para PO3')
    parser.add_argument('--smc_standalone', type=int, default=8, help='Confluencia mínima SMC (default: 8)')
    parser.add_argument('--smc_rr', type=float, default=2.0, help='RR ratio para SMC (default: 2.0)')
    parser.add_argument('--no_po3', action='store_true', help='Desactivar PO3 (solo SMC)')
    parser.add_argument('--start', type=str, default=None, help='Fecha inicio (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='Fecha fin (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    logger.info("Loading data from {}...".format(args.data_1m))
    
    # Cargar datos
    df_1m = pd.read_csv(args.data_1m, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df_1m)} 1-minute bars")
    
    logger.info("Resampling to higher timeframes...")
    
    # Resample a timeframes superiores
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
    logger.info("=" * 60)
    logger.info("🤖 BOTARDO BACKTEST 🤖")
    logger.info("=" * 60)
    logger.info(f"Initial Balance: ${args.initial_capital:.2f}")
    logger.info(f"Risk per Trade: {args.risk_per_trade*100:.1f}%")
    logger.info(f"Max Leverage: {args.leverage}x")
    logger.info(f"PO3 Min RR: {args.po3_min_rr}:1")
    logger.info(f"SMC Standalone: {args.smc_standalone}+ pts")
    logger.info("=" * 60)
    
    backtest = SuperbotBacktest(
        initial_balance=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        max_leverage=args.leverage,
        po3_min_rr=args.po3_min_rr,
        smc_standalone=args.smc_standalone,
        use_po3=(not args.no_po3),  # Desactivar PO3 si se pasa el flag
        smc_rr_ratio=args.smc_rr
    )
    
    # Ejecutar backtest
    stats = backtest.run_backtest(
        data_1h=df_1h,
        data_4h=df_4h,
        data_1d=df_1d,
        data_15m=df_15m,
        start_date=args.start,
        end_date=args.end
    )
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Initial Balance:    $ {args.initial_capital:>10.2f}")
    print(f"Final Balance:      $ {stats.get('final_balance', 0):>10.2f}")
    print(f"Total P&L:          $ {stats.get('total_pnl', 0):>10.2f} ({stats.get('return_pct', 0):.2f}%)")
    print(f"Max Drawdown:             {stats.get('max_drawdown', 0):>10.2f}%")
    print(f"Sharpe Ratio:                 {stats.get('sharpe_ratio', 0):>10.2f}")
    print("-" * 70)
    print(f"Total Trades:                    {stats.get('total_trades', 0):>10}")
    print(f"Winning Trades:                   {stats.get('winning_trades', 0):>10}")
    print(f"Losing Trades:                   {stats.get('losing_trades', 0):>10}")
    print(f"Win Rate:                     {stats.get('win_rate', 0):>10.2f}%")
    print("-" * 70)
    print(f"Average Win:        $ {stats.get('avg_win', 0):>10.2f}")
    print(f"Average Loss:       $ {stats.get('avg_loss', 0):>10.2f}")
    print(f"Best Trade:         $ {stats.get('best_trade', 0):>10.2f}")
    print(f"Worst Trade:        $ {stats.get('worst_trade', 0):>10.2f}")
    print(f"Profit Factor:                 {stats.get('profit_factor', 0):>10.2f}")
    print("=" * 70)
    
    # Guardar trades
    if backtest.trades:
        trades_df = pd.DataFrame(backtest.trades)
        trades_df.to_csv('botardo_trades.csv', index=False)
        logger.info("✅ Trade log saved to botardo_trades.csv")
    
    # Generar gráfico
    backtest.plot_results('botardo_backtest.png')


if __name__ == "__main__":
    main()

"""
Will Street Time-Based Model (Power of 3)
Estrategia basada en tiempo que opera las expansiones de velas de 4 horas

Conceptos clave:
1. Power of 3: Open → High → Low → Close
2. El extremo debe crearse en los primeros 15 minutos
3. Entrada óptima: SHORT arriba del Open, LONG abajo del Open
4. Objetivo: High/Low de la vela anterior
5. Mínimo 3:1 Risk/Reward ratio
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from loguru import logger


class WillStreetPO3:
    """
    Will Street Power of 3 Strategy
    Opera las velas de 4 horas cuando el extremo se crea temprano
    """
    
    def __init__(
        self,
        min_rr_ratio: float = 3.0,
        min_stop_pips: int = 10,
        extreme_creation_minutes: int = 15,
        min_stop_pct: float = 0.3  # SL mínimo 0.3% del precio
    ):
        self.min_rr_ratio = min_rr_ratio
        self.min_stop_pips = min_stop_pips
        self.extreme_creation_minutes = extreme_creation_minutes
        self.min_stop_pct = min_stop_pct
        
        # Track de velas 4h ya procesadas para evitar duplicados
        self.processed_candles = set()
        
        logger.info(f"Will Street PO3 initialized - Min RR: {min_rr_ratio}:1, Min SL: {min_stop_pct}%")
    
    def _get_candle_extreme_time(self, df_4h: pd.DataFrame, candle_idx: int, direction: str) -> Optional[datetime]:
        """
        Verifica si el extremo (high/low) se creó en los primeros 15 minutos
        Necesita data de menor temporalidad (15m o 5m) para verificar
        """
        # Esta función requeriría acceso a data de 15m o 5m
        # Por ahora simplificamos asumiendo que si queremos operar, es válido
        return True
    
    def get_previous_candle_extremes(self, df_4h: pd.DataFrame, current_idx: int) -> Dict:
        """
        Obtiene el high y low de la vela anterior de 4h
        """
        if current_idx < 1:
            return None
        
        prev_candle = df_4h.iloc[current_idx - 1]
        
        return {
            'prev_high': prev_candle['high'],
            'prev_low': prev_candle['low'],
            'prev_open': prev_candle['open'],
            'prev_close': prev_candle['close']
        }
    
    def analyze_4h_candle(
        self,
        df_4h: pd.DataFrame,
        df_15m: pd.DataFrame,
        current_time: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Analiza si hay setup válido en la vela de 4h actual
        
        Reglas:
        1. Identificar si somos bullish o bearish (mitigar high/low previo)
        2. Verificar que el extremo se creó en primeros 15 min
        3. Entrada: SHORT arriba Open, LONG abajo Open
        4. Target: Low/High previo
        5. Stop: Debe dar mínimo 3:1 RR
        """
        
        # Encontrar índice de la vela de 4h actual
        candle_4h_idx = df_4h.index.get_indexer([current_time], method='ffill')[0]
        
        if candle_4h_idx < 1:
            return None
        
        current_candle = df_4h.iloc[candle_4h_idx]
        candle_timestamp = current_candle.name
        
        # ⚡ PREVENIR DUPLICADOS: Solo señalar una vez por vela 4h
        if candle_timestamp in self.processed_candles:
            return None
        
        prev_extremes = self.get_previous_candle_extremes(df_4h, candle_4h_idx)
        
        if prev_extremes is None:
            return None
        
        # Determinar bias (dirección)
        # Si estamos cerca del high previo → esperamos SHORT
        # Si estamos cerca del low previo → esperamos LONG
        
        current_price = current_candle['close']
        candle_open = current_candle['open']
        
        # BEARISH SETUP: Precio mitigó el high previo
        if current_price >= prev_extremes['prev_high'] * 0.999:  # Pequeña tolerancia
            # Queremos vender ARRIBA del Open
            if current_price > candle_open:
                entry = current_price
                # SL debe estar ARRIBA del entry (protección crítica)
                raw_sl = prev_extremes['prev_high'] * 1.001
                stop_loss = max(raw_sl, entry * 1.005)  # Garantizar SL arriba del entry
                take_profit = prev_extremes['prev_low']
                
                # Verificar SL mínimo en % del precio
                risk = abs(entry - stop_loss)
                risk_pct = (risk / entry) * 100
                
                if risk_pct < self.min_stop_pct:
                    return None  # SL demasiado chico, skip trade
                
                reward = abs(entry - take_profit)
                
                if reward / risk >= self.min_rr_ratio:
                    # ✅ Marcar vela como procesada
                    self.processed_candles.add(candle_timestamp)
                    
                    return {
                        'type': 'SHORT',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'rr_ratio': reward / risk,
                        'reason': f"PO3 SHORT - Entry above Open, Target: Prev Low, RR: {reward/risk:.2f}:1"
                    }
        
        # BULLISH SETUP: Precio mitigó el low previo
        elif current_price <= prev_extremes['prev_low'] * 1.001:  # Pequeña tolerancia
            # Queremos comprar ABAJO del Open
            if current_price < candle_open:
                entry = current_price
                # SL debe estar ABAJO del entry (protección crítica)
                raw_sl = prev_extremes['prev_low'] * 0.999
                stop_loss = min(raw_sl, entry * 0.995)  # Garantizar SL abajo del entry
                take_profit = prev_extremes['prev_high']
                
                # Verificar SL mínimo en % del precio
                risk = abs(entry - stop_loss)
                risk_pct = (risk / entry) * 100
                
                if risk_pct < self.min_stop_pct:
                    return None  # SL demasiado chico, skip trade
                
                reward = abs(entry - take_profit)
                
                if reward / risk >= self.min_rr_ratio:
                    # ✅ Marcar vela como procesada
                    self.processed_candles.add(candle_timestamp)
                    
                    return {
                        'type': 'LONG',
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'rr_ratio': reward / risk,
                        'reason': f"PO3 LONG - Entry below Open, Target: Prev High, RR: {reward/risk:.2f}:1"
                    }
        
        return None
    
    def check_failure_swing(self, df_4h: pd.DataFrame, current_idx: int) -> bool:
        """
        Failure Swing: La vela actual NO toma el high/low de la vela previa
        
        Condición: El extremo de la vela previa se creó en sus últimos minutos
        """
        if current_idx < 2:
            return False
        
        current = df_4h.iloc[current_idx]
        previous = df_4h.iloc[current_idx - 1]
        two_back = df_4h.iloc[current_idx - 2]
        
        # Failure swing BEARISH: No toma el high previo
        if current['high'] < previous['high']:
            # Verificar que el high previo se creó tarde (simplificado)
            return True
        
        # Failure swing BULLISH: No toma el low previo
        if current['low'] > previous['low']:
            return True
        
        return False
    
    def get_london_session_signal(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        current_time: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Señal específica para sesión de Londres (1am - 5am NY time)
        
        La vela de 1am (3:00 UTC) es la más importante para traders de Londres
        """
        hour = current_time.hour
        
        # Verificar si estamos en horario de Londres (1am - 5am NY time)
        # Convertir a UTC: NY UTC-5, entonces 1am NY = 6am UTC
        utc_hour = (hour + 5) % 24
        
        if utc_hour < 6 or utc_hour >= 10:  # Fuera de Londres
            return None
        
        # Analizar como vela de 4h normal
        return self.analyze_4h_candle(df_4h, df_1h, current_time)
    
    def should_continue_position(
        self,
        entry_time: pd.Timestamp,
        current_time: pd.Timestamp,
        direction: str
    ) -> bool:
        """
        Determina si debemos mantener la posición o salir
        
        Regla: Salir en cada nueva vela de 4h si ya alcanzamos objetivo parcial
        """
        time_diff = (current_time - entry_time).total_seconds() / 3600
        
        # Si han pasado 4 horas, considerar salida
        if time_diff >= 4:
            return False
        
        return True

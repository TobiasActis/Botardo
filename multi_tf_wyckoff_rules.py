"""
Multi-Timeframe Wyckoff Analysis Engine
Implementa la metodología Wyckoff para análisis de mercado en múltiples temporalidades
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from loguru import logger


class WyckoffPhase(Enum):
    """Fases del ciclo Wyckoff"""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class WyckoffEvent(Enum):
    """Eventos clave Wyckoff"""
    PS = "preliminary_support"  # Soporte Preliminar
    SC = "selling_climax"  # Clímax de Venta
    AR = "automatic_rally"  # Rally Automático
    ST = "secondary_test"  # Test Secundario
    SPRING = "spring"  # Spring
    SOS = "sign_of_strength"  # Señal de Fuerza
    LPS = "last_point_support"  # Último Punto de Soporte
    
    PSY = "preliminary_supply"  # Oferta Preliminar
    BC = "buying_climax"  # Clímax de Compra
    AR_DIST = "automatic_reaction"  # Reacción Automática
    ST_DIST = "secondary_test_dist"  # Test Secundario Distribución
    UTAD = "upthrust_after_dist"  # Upthrust
    SOW = "sign_of_weakness"  # Señal de Debilidad
    LPSY = "last_point_supply"  # Último Punto de Oferta


class MultiTimeframeWyckoff:
    """
    Analizador Wyckoff Multi-Timeframe
    
    Analiza múltiples temporalidades para identificar:
    - Fases del ciclo Wyckoff
    - Eventos clave
    - Confluencias entre timeframes
    - Señales de entrada/salida
    """
    
    def __init__(self, timeframes: List[str] = None):
        """
        Args:
            timeframes: Lista de timeframes a analizar (ej: ['1h', '4h', '1d'])
        """
        self.timeframes = timeframes or ['1h', '4h', '1d']
        self.data: Dict[str, pd.DataFrame] = {}
        self.analysis: Dict[str, Dict] = {}
        
    def load_data(self, timeframe: str, data: pd.DataFrame):
        """Carga datos OHLCV para un timeframe específico"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        self.data[timeframe] = data.copy()
        logger.info(f"Loaded {len(data)} bars for {timeframe}")
        
    def analyze_all_timeframes(self) -> Dict[str, Dict]:
        """Analiza todos los timeframes cargados"""
        for tf in self.timeframes:
            if tf not in self.data:
                logger.warning(f"No data for timeframe {tf}")
                continue
                
            self.analysis[tf] = self._analyze_single_timeframe(tf)
            
        return self.analysis
    
    def _analyze_single_timeframe(self, timeframe: str) -> Dict:
        """Analiza un timeframe individual"""
        df = self.data[timeframe].copy()
        
        # Calcular indicadores técnicos
        df = self._calculate_indicators(df)
        
        # Identificar fase actual
        current_phase = self._identify_phase(df)
        
        # Detectar eventos Wyckoff
        events = self._detect_events(df, current_phase)
        
        # Calcular niveles clave
        levels = self._calculate_key_levels(df, current_phase)
        
        return {
            'phase': current_phase,
            'events': events,
            'levels': levels,
            'dataframe': df
        }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos necesarios"""
        # Volumen promedio
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Rango de precio
        df['range'] = df['high'] - df['low']
        df['range_ma'] = df['range'].rolling(window=20).mean()
        
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Effort vs Result
        df['effort'] = df['volume_ratio']
        df['result'] = abs(df['close'] - df['open']) / df['range'].replace(0, 0.0001)
        
        # Soporte/Resistencia dinámicos
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        
        return df
    
    def _identify_phase(self, df: pd.DataFrame) -> WyckoffPhase:
        """Identifica la fase actual del ciclo Wyckoff"""
        if len(df) < 50:
            return WyckoffPhase.UNKNOWN
        
        recent = df.tail(30)
        
        # Calcular características de la fase
        price_range = (recent['high'].max() - recent['low'].min()) / recent['close'].mean()
        volume_trend = recent['volume'].tail(10).mean() / recent['volume'].head(10).mean()
        price_trend = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        # Lógica de identificación de fase
        if price_range < 0.05 and volume_trend > 1.2:
            if recent['close'].iloc[-1] < recent['close'].mean():
                return WyckoffPhase.ACCUMULATION
            else:
                return WyckoffPhase.DISTRIBUTION
        elif price_trend > 0.1 and volume_trend > 1.0:
            return WyckoffPhase.MARKUP
        elif price_trend < -0.1 and volume_trend > 1.0:
            return WyckoffPhase.MARKDOWN
        
        return WyckoffPhase.UNKNOWN
    
    def _detect_events(self, df: pd.DataFrame, phase: WyckoffPhase) -> List[Dict]:
        """Detecta eventos Wyckoff en el dataframe"""
        events = []
        
        if phase == WyckoffPhase.ACCUMULATION:
            events.extend(self._detect_accumulation_events(df))
        elif phase == WyckoffPhase.DISTRIBUTION:
            events.extend(self._detect_distribution_events(df))
        
        return events
    
    def _detect_accumulation_events(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta eventos de fase de acumulación"""
        events = []
        recent = df.tail(50)
        
        # Selling Climax (SC): Alto volumen + caída fuerte
        for i in range(len(recent) - 5, len(recent)):
            if i < 5:
                continue
            row = recent.iloc[i]
            if (row['volume_ratio'] > 2.0 and 
                row['close'] < row['open'] and
                row['range'] > row['range_ma'] * 1.5):
                events.append({
                    'event': WyckoffEvent.SC,
                    'timestamp': row.name,
                    'price': row['close'],
                    'confidence': min(row['volume_ratio'] / 3.0, 1.0)
                })
        
        # Spring: Penetración breve por debajo del soporte con volumen bajo
        support = recent['low'].min()
        for i in range(len(recent) - 10, len(recent)):
            if i < 5:
                continue
            row = recent.iloc[i]
            if (row['low'] < support * 0.995 and 
                row['close'] > support and
                row['volume_ratio'] < 1.2):
                events.append({
                    'event': WyckoffEvent.SPRING,
                    'timestamp': row.name,
                    'price': row['low'],
                    'confidence': 0.8
                })
        
        return events
    
    def _detect_distribution_events(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta eventos de fase de distribución"""
        events = []
        recent = df.tail(50)
        
        # Buying Climax (BC): Alto volumen + subida fuerte
        for i in range(len(recent) - 5, len(recent)):
            if i < 5:
                continue
            row = recent.iloc[i]
            if (row['volume_ratio'] > 2.0 and 
                row['close'] > row['open'] and
                row['range'] > row['range_ma'] * 1.5):
                events.append({
                    'event': WyckoffEvent.BC,
                    'timestamp': row.name,
                    'price': row['close'],
                    'confidence': min(row['volume_ratio'] / 3.0, 1.0)
                })
        
        # UTAD: Penetración breve por encima de resistencia con volumen bajo
        resistance = recent['high'].max()
        for i in range(len(recent) - 10, len(recent)):
            if i < 5:
                continue
            row = recent.iloc[i]
            if (row['high'] > resistance * 1.005 and 
                row['close'] < resistance and
                row['volume_ratio'] < 1.2):
                events.append({
                    'event': WyckoffEvent.UTAD,
                    'timestamp': row.name,
                    'price': row['high'],
                    'confidence': 0.8
                })
        
        return events
    
    def _calculate_key_levels(self, df: pd.DataFrame, phase: WyckoffPhase) -> Dict:
        """Calcula niveles clave de soporte/resistencia"""
        recent = df.tail(50)
        
        levels = {
            'support': recent['low'].min(),
            'resistance': recent['high'].max(),
            'midpoint': (recent['high'].max() + recent['low'].min()) / 2,
            'current': df['close'].iloc[-1]
        }
        
        # Niveles específicos según fase
        if phase == WyckoffPhase.ACCUMULATION:
            levels['creek'] = recent['low'].min()  # Creek (soporte principal)
            levels['spring_level'] = levels['creek'] * 0.99
        elif phase == WyckoffPhase.DISTRIBUTION:
            levels['ice'] = recent['high'].max()  # Ice (resistencia principal)
            levels['utad_level'] = levels['ice'] * 1.01
        
        return levels
    
    def get_trading_signal(self) -> Optional[Dict]:
        """
        Genera señal de trading basada en confluencia multi-timeframe
        
        Returns:
            Dict con señal de trading o None
        """
        if not self.analysis:
            logger.warning("No analysis available. Run analyze_all_timeframes() first.")
            return None
        
        # Verificar confluencia entre timeframes
        phases = [self.analysis[tf]['phase'] for tf in self.timeframes if tf in self.analysis]
        
        # Señal LONG: Acumulación en múltiples TFs
        accumulation_count = sum(1 for p in phases if p == WyckoffPhase.ACCUMULATION)
        if accumulation_count >= 2:
            # Buscar Spring en el menor timeframe
            shortest_tf = self.timeframes[0]
            events = self.analysis.get(shortest_tf, {}).get('events', [])
            spring_events = [e for e in events if e['event'] == WyckoffEvent.SPRING]
            
            if spring_events:
                latest_spring = max(spring_events, key=lambda x: x['timestamp'])
                return {
                    'type': 'LONG',
                    'confidence': latest_spring['confidence'],
                    'entry': latest_spring['price'] * 1.001,
                    'stop_loss': self.analysis[shortest_tf]['levels']['support'] * 0.995,
                    'take_profit': self.analysis[shortest_tf]['levels']['resistance'],
                    'reason': f"Spring detected with {accumulation_count} TF confluence"
                }
        
        # Señal SHORT: Distribución en múltiples TFs
        distribution_count = sum(1 for p in phases if p == WyckoffPhase.DISTRIBUTION)
        if distribution_count >= 2:
            shortest_tf = self.timeframes[0]
            events = self.analysis.get(shortest_tf, {}).get('events', [])
            utad_events = [e for e in events if e['event'] == WyckoffEvent.UTAD]
            
            if utad_events:
                latest_utad = max(utad_events, key=lambda x: x['timestamp'])
                return {
                    'type': 'SHORT',
                    'confidence': latest_utad['confidence'],
                    'entry': latest_utad['price'] * 0.999,
                    'stop_loss': self.analysis[shortest_tf]['levels']['resistance'] * 1.005,
                    'take_profit': self.analysis[shortest_tf]['levels']['support'],
                    'reason': f"UTAD detected with {distribution_count} TF confluence"
                }
        
        return None


if __name__ == "__main__":
    # Ejemplo de uso
    logger.info("Wyckoff Multi-Timeframe Analysis Engine initialized")
    
    # Crear instancia
    wyckoff = MultiTimeframeWyckoff(timeframes=['1h', '4h', '1d'])
    
    # Aquí cargarías datos reales con wyckoff.load_data()
    # wyckoff.load_data('1h', df_1h)
    # wyckoff.analyze_all_timeframes()
    # signal = wyckoff.get_trading_signal()
    
    logger.info("Ready to analyze market data")

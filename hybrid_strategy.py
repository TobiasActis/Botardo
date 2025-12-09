"""
Estrategia Híbrida: Wyckoff + Smart Money Concepts
Combina análisis de fases de Wyckoff con señales SMC precisas
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger

from multi_tf_wyckoff_rules import MultiTimeframeWyckoff, WyckoffPhase
from smc_engine import SMCEngine


class HybridStrategy:
    """
    Estrategia híbrida que combina:
    - Wyckoff: Para identificar fase del mercado (contexto macro)
    - SMC: Para entradas precisas (Order Blocks, FVG, Sweeps)
    """
    
    def __init__(
        self,
        # Parámetros Wyckoff
        wyckoff_timeframes: list = ['1h', '4h', '1d'],
        
        # Parámetros SMC
        smc_swing_length: int = 5,
        smc_fvg_threshold: float = 0.001,
        smc_ob_lookback: int = 20,
        
        # Requisitos de confluencia
        min_confluence_score: int = 5,
        require_wyckoff_alignment: bool = True
    ):
        """
        Args:
            wyckoff_timeframes: Timeframes para análisis Wyckoff
            smc_swing_length: Longitud de swings para SMC
            smc_fvg_threshold: Umbral mínimo para FVG
            smc_ob_lookback: Lookback para Order Blocks
            min_confluence_score: Score mínimo para generar señal
            require_wyckoff_alignment: Si requiere alineación con fase Wyckoff
        """
        self.wyckoff = MultiTimeframeWyckoff(timeframes=wyckoff_timeframes)
        self.smc = SMCEngine(
            swing_length=smc_swing_length,
            fvg_threshold=smc_fvg_threshold,
            ob_lookback=smc_ob_lookback
        )
        
        self.min_confluence_score = min_confluence_score
        self.require_wyckoff_alignment = require_wyckoff_alignment
        
        logger.info(f"Hybrid Strategy initialized (Wyckoff + SMC)")
    
    def get_trading_signal(
        self,
        data_dict: Dict[str, pd.DataFrame],
        current_time: pd.Timestamp,
        execution_tf: str = '15m'  # Timeframe de ejecución
    ) -> Optional[Dict]:
        """
        Genera señal de trading híbrida
        
        Args:
            data_dict: Dict con datos de múltiples timeframes {'1h': df, '4h': df, ...}
            current_time: Timestamp actual
            execution_tf: Timeframe donde se ejecutará el trade
            
        Returns:
            Señal de trading con confluencia Wyckoff + SMC
        """
        
        # 1. ANÁLISIS WYCKOFF (Opcional - solo si se requiere alineación)
        wyckoff_signal = None
        wyckoff_phases = {}
        
        if self.require_wyckoff_alignment:
            # Solo analizar Wyckoff si es estrictamente necesario
            for tf, df in data_dict.items():
                if tf in self.wyckoff.timeframes:
                    self.wyckoff.load_data(tf, df)
            
            self.wyckoff.analyze_all_timeframes()
            wyckoff_signal = self.wyckoff.get_trading_signal()
            
            if wyckoff_signal:
                logger.info(f"Wyckoff: {wyckoff_signal.get('type', 'N/A')} - {wyckoff_signal.get('reason', 'N/A')}")
            
            if self.wyckoff.analysis:
                for tf in ['1d', '4h', '1h']:
                    if tf in self.wyckoff.analysis:
                        wyckoff_phases[f'phase_{tf}'] = self.wyckoff.analysis[tf]['phase']
        
        # 2. ANÁLISIS SMC (Entrada precisa)
        # Usar el timeframe de ejecución para SMC
        if execution_tf not in data_dict:
            logger.warning(f"Execution timeframe {execution_tf} not available")
            return None
        
        execution_data = data_dict[execution_tf]
        
        # Asegurarse de que current_time existe en los datos
        if current_time not in execution_data.index:
            logger.warning(f"Current time {current_time} not in {execution_tf} data")
            return None
        
        smc_signal = self.smc.get_trading_signals(execution_data, current_time)
        
        # Log del análisis SMC
        if smc_signal:
            logger.info(f"SMC Analysis at {current_time}:")
            logger.info(f"  Signal: {smc_signal['signal'].upper()}")
            logger.info(f"  Confluence Score: {smc_signal['confluence_score']}")
            logger.info(f"  Reasons: {', '.join(smc_signal['reasons'])}")
            logger.info(f"  RSI: {smc_signal['rsi']:.2f}")
        
        # 3. COMBINAR SEÑALES
        
        # Si no hay señal SMC, no hay trade
        if not smc_signal:
            return None
        
        # Verificar confluencia mínima
        if smc_signal['confluence_score'] < self.min_confluence_score:
            logger.info(f"Insufficient confluence score: {smc_signal['confluence_score']} < {self.min_confluence_score}")
            return None
        
        # Si se requiere alineación con Wyckoff
        if self.require_wyckoff_alignment and wyckoff_signal:
            wyckoff_direction = wyckoff_signal.get('type', '').lower()  # 'LONG' o 'SHORT'
            smc_direction = smc_signal['signal']  # 'long' o 'short'
            
            # Verificar alineación
            if wyckoff_direction == 'long' and smc_direction != 'long':
                logger.info(f"Wyckoff says LONG but SMC says {smc_direction.upper()} - no alignment")
                return None
            elif wyckoff_direction == 'short' and smc_direction != 'short':
                logger.info(f"Wyckoff says SHORT but SMC says {smc_direction.upper()} - no alignment")
                return None
            
            # Aumentar confluence score si hay alineación perfecta
            if wyckoff_direction == smc_direction:
                smc_signal['confluence_score'] += 3
                smc_signal['reasons'].append(f"Wyckoff alignment ({wyckoff_direction.upper()})")
                logger.success(f"✅ Perfect alignment: Wyckoff + SMC both {smc_direction.upper()}")
        
        # 4. CONSTRUIR SEÑAL FINAL
        final_signal = {
            'signal': smc_signal['signal'],
            'entry_price': smc_signal['entry_price'],
            'stop_loss': smc_signal['stop_loss'],
            'take_profit': smc_signal['take_profit'],
            'confluence_score': smc_signal['confluence_score'],
            'reasons': smc_signal['reasons'],
            'rsi': smc_signal['rsi'],
            'timestamp': current_time,
            
            # Información adicional Wyckoff
            'wyckoff_signal': wyckoff_signal.get('type') if wyckoff_signal else None,
            'wyckoff_confidence': wyckoff_signal.get('confidence') if wyckoff_signal else None,
            **wyckoff_phases,  # Agregar fases dinámicamente
            'strategy': 'Hybrid_Wyckoff_SMC'
        }
        
        logger.success(f"🎯 TRADE SIGNAL GENERATED:")
        logger.success(f"   Direction: {final_signal['signal'].upper()}")
        logger.success(f"   Entry: ${final_signal['entry_price']:.2f}")
        logger.success(f"   Stop Loss: ${final_signal['stop_loss']:.2f}")
        logger.success(f"   Take Profit: ${final_signal['take_profit']:.2f}")
        logger.success(f"   Confluence: {final_signal['confluence_score']} points")
        
        return final_signal
    
    def update_wyckoff_params(self, **kwargs):
        """Actualiza parámetros de Wyckoff"""
        for key, value in kwargs.items():
            if hasattr(self.wyckoff, key):
                setattr(self.wyckoff, key, value)
                logger.info(f"Updated Wyckoff param: {key} = {value}")
    
    def update_smc_params(self, **kwargs):
        """Actualiza parámetros de SMC"""
        for key, value in kwargs.items():
            if hasattr(self.smc, key):
                setattr(self.smc, key, value)
                logger.info(f"Updated SMC param: {key} = {value}")


if __name__ == "__main__":
    logger.info("Hybrid Strategy module loaded")

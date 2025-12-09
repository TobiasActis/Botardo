"""
Estrategia Unificada: SMC + Wyckoff + Will Street PO3
Combina los mejores elementos de cada metodología
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger

from smc_engine import SMCEngine
from multi_tf_wyckoff_rules import MultiTimeframeWyckoff
from willstreet_po3 import WillStreetPO3


class UnifiedStrategy:
    """
    🤖 SUPERBOT INTELIGENTE 🤖
    
    Jerarquía de Decisión:
    1. Will Street PO3: MOTOR PRINCIPAL (timing preciso en velas de 4h)
       → Si PO3 da señal con RR ≥ 3:1 → ENTRADA DIRECTA
    
    2. SMC Engine: CONFIRMACIÓN ADICIONAL
       → Si coincide con PO3 → BOOST de confianza (+puntos)
       → Si PO3 no da señal pero SMC tiene ≥ 7 confluencia → Entrada secundaria
    
    3. Wyckoff: FILTRO DIRECCIONAL (opcional)
       → Solo bloquea si va en contra de tendencia macro
    """
    
    def __init__(
        self,
        use_po3: bool = True,
        use_smc: bool = True,
        use_wyckoff: bool = False,
        min_confluence: int = 1,
        po3_min_rr: float = 3.0,
        smc_standalone_threshold: int = 7,
        po3_weight: int = 10,
        smc_weight: int = 3,
        wyckoff_weight: int = 2
    ):
        self.use_po3 = use_po3
        self.use_smc = use_smc
        self.use_wyckoff = use_wyckoff
        self.min_confluence = min_confluence
        self.po3_min_rr = po3_min_rr
        self.smc_standalone_threshold = smc_standalone_threshold
        
        # Weights para scoring
        self.po3_weight = po3_weight
        self.smc_weight = smc_weight
        self.wyckoff_weight = wyckoff_weight
        
        # Inicializar engines
        if use_po3:
            self.po3 = WillStreetPO3(min_rr_ratio=po3_min_rr)  # Solo RR ratio importa
        
        if use_smc:
            self.smc = SMCEngine(swing_length=5, fvg_threshold=0.001)
        
        if use_wyckoff:
            self.wyckoff = MultiTimeframeWyckoff(timeframes=['1h', '4h', '1d'])
        
        logger.success(f"🤖 SUPERBOT Initialized!")
        logger.info(f"   └─ PO3 Primary: {use_po3} (Min RR: {po3_min_rr}:1)")
        logger.info(f"   └─ SMC Confirm: {use_smc} (Standalone: {smc_standalone_threshold}+ pts)")
        logger.info(f"   └─ Wyckoff Filter: {use_wyckoff}")
    
    def get_trading_signal(
        self,
        data_dict: Dict[str, pd.DataFrame],
        current_time: pd.Timestamp,
        execution_tf: str = '15m'
    ) -> Optional[Dict]:
        """Genera señal combinando todas las metodologías"""
        
        po3_signal = None
        smc_signal = None
        wyckoff_signal = None
        
        confluence_score = 0
        reasons = []
        
        # PASO 1: WILL STREET PO3 - MOTOR PRINCIPAL
        if self.use_po3 and '4h' in data_dict:
            po3_signal = self.po3.analyze_4h_candle(
                df_4h=data_dict['4h'],
                df_15m=data_dict.get('15m', data_dict.get('1h')),
                current_time=current_time
            )
            
            if po3_signal and po3_signal.get('rr_ratio', 0) >= self.po3_min_rr:
                confluence_score += self.po3_weight
                reasons.append(f"🎯 PO3 {po3_signal['type']} (RR: {po3_signal['rr_ratio']:.1f}:1)")
                logger.success(f"⚡ PO3 PRIMARY SIGNAL: {po3_signal['type']} @ ${po3_signal['entry']:.2f}")
        
        # PASO 2: SMC - CONFIRMACIÓN O ENTRADA SECUNDARIA
        if self.use_smc and execution_tf in data_dict:
            try:
                smc_signal = self.smc.get_trading_signals(
                    data_dict[execution_tf],
                    current_time
                )
                
                if smc_signal and smc_signal.get('confluence_score', 0) >= 5:
                    smc_conf = smc_signal['confluence_score']
                    
                    # Si coincide con PO3, es BOOST
                    if po3_signal and smc_signal.get('signal') == po3_signal.get('type'):
                        confluence_score += self.smc_weight
                        reasons.append(f"✨ SMC Confirm ({smc_conf} pts)")
                        logger.info(f"   └─ SMC BOOST: {smc_conf} confluence points")
                    
                    # Si NO hay PO3 pero SMC es ultra-fuerte, puede entrar solo
                    elif not po3_signal and smc_conf >= self.smc_standalone_threshold:
                        confluence_score += self.smc_weight
                        reasons.append(f"🔥 SMC Standalone ({smc_conf} pts)")
                        logger.warning(f"⚠️  SMC STANDALONE: {smc_signal.get('signal', 'UNKNOWN')} with {smc_conf} points")
            except Exception as e:
                logger.debug(f"SMC analysis error: {e}")
        
        # PASO 3: WYCKOFF - FILTRO DIRECCIONAL (OPCIONAL)
        if self.use_wyckoff:
            try:
                for tf, df in data_dict.items():
                    if tf in ['1h', '4h', '1d']:
                        self.wyckoff.load_data(tf, df)
                
                self.wyckoff.analyze_all_timeframes()
                wyckoff_signal = self.wyckoff.get_trading_signal()
                
                if wyckoff_signal and wyckoff_signal['type'] in ['LONG', 'SHORT']:
                    if (po3_signal and wyckoff_signal['type'] == po3_signal['type']) or \
                       (smc_signal and wyckoff_signal['type'] == smc_signal['type']):
                        confluence_score += self.wyckoff_weight
                        reasons.append(f"📈 Wyckoff {wyckoff_signal['type']}")
                    elif po3_signal and wyckoff_signal['type'] != po3_signal['type']:
                        logger.warning(f"⚠️  WYCKOFF CONFLICT")
            except Exception as e:
                logger.debug(f"Wyckoff skipped: {e}")
        
        # DECISIÓN FINAL
        
        # Caso 1: PO3 dio señal de alta calidad → ENTRADA DIRECTA
        if po3_signal and po3_signal.get('rr_ratio', 0) >= self.po3_min_rr:
            logger.success(f"🚀 SUPERBOT ENTRY: PO3 Primary | Score: {confluence_score}")
            
            return {
                'type': po3_signal['type'],
                'entry': po3_signal['entry'],
                'stop_loss': po3_signal['stop_loss'],
                'take_profit': po3_signal['take_profit'],
                'confluence_score': confluence_score,
                'reason': " + ".join(reasons),
                'rr_ratio': po3_signal['rr_ratio'],
                'confidence': 'HIGH' if confluence_score >= 10 else 'MEDIUM'
            }
        
        # Caso 2: SMC tiene confluencia ultra-alta → Entrada secundaria
        if smc_signal and smc_signal.get('confluence_score', 0) >= self.smc_standalone_threshold:
            logger.warning(f"⚡ SUPERBOT ENTRY: SMC Standalone | Score: {confluence_score}")
            
            return {
                'type': smc_signal['signal'],  # SMC uses 'signal' not 'type'
                'entry': smc_signal.get('entry_price', smc_signal.get('entry')),
                'stop_loss': smc_signal['stop_loss'],
                'take_profit': smc_signal['take_profit'],
                'confluence_score': confluence_score,
                'reason': " + ".join(reasons),
                'rr_ratio': 3.0,
                'confidence': 'MEDIUM'
            }
        
        # Caso 3: No hay señal válida
        return None

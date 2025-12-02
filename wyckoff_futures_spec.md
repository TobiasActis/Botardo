# Wyckoff Futures Bot - Especificación Técnica

## Resumen Ejecutivo

Bot de trading automatizado que implementa la metodología Wyckoff para futuros de criptomonedas con análisis multi-timeframe. Diseñado para identificar fases institucionales de acumulación y distribución, ejecutar trades con gestión automática de riesgo y protección contra liquidación.

## Arquitectura del Sistema

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    BOTARDO SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────┐         ┌────────────────────┐     │
│  │  Data Ingestion    │────────▶│  Wyckoff Analysis  │     │
│  │  (Binance API)     │         │  (Multi-TF)        │     │
│  └────────────────────┘         └──────────┬─────────┘     │
│                                             │               │
│                                             ▼               │
│                                  ┌────────────────────┐     │
│                                  │  Signal Generator  │     │
│                                  └──────────┬─────────┘     │
│                                             │               │
│                                             ▼               │
│                                  ┌────────────────────┐     │
│                                  │  Risk Manager      │     │
│                                  └──────────┬─────────┘     │
│                                             │               │
│                                             ▼               │
│  ┌────────────────────┐         ┌────────────────────┐     │
│  │  Position Monitor  │◀────────│  Order Executor    │     │
│  │  (Liquidation)     │         │  (Binance Futures) │     │
│  └────────────────────┘         └────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Datos

1. **Ingestión**: Descarga datos OHLCV multi-timeframe (1h, 4h, 1d)
2. **Análisis Wyckoff**: Identifica fases y eventos clave
3. **Generación de Señales**: Evalúa confluencia entre timeframes
4. **Gestión de Riesgo**: Calcula posición óptima y leverage
5. **Ejecución**: Coloca órdenes con SL/TP automáticos
6. **Monitoreo**: Tracking continuo de liquidación y P&L

## Metodología Wyckoff Implementada

### Fases del Ciclo

#### 1. Acumulación (Accumulation)
Zona de compra institucional caracterizada por:
- Rango de precio estrecho (< 5% del precio medio)
- Incremento de volumen (> 120% del promedio)
- Price action lateral

**Eventos Clave**:
- **PS (Preliminary Support)**: Primera señal de soporte
- **SC (Selling Climax)**: Volumen masivo + caída fuerte
  - Condición: `volume_ratio > 2.0 AND range > 1.5 * range_ma`
- **AR (Automatic Rally)**: Rebote post-clímax
- **ST (Secondary Test)**: Prueba del mínimo con volumen menor
- **Spring**: Penetración falsa del soporte
  - Condición: `low < support * 0.995 AND close > support AND volume_ratio < 1.2`
- **LPS (Last Point of Support)**: Último test antes del markup

#### 2. Markup (Uptrend)
Tendencia alcista en desarrollo:
- Price trend > +10%
- Volumen sostenido (ratio > 1.0)
- Series de Higher Highs y Higher Lows

#### 3. Distribución (Distribution)
Zona de venta institucional:
- Rango de precio estrecho cerca de máximos
- Incremento de volumen > 120%
- Price action lateral en zona alta

**Eventos Clave**:
- **PSY (Preliminary Supply)**: Primera oferta significativa
- **BC (Buying Climax)**: Volumen masivo + subida fuerte
  - Condición: `volume_ratio > 2.0 AND close > open AND range > 1.5 * range_ma`
- **AR (Automatic Reaction)**: Caída post-clímax
- **ST (Secondary Test)**: Prueba del máximo con volumen menor
- **UTAD (Upthrust After Distribution)**: Penetración falsa de resistencia
  - Condición: `high > resistance * 1.005 AND close < resistance AND volume_ratio < 1.2`
- **LPSY (Last Point of Supply)**: Última distribución antes del markdown

#### 4. Markdown (Downtrend)
Tendencia bajista en desarrollo:
- Price trend < -10%
- Volumen sostenido
- Series de Lower Highs y Lower Lows

### Análisis Multi-Timeframe

El bot analiza **3 timeframes simultáneamente**:

| Timeframe | Propósito | Peso |
|-----------|-----------|------|
| 1h | Timing de entrada/salida | 50% |
| 4h | Confirmación de tendencia | 30% |
| 1d | Contexto de mercado | 20% |

**Lógica de Confluencia**:
- Señal LONG: ≥2 TFs en Acumulación + Spring en 1h
- Señal SHORT: ≥2 TFs en Distribución + UTAD en 1h

## Gestión de Riesgo y Tamaño de Posición

### Cálculo de Position Sizing

```python
# Variables
risk_per_trade = 2%  # % del balance arriesgado
entry_price = precio de entrada
stop_loss = precio de invalidación
max_leverage = 5x

# Cálculo
stop_distance = abs(entry_price - stop_loss) / entry_price
leverage_optimal = min(1 / stop_distance, max_leverage)
risk_amount = balance * risk_per_trade
position_value = risk_amount / stop_distance
quantity = position_value / entry_price
```

### Protección contra Liquidación

**Cálculo de Precio de Liquidación**:

```python
# Para LONG
liquidation_price = entry_price * (1 - 1/leverage + margin_ratio)

# Para SHORT
liquidation_price = entry_price * (1 + 1/leverage - margin_ratio)
```

**Reglas de Protección**:
1. Buffer mínimo del 15% entre SL y precio de liquidación
2. Uso máximo del 70% del margen disponible
3. Monitoreo continuo del margin ratio

### Stop Loss y Take Profit

**Stop Loss**:
- **Acumulación**: Por debajo del Spring level
  - `SL = creek * 0.995`
- **Distribución**: Por encima del UTAD level
  - `SL = ice * 1.005`

**Take Profit**:
- Ratio Risk:Reward mínimo de 2:1
- **LONG**: Resistencia del rango
- **SHORT**: Soporte del rango

## Indicadores Técnicos Utilizados

### 1. Volumen Relativo
```python
volume_ma = SMA(volume, 20)
volume_ratio = volume / volume_ma
```

### 2. Average True Range (ATR)
```python
tr = max(high - low, |high - close_prev|, |low - close_prev|)
atr = SMA(tr, 14)
```

### 3. Effort vs Result
```python
effort = volume_ratio
result = |close - open| / range
```

Divergencias effort vs result indican absorción/distribución.

### 4. Soporte/Resistencia Dinámicos
```python
resistance = max(high, 20)
support = min(low, 20)
midpoint = (resistance + support) / 2
```

## Configuración del Sistema

### Variables de Entorno (.env)

```bash
# API Configuration
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=true

# Trading Parameters
TRADING_SYMBOL=BTCUSDT
TIMEFRAMES=1h,4h,1d
RISK_PER_TRADE=0.02
MAX_LEVERAGE=5

# Wyckoff Parameters
MIN_ACCUMULATION_BARS=20
MIN_DISTRIBUTION_BARS=20
VOLUME_THRESHOLD_MULTIPLIER=1.5
SPRING_DETECTION_ENABLED=true

# Safety
LIQUIDATION_BUFFER_PERCENT=0.15
MAX_MARGIN_USAGE=0.7
```

## Backtesting

### Métricas Evaluadas

1. **Total Return**: P&L total en %
2. **Win Rate**: % de trades ganadores
3. **Profit Factor**: (Avg Win * Win Count) / (Avg Loss * Loss Count)
4. **Max Drawdown**: Máxima caída desde peak
5. **Sharpe Ratio**: Return ajustado por riesgo
6. **Average Trade Duration**: Tiempo promedio en posición

### Proceso de Backtest

```bash
# 1. Descargar datos históricos
python download_btc_futures_1m.py

# 2. Ejecutar backtest
python backtest_wyckoff.py

# 3. Analizar resultados
# - backtest_results.png (gráficos)
# - trades.csv (registro detallado)
```

## Seguridad y Validaciones

### Pre-ejecución
- ✅ Validación de API keys
- ✅ Verificación de balance disponible
- ✅ Check de symbol filters (stepSize, minNotional)
- ✅ Validación de leverage permitido

### Durante Operación
- ✅ Monitoreo continuo de margin ratio
- ✅ Alertas de proximidad a liquidación
- ✅ Rate limiting de API requests
- ✅ Manejo de errores de conexión

### Post-ejecución
- ✅ Logging de todas las operaciones
- ✅ Registro de P&L en base de datos
- ✅ Notificaciones (Telegram opcional)

## Limitaciones y Consideraciones

### Limitaciones Técnicas
1. Requiere datos históricos suficientes (mín. 100 velas por TF)
2. Sensible a slippage en mercados de baja liquidez
3. Performance depende de volatilidad del mercado

### Riesgos
1. **Riesgo de Liquidación**: Siempre presente en futuros apalancados
2. **Riesgo de Cascada**: Movimientos bruscos pueden invalidar SL
3. **Riesgo de API**: Fallos de conexión pueden impedir cerrar posiciones
4. **Riesgo de Overtrading**: En mercados laterales prolongados

### Mejores Prácticas
- Comenzar siempre en TESTNET
- Usar leverage conservador (≤3x inicialmente)
- No arriesgar más del 2% por trade
- Mantener registro de todos los trades
- Revisar performance semanalmente

## Roadmap Futuro

### v1.1
- [ ] Múltiples símbolos simultáneos
- [ ] Machine Learning para optimización de parámetros
- [ ] Integración con TradingView webhooks

### v1.2
- [ ] Dashboard web en tiempo real
- [ ] Sistema de alertas avanzado
- [ ] Auto-ajuste de parámetros según volatilidad

### v1.3
- [ ] Soporte para otros exchanges (Bybit, OKX)
- [ ] Estrategias combinadas (Wyckoff + VSA + Order Flow)
- [ ] Backtesting paralelo en múltiples timeframes

## Referencias

- **Wyckoff, Richard D.** - "The Richard D. Wyckoff Method of Trading and Investing in Stocks"
- **Binance Futures API**: https://binance-docs.github.io/apidocs/futures/en/
- **Python-Binance**: https://python-binance.readthedocs.io/

---

**Versión**: 1.0  
**Última actualización**: Diciembre 2024  
**Autor**: Tobias Actis

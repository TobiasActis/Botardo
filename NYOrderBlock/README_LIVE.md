# NY+EU OrderBlock Bot - Trading en Vivo

Bot de trading automatizado que opera en Binance Testnet Futures detectando orderblocks en las sesiones EU y NY.

## 🎯 Estrategia

### Sesiones (Hora NY - America/New_York)
- **EU Pre-Sesión:** 01:00 - 02:00
- **EU Sesión:** 02:00 - 06:00
- **NY Pre-Sesión:** 06:45 - 07:30
- **NY Sesión:** 07:30 - 11:00

### Lógica de Trading
1. **Detección de Orderblock:** Durante la pre-sesión, identifica la última vela con fuerte momentum (body ≥ 50% del rango)
2. **Entry Level:** 
   - SELL: Bajo de la vela bajista
   - BUY: Alto de la vela alcista
3. **Stop Loss:** Lado opuesto de la vela ± 0.001 buffer
4. **Take Profit:** Ratio 1:1 del riesgo (risk-reward fijo)
5. **Filtros de Calidad:**
   - Vela debe tener mínimo 0.15% de rango
   - Body debe ser ≥ 50% del rango total
   - Tendencia 1h debe coincidir (EMA20 vs EMA50)

### Parámetros
- **Capital Inicial:** Balance disponible en cuenta
- **Riesgo por Trade:** 2% del capital
- **Apalancamiento:** 5x
- **Comisiones:** 0.04% por trade (solo apertura)
- **Símbolo:** BTCUSDT

## 📦 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Uso

### Ejecutar el bot

```bash
python trade_live_nyorderblock.py
```

### En servidor Linux (background con screen)

```bash
# Crear sesión de screen
screen -S nyorderblock

# Ejecutar bot
python trade_live_nyorderblock.py

# Detach (Ctrl+A, luego D)
# Para reconectar: screen -r nyorderblock
```

## 🔑 Configuración API

Edita el archivo `trade_live_nyorderblock.py` y configura tus credenciales:

```python
API_KEY = "tu_api_key_aqui"
API_SECRET = "tu_api_secret_aqui"
TESTNET_URL = "https://testnet.binancefuture.com"  # Para testnet
# TESTNET_URL = "https://fapi.binance.com"  # Para producción REAL
```

⚠️ **IMPORTANTE:** El bot está configurado para Binance Testnet por defecto. Para operar con dinero real:
1. Cambia el `TESTNET_URL` a `https://fapi.binance.com`
2. Usa tus API keys de producción
3. **PRUEBA PRIMERO EN TESTNET** antes de usar dinero real

## 📊 Monitoreo

El bot muestra en consola:
- Balance disponible
- Señales detectadas (BUY/SELL)
- Entrada, SL y TP de cada trade
- Monitoreo en tiempo real de posiciones abiertas
- Cierre de posiciones (TP o SL alcanzado)

También genera un archivo CSV con todas las señales:
- `signals_detected_BTCUSDT.csv`

## 🛡️ Seguridad

- El bot solo abre **una posición a la vez**
- Siempre usa **Stop Loss** (definido antes de entrar)
- **Take Profit** fijo en ratio 1:1
- Riesgo limitado al 2% del capital por trade
- Filtros de calidad para evitar señales falsas

## 📈 Resultados Históricos (Backtest 2018-2025)

- **Total Trades:** 1044
- **Win Rate:** 64.94%
- **Retorno Total:** +1978.57%
- **Profit Factor:** 1.27
- **Max Drawdown:** -26.17%

## ⚙️ Personalización

Puedes ajustar los parámetros en el archivo:

```python
LEVERAGE = 5  # Apalancamiento
RISK_PER_TRADE = 0.02  # 2% riesgo por trade
MIN_CANDLE_RANGE_PCT = 0.15 / 100  # Rango mínimo de vela
MIN_BODY_PCT = 0.50  # Body mínimo 50%
```

## 🐛 Troubleshooting

### Error de timestamp
Si ves errores de timestamp con Binance, el bot usa automáticamente el servidor de Binance para sincronizar la hora.

### Error de conexión
El bot reintenta automáticamente cada 30 segundos si pierde conexión.

### No detecta señales
- Verifica que estés en horario de sesión (EU o NY en timezone America/New_York)
- Los filtros de calidad pueden estar descartando señales débiles (esto es intencional)

## 📝 Notas

- El bot verifica el mercado cada **60 segundos**
- Solo opera durante las **sesiones EU (02:00-06:00) y NY (07:30-11:00)** hora NY
- **No martingale:** Cada trade es independiente con riesgo fijo del 2%
- Los orderblocks se detectan en la **pre-sesión** y se operan durante la **sesión**

## ⚠️ Disclaimer

Este bot es para propósitos educativos. El trading de futuros conlleva riesgo de pérdida de capital. Usa bajo tu propio riesgo y siempre prueba en testnet antes de operar con dinero real.

---

**¿Preguntas?** Revisa el código fuente y los comentarios para entender la lógica completa.

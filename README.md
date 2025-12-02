# Botardo - Wyckoff Multi-Timeframe Futures Bot

Bot de trading automatizado para futuros de criptomonedas que implementa la metodología Wyckoff con análisis multi-timeframe.

## 🎯 Características

- **Análisis Wyckoff Multi-Timeframe**: Detección de fases de acumulación/distribución en múltiples temporalidades
- **Trading de Futuros**: Optimizado para Binance Futures (Testnet y Mainnet)
- **Gestión de Liquidez**: Cálculo automático de niveles de liquidación y gestión de riesgo
- **Backtesting**: Sistema completo de backtesting con datos históricos de 1 minuto
- **CI/CD**: Workflows automáticos para validación y despliegue

## 📋 Requisitos

- Python 3.9+
- Cuenta en Binance Futures (Testnet para pruebas)
- API Keys de Binance Futures

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/TobiasActis/Botardo.git
cd Botardo

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

## 📊 Uso

### 1. Descargar Datos Históricos

```bash
python download_btc_futures_1m.py
```

Descarga datos de BTC/USDT futures en temporalidad de 1 minuto para backtesting.

### 2. Ejecutar Backtest

```bash
python backtest_wyckoff.py
```

Ejecuta el backtesting del sistema Wyckoff con los datos descargados. Genera reportes de performance y gráficos.

### 3. Trading en Vivo (Testnet)

```bash
python futures_executor_with_liq.py
```

⚠️ **IMPORTANTE**: Primero prueba en Testnet antes de usar fondos reales.

## 🏗️ Arquitectura

```
Botardo/
├── multi_tf_wyckoff_rules.py      # Lógica de análisis Wyckoff multi-timeframe
├── futures_executor_with_liq.py   # Executor de órdenes con gestión de liquidación
├── backtest_wyckoff.py            # Motor de backtesting
├── download_btc_futures_1m.py     # Script de descarga de datos
├── wyckoff_futures_spec.md        # Especificación técnica detallada
├── requirements.txt               # Dependencias Python
├── .env.example                   # Plantilla de configuración
└── .github/workflows/grid.yml     # CI/CD automation
```

## 📈 Metodología Wyckoff

El bot implementa las siguientes fases del ciclo Wyckoff:

- **Acumulación**: Identificación de zonas de compra institucional
- **Markup**: Detección de tendencia alcista en desarrollo
- **Distribución**: Identificación de zonas de venta institucional
- **Markdown**: Detección de tendencia bajista en desarrollo

Ver `wyckoff_futures_spec.md` para detalles técnicos completos.

## ⚙️ Configuración

Variables de entorno requeridas en `.env`:

```env
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret
BINANCE_TESTNET=true
TRADING_SYMBOL=BTCUSDT
TIMEFRAMES=1h,4h,1d
RISK_PER_TRADE=0.02
```

## 🧪 Testing

```bash
# Ejecutar tests unitarios
python -m pytest tests/

# Ejecutar backtest con datos de ejemplo
python backtest_wyckoff.py --mode=quick
```

## 📝 Gestión de Riesgo

- **Stop Loss**: Basado en invalidación de estructura Wyckoff
- **Take Profit**: Objetivos calculados según fases del ciclo
- **Tamaño de Posición**: Calculado automáticamente según capital y riesgo
- **Protección de Liquidación**: Monitoreo continuo de margen y niveles de liquidación

## 🔐 Seguridad

- ⚠️ **NUNCA** commitear archivos `.env` con API keys reales
- Usar Testnet para todas las pruebas iniciales
- Validar cálculos de liquidación antes de operar
- Implementar límites de pérdida diaria/semanal

## 📖 Documentación

- [Especificación Técnica](wyckoff_futures_spec.md)
- [Metodología Wyckoff](docs/wyckoff-methodology.md) (próximamente)
- [API Reference](docs/api-reference.md) (próximamente)

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Add: nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## ⚖️ Licencia

Este proyecto es de código privado. No distribuir sin autorización.

## ⚠️ Disclaimer

Este software es para fines educativos y de investigación. El trading de futuros conlleva riesgo de pérdida total del capital. Usa bajo tu propio riesgo.

---

**Desarrollado por Tobias Actis** | [GitHub](https://github.com/TobiasActis)

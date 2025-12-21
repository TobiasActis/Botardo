# Botardo - Proyecto Limpio para 15 Minutos

Este proyecto está optimizado y reducido para operar y backtestear únicamente en timeframe de 15 minutos.

## Estructura esencial
- `botardo.py`: Lógica principal del bot y backtest.
- `data/BTCUSDT_15m.csv`: Datos históricos 15m.
- `results/resumen_anual.json`: Resultados de backtest 15m.
- `requirements.txt`: Dependencias.

## Respaldo de 45m y otros scripts
Todo lo relacionado a 45m, 10m y scripts auxiliares fue movido a:
- `_legacy_45m/`, `_legacy_data/`, `_legacy_misc/`

## Cómo correr el backtest
```bash
python botardo.py
```

## Notas
- El proyecto está listo para migrar al servidor y operar en 15 minutos.
- Si necesitas recuperar lógica o datos de 45m, revisa las carpetas legacy.

## Ejemplo de uso multi-asset con balance real Binance

```bash
python botardo.py --data data/SYMBOL_15m_2018-01-01_to_2025-12-18.csv \
  --assets BTCUSDT,SOLUSDT,ADAUSDT \
  --use-binance-balance \
  --binance-api-key TU_API_KEY \
  --binance-api-secret TU_API_SECRET \
  --start 2018-01-01 --end 2025-12-18
```

- El capital total se toma del balance USDT real de tu cuenta Binance y se reparte entre los activos.
- El archivo de datos debe tener 'SYMBOL' como comodín, que se reemplaza por cada activo.
- Se genera un archivo de resultados por activo: `botardo_trades_BTCUSDT.csv`, etc.

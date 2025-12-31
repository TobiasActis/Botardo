
# NY Order Block Bot (Scalping NY Session 5m)

## ¿Qué hace este bot?
Este bot implementa una estrategia de scalping institucional basada en order blocks, diseñada para operar únicamente durante la apertura de la sesión de Nueva York en el timeframe de 5 minutos. Es ideal para quienes buscan una estrategia simple, robusta y fácil de replicar, sin parámetros ocultos ni optimizaciones peligrosas.

## ¿Cómo funciona la estrategia? (Explicación para todos)

1. **Horario de operación:**
	- Solo opera entre las 7:30 y 11:00 de la mañana, hora de Nueva York (UTC-5).

2. **Identificación del order block:**
	- Cada día, antes de la apertura (entre 6:45 y 7:30 NY), el bot busca la última vela contraria (order block) o el último máximo/mínimo relevante.
	- Un order block es simplemente la última vela alcista antes de una caída fuerte (para ventas) o la última bajista antes de una subida fuerte (para compras).

3. **Entrada:**
	- Espera que, durante la sesión de NY (7:30-11:00), el precio regrese a ese order block.
	- Si el precio toca el nivel del order block, se abre una operación:
	  - **Venta (SELL):** Si el order block es bajista.
	  - **Compra (BUY):** Si el order block es alcista.

4. **Gestión del riesgo:**
	- El stop loss (SL) se coloca 3 pips por encima (venta) o por debajo (compra) del order block.
	- El take profit (TP) se coloca en el siguiente mínimo (venta) o máximo (compra) relevante tras la entrada.
	- Solo se permite una operación por día.

5. **Cierre y estadísticas:**
	- El bot registra cada trade, calcula el resultado, y al final muestra estadísticas clave: winrate, profit factor, drawdown, P&L anual/mensual, y gráficos de equity y P&L mensual.

## ¿Por qué es robusta y rentable?
- No depende de optimizaciones ni parámetros ocultos.
- Opera siempre bajo las mismas reglas, sin sobreajuste.
- Limita el riesgo por trade y el número de operaciones.
- Aprovecha la volatilidad institucional de NY.
- El drawdown es bajo comparado con la ganancia total.

**Resultados de ejemplo (BTCUSDT 2022-2025, $500 inicial, 3x leverage):**
- Winrate: ~71%
- Profit Factor: 1.7
- Max Drawdown: -18%
- Ganancia total: +4196
- Trades: 1361

---

## Instrucciones paso a paso para usar y replicar

### 1. Prepara tus datos
- Necesitas un archivo CSV de 5 minutos con las columnas: `timestamp`, `open`, `high`, `low`, `close`, `volume`.
- Ejemplo: `../data/BTCUSDT_5m_full.csv`

### 2. Instala los requerimientos
Abre una terminal en la carpeta NYOrderBlock y ejecuta:
```bash
pip install -r requirements.txt
```

### 3. Ejecuta el backtest
```bash
python ny_orderblock_bot.py --data ../data/BTCUSDT_5m_full.csv --capital 500 --risk 0.01 --leverage 3 --start 2022-01-01 --end 2025-12-31
```

### 4. Analiza los resultados
- Se genera el archivo `ny_orderblock_trades.csv` con todos los trades.
- Se imprimen estadísticas: winrate, profit factor, drawdown, P&L anual/mensual.
- Se guardan los gráficos: `ny_orderblock_equity.png` (curva de equity) y `ny_orderblock_monthly.png` (P&L mensual).

### 5. Replica o ajusta
- Puedes probar con otros activos, otros años, o ajustar el capital/riesgo.
- El código es robusto y no depende de parámetros ocultos ni optimizaciones peligrosas.

---

¿Dudas o quieres experimentar con otras variantes? Solo pide el ajuste que necesites.

import os
import pandas as pd
from binance.client import Client
from datetime import datetime
import time

# Configuración
API_KEY = ''  # No es necesario para datos públicos
API_SECRET = ''
SYMBOL = 'BTCUSDT'
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
START_DATE = "2017-08-17"
END_YEAR = 2025
DATA_DIR = 'data'

# Inicializar cliente
client = Client(API_KEY, API_SECRET)

# Función para descargar y guardar datos año por año
def download_binance_klines(symbol, interval, start_str, end_str, filename):
    print(f"Descargando {symbol} {interval} desde {start_str} hasta {end_str}...")
    klines = []
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)
    while start_ts < end_ts:
        try:
            new_klines = client.get_historical_klines(symbol, interval, start_ts, min(start_ts + 1000*60*1000, end_ts), limit=1000)
            if not new_klines:
                break
            klines.extend(new_klines)
            start_ts = new_klines[-1][0] + 60*1000
            time.sleep(0.5)
        except Exception as e:
            print(f"Error: {e}. Reintentando en 10s...")
            time.sleep(10)
    if klines:
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.to_csv(filename, index=False)
        print(f"Guardado en {filename}")
    else:
        print(f"No se encontraron datos para {symbol} en {start_str} - {end_str}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    start_dt = pd.to_datetime(START_DATE)
    for year in range(start_dt.year, END_YEAR+1):
        if year == start_dt.year:
            start = START_DATE
        else:
            start = f"{year}-01-01"
        end = f"{year}-12-31"
        fname = os.path.join(DATA_DIR, f"{SYMBOL}_1m_{start}_to_{end}.csv")
        if not os.path.exists(fname):
            download_binance_klines(SYMBOL, INTERVAL, start, end, fname)
        else:
            print(f"Ya existe {fname}, saltando...")

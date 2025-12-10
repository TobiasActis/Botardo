"""
Download BTC Futures 1-minute Data from Binance
Descarga datos históricos de futuros de BTC para backtesting
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from loguru import logger
import time


def download_futures_data(
    symbol: str = 'BTCUSDT',
    interval: str = '1m',
    start_date: str = '2024-01-01',
    end_date: str = None,
    save_path: str = 'data'
):
    """
    Descarga datos históricos de futuros de Binance
    
    Args:
        symbol: Par de trading (default: BTCUSDT)
        interval: Intervalo de velas (1m, 5m, 15m, 1h, 4h, 1d)
        start_date: Fecha de inicio (formato: YYYY-MM-DD)
        end_date: Fecha de fin (default: hoy)
        save_path: Carpeta donde guardar los datos
    """
    # Cliente público (no requiere API keys para datos históricos)
    client = Client()
    
    # Crear directorio si no existe
    os.makedirs(save_path, exist_ok=True)
    
    # Convertir fechas
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    
    if end_date:
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    else:
        end_ts = int(datetime.now().timestamp() * 1000)
    
    logger.info(f"Downloading {symbol} {interval} data from {start_date} to {end_date or 'now'}")
    
    # Binance limita a 1000 velas por request
    limit = 1000
    
    # Calcular intervalos en ms
    interval_ms = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
    }
    
    step_ms = interval_ms[interval] * limit
    
    all_data = []
    current_start = start_ts
    
    while current_start < end_ts:
        try:
            # Descargar chunk
            logger.info(f"Downloading from {datetime.fromtimestamp(current_start/1000)}")
            
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                limit=limit
            )
            
            if not klines:
                break
            
            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Seleccionar columnas relevantes
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convertir tipos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            all_data.append(df)
            
            # Actualizar timestamp para siguiente iteración
            current_start = int(klines[-1][0]) + interval_ms[interval]
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            time.sleep(5)
            continue
    
    if not all_data:
        logger.error("No data downloaded")
        return None
    
    # Concatenar todos los chunks
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['timestamp'])
    final_df = final_df.sort_values('timestamp')
    final_df.set_index('timestamp', inplace=True)
    
    # Guardar a CSV
    filename = f"{symbol}_{interval}_{start_date}_to_{end_date or 'now'}.csv"
    filepath = os.path.join(save_path, filename)
    final_df.to_csv(filepath)
    
    logger.info(f"Downloaded {len(final_df)} bars")
    logger.info(f"Saved to: {filepath}")
    logger.info(f"Date range: {final_df.index[0]} to {final_df.index[-1]}")
    
    return final_df


def resample_to_higher_timeframes(df_1m: pd.DataFrame, save_path: str = 'data'):
    """
    Resamples 1-minute data to higher timeframes
    
    Args:
        df_1m: DataFrame con datos de 1 minuto
        save_path: Carpeta donde guardar los datos
    """
    timeframes = {
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }
    
    resampled_data = {}
    
    for name, freq in timeframes.items():
        logger.info(f"Resampling to {name}...")
        
        df_resampled = df_1m.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Guardar
        filename = f"BTCUSDT_{name}_resampled.csv"
        filepath = os.path.join(save_path, filename)
        df_resampled.to_csv(filepath)
        
        logger.info(f"Saved {name} data: {len(df_resampled)} bars to {filepath}")
        resampled_data[name] = df_resampled
    
    return resampled_data


def main():
    """Función principal para descarga de datos"""
    
    # Configuración
    symbol = 'BTCUSDT'
    start_date = '2024-01-01'
    end_date = None  # None = hasta hoy
    
    logger.info("Starting data download...")
    
    # Descargar datos de 1 minuto
    df_1m = download_futures_data(
        symbol=symbol,
        interval='1m',
        start_date=start_date,
        end_date=end_date
    )
    
    if df_1m is None:
        logger.error("Failed to download data")
        return
    
    # Resamplear a timeframes superiores
    logger.info("\nResampling to higher timeframes...")
    resampled = resample_to_higher_timeframes(df_1m)
    
    logger.success("\n✅ Data download complete!")
    logger.info("\nDownloaded files:")
    logger.info("- BTCUSDT_1m_*.csv (raw 1-minute data)")
    logger.info("- BTCUSDT_1h_resampled.csv")
    logger.info("- BTCUSDT_4h_resampled.csv")
    logger.info("- BTCUSDT_1d_resampled.csv")
    
    logger.info("\nYou can now run: python backtest_wyckoff.py")


if __name__ == "__main__":
    main()

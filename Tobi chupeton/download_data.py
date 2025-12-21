"""
Download BTC Futures 1-minute Data from Binance
Descarga datos históricos de futuros de BTC para backtesting
"""

import os
import argparse
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
    client = Client()
    os.makedirs(save_path, exist_ok=True)
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    if end_date:
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    else:
        end_ts = int(datetime.now().timestamp() * 1000)
    logger.info(f"Downloading {symbol} {interval} data from {start_date} to {end_date or 'now'}")
    limit = 1000
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
            logger.info(f"Downloading from {datetime.fromtimestamp(current_start/1000)}")
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                limit=limit
            )
            if not klines:
                break
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            all_data.append(df)
            current_start = int(klines[-1][0]) + interval_ms[interval]
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            time.sleep(5)
            continue
    if not all_data:
        logger.error("No data downloaded")
        return None
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['timestamp'])
    final_df = final_df.sort_values('timestamp')
    final_df.set_index('timestamp', inplace=True)
    filename = f"{symbol}_{interval}_{start_date}_to_{end_date or 'now'}.csv"
    filepath = os.path.join(save_path, filename)
    final_df.to_csv(filepath)
    logger.info(f"Downloaded {len(final_df)} bars")
    logger.info(f"Saved to: {filepath}")
    logger.info(f"Date range: {final_df.index[0]} to {final_df.index[-1]}")
    return final_df

# --- NUEVO: descarga datos spot ---
def download_spot_data(
    symbol: str = 'BTCUSDT',
    interval: str = '1m',
    start_date: str = '2024-01-01',
    end_date: str = None,
    save_path: str = 'data'
):
    """
    Descarga datos históricos spot de Binance
    """
    client = Client()
    os.makedirs(save_path, exist_ok=True)
    # Convertir fechas a string formato API
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    logger.info(f"Downloading SPOT {symbol} {interval} data from {start_date} to {end_date}")
    # Binance API espera formato: '1 Jan, 2018'
    start_str = datetime.strptime(start_date, '%Y-%m-%d').strftime('%d %b, %Y')
    end_str = datetime.strptime(end_date, '%Y-%m-%d').strftime('%d %b, %Y')
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    except Exception as e:
        logger.error(f"Error downloading spot data: {e}")
        return None
    if not klines:
        logger.error("No spot data downloaded")
        return None
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    filename = f"{symbol}_SPOT_{interval}_{start_date}_to_{end_date}.csv"
    filepath = os.path.join(save_path, filename)
    df.to_csv(filepath)
    logger.info(f"Downloaded {len(df)} spot bars")
    logger.info(f"Saved to: {filepath}")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    return df

def resample_to_higher_timeframes(df_1m: pd.DataFrame, symbol: str = 'BTCUSDT', save_path: str = 'data'):
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
        filename = f"{symbol}_{name}_resampled.csv"
        filepath = os.path.join(save_path, filename)
        df_resampled.to_csv(filepath)
        logger.info(f"Saved {name} data: {len(df_resampled)} bars to {filepath}")
        resampled_data[name] = df_resampled
    return resampled_data

def main():
    parser = argparse.ArgumentParser(description='Download Binance historical data (Futures or Spot)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol to download (default: BTCUSDT)')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD (None = now)')
    parser.add_argument('--market', type=str, default='futures', choices=['futures', 'spot'], help='Market type: futures or spot')
    args = parser.parse_args()
    logger.info("Starting data download...")
    if args.market == 'futures':
        df_1m = download_futures_data(
            symbol=args.symbol,
            interval='1m',
            start_date=args.start,
            end_date=args.end
        )
    else:
        df_1m = download_spot_data(
            symbol=args.symbol,
            interval='1m',
            start_date=args.start,
            end_date=args.end
        )
    if df_1m is None:
        logger.error("Failed to download data")
        return
    logger.info("\nResampling to higher timeframes...")
    resampled = resample_to_higher_timeframes(df_1m, symbol=args.symbol)
    logger.success("\n✅ Data download complete!")
    logger.info("\nDownloaded files:")
    logger.info(f"- {args.symbol}_1m_*.csv (raw 1-minute data)")
    logger.info(f"- {args.symbol}_1h_resampled.csv")
    logger.info(f"- {args.symbol}_4h_resampled.csv")
    logger.info(f"- {args.symbol}_1d_resampled.csv")
    logger.info("\nYou can now run: python botardo.py")

if __name__ == "__main__":
    main()

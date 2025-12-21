
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Resample archivo OHLCV a otro timeframe')
    parser.add_argument('--input', required=True, help='Archivo CSV de entrada (1m)')
    parser.add_argument('--output', required=True, help='Archivo CSV de salida (resampleado)')
    parser.add_argument('--freq', required=True, help='Frecuencia de resampleo, ej: 15T, 10T, 1H, etc.')
    args = parser.parse_args()

    # Leer solo columnas necesarias para ahorrar memoria
    df = pd.read_csv(args.input, usecols=['timestamp','open','high','low','close','volume'], parse_dates=['timestamp'])
    df = df.set_index('timestamp')

    # Resampleo
    resampled = df.resample(args.freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    resampled = resampled.reset_index()
    resampled.to_csv(args.output, index=False)
    print(f'Resampleo completo: {args.output}')

if __name__ == '__main__':
    main()

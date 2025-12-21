import pandas as pd
import glob
import os

# Ruta de los archivos 1m descargados para 2018-2023
data_folder = 'data'
pattern = os.path.join(data_folder, 'BTCUSDT_1m_*.csv')
files = sorted(glob.glob(pattern))

# Filtrar solo los archivos de 2018 a 2023 (sin incluir 2024 ni 2025)
files = [f for f in files if any(str(y) in f for y in range(2018, 2024))]

# Leer y concatenar todos los archivos 1m
frames = []
for file in files:
    df = pd.read_csv(file, parse_dates=['timestamp'])
    frames.append(df)
df_1m = pd.concat(frames)

# Asegurar orden cronológico
if 'timestamp' in df_1m.columns:
    df_1m = df_1m.sort_values('timestamp')
    df_1m = df_1m.set_index('timestamp')
else:
    raise Exception('No se encontró la columna timestamp en los archivos 1m')

# Resampleo a 45m OHLCV
agg = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}
df_45m = df_1m.resample('45T').agg(agg).dropna()

# Guardar archivo 45m para 2018-2023
output_file = os.path.join(data_folder, 'BTCUSDT_45m_2018-2023.csv')
df_45m.to_csv(output_file)
print(f'Archivo 45m generado: {output_file}')
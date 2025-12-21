import pandas as pd
import os

folder = 'data'
files = [
    os.path.join(folder, 'BTCUSDT_45m_2018-2023.csv'),
    os.path.join(folder, 'BTCUSDT_45m.csv')  # 2024-2025
]

frames = []
for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file, parse_dates=['timestamp'])
        frames.append(df)
    else:
        print(f'No existe: {file}')

if frames:
    df_all = pd.concat(frames)
    df_all = df_all.sort_values('timestamp')
    df_all.to_csv(os.path.join(folder, 'BTCUSDT_45m.csv'), index=False)
    print('Archivo BTCUSDT_45m.csv actualizado con todos los años.')
else:
    print('No se encontraron archivos para unir.')

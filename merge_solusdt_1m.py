import pandas as pd

# Archivos de entrada
file1 = 'data/SOLUSDT_1m_2018-01-01_to_2023-12-31.csv'
file2 = 'data/SOLUSDT_1m_2024-01-01_to_2025-12-31.csv'
output_file = 'data/SOLUSDT_1m_2018-01-01_to_2025-12-31.csv'

# Leer ambos archivos
print('Leyendo', file1)
df1 = pd.read_csv(file1)
print('Leyendo', file2)
df2 = pd.read_csv(file2)

# Unir y eliminar duplicados
print('Concatenando...')
df = pd.concat([df1, df2], ignore_index=True)
df = df.drop_duplicates(subset=['timestamp'])

# Ordenar por timestamp
print('Ordenando...')
df = df.sort_values('timestamp')

# Guardar archivo final
print('Guardando', output_file)
df.to_csv(output_file, index=False)
print('Archivo generado:', output_file)

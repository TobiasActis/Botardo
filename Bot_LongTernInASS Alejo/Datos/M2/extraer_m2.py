"""
Script para extraer datos de M2 Money Stock desde FRED
y convertirlos a JSON mes por mes.
"""

import pandas as pd
import json
from datetime import datetime

# URL para descargar los datos en formato CSV desde FRED
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"

print("Descargando datos de M2 desde FRED...")

try:
    # Leer los datos directamente desde FRED
    df = pd.read_csv(url)
    
    # Mostrar información básica
    print(f"\n[OK] Datos descargados exitosamente")
    print(f"  Total de registros: {len(df)}")
    print(f"  Periodo: {df.iloc[0, 0]} a {df.iloc[-1, 0]}")
    print(f"  Columnas: {list(df.columns)}")
    
    # Renombrar columnas para mayor claridad
    df.columns = ['fecha', 'valor']
    
    # Convertir fecha a formato datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Eliminar valores NaN
    df = df.dropna()
    
    # Crear el JSON en formato SIMPLE (solo fechas y valores)
    datos_json = {
        "fechas": df['fecha'].dt.strftime("%Y-%m-%d").tolist(),
        "valores": df['valor'].tolist()
    }
    
    # Guardar el JSON
    archivo_salida = "M2.json"
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(datos_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Archivo JSON creado exitosamente: {archivo_salida}")
    print(f"\nEstadisticas:")
    print(f"  Valor minimo: ${df['valor'].min():,.2f} miles de millones")
    print(f"  Valor maximo: ${df['valor'].max():,.2f} miles de millones")
    print(f"  Valor promedio: ${df['valor'].mean():,.2f} miles de millones")
    print(f"  Ultimo valor: ${df['valor'].iloc[-1]:,.2f} miles de millones ({df['fecha'].iloc[-1].strftime('%Y-%m-%d')})")
    
except Exception as e:
    print(f"\n[ERROR] Error al descargar o procesar los datos: {e}")
    print("\nIntentando metodo alternativo...")
    
    # Método alternativo: usar la API de FRED (requiere requests)
    try:
        import requests
        
        # Descargar el CSV
        response = requests.get(url)
        response.raise_for_status()
        
        # Guardar temporalmente
        with open('temp.csv', 'wb') as f:
            f.write(response.content)
        
        # Procesar el archivo temporal
        df = pd.read_csv('temp.csv')
        print(f"[OK] Datos descargados con metodo alternativo")
        
        # Repetir el procesamiento...
        # (el mismo codigo de arriba)
        
    except Exception as e2:
        print(f"[ERROR] Error con metodo alternativo: {e2}")
        print("\nPor favor, descarga manualmente el archivo CSV desde:")
        print("https://fred.stlouisfed.org/series/M2SL")
        print("Y ejecuta este script nuevamente en el mismo directorio.")


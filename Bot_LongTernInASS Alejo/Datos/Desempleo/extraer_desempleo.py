"""
Script para extraer datos de la Tasa de Desempleo desde FRED
y convertirlos a JSON mes por mes.
"""

import pandas as pd
import json
from datetime import datetime

# URL para descargar los datos en formato CSV desde FRED
# UNRATE: Unemployment Rate
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"

print("Descargando datos de Tasa de Desempleo desde FRED...")

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
    df_original = df.copy()
    df = df.dropna()
    
    print(f"  Registros con datos: {len(df)} (eliminados {len(df_original) - len(df)} valores NaN)")
    
    # Crear el JSON en formato SIMPLE (solo fechas y valores)
    datos_json = {
        "fechas": df['fecha'].dt.strftime("%Y-%m-%d").tolist(),
        "valores": df['valor'].tolist()
    }
    
    # Guardar el JSON
    archivo_salida = "Desempleo.json"
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(datos_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Archivo JSON creado exitosamente: {archivo_salida}")
    print(f"\nEstadisticas (porcentaje de desempleo):")
    print(f"  Valor minimo: {df['valor'].min():.1f}%")
    print(f"  Valor maximo: {df['valor'].max():.1f}%")
    print(f"  Valor promedio: {df['valor'].mean():.1f}%")
    print(f"  Ultimo valor: {df['valor'].iloc[-1]:.1f}% ({df['fecha'].iloc[-1].strftime('%B %Y')})")
    
    # Mostrar últimos 12 meses
    print(f"\nUltimos 12 meses:")
    for i in range(min(12, len(df))):
        idx = -(12-i)
        fecha = df.iloc[idx]['fecha']
        valor = df.iloc[idx]['valor']
        print(f"  {fecha.strftime('%b %Y')}: {valor:.1f}%")
    
except Exception as e:
    print(f"\n[ERROR] Error al descargar o procesar los datos: {e}")
    print("\nIntentando metodo alternativo...")
    
    # Método alternativo: usar requests
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
        
    except Exception as e2:
        print(f"[ERROR] Error con metodo alternativo: {e2}")
        print("\nPor favor, descarga manualmente el archivo CSV desde:")
        print("https://fred.stlouisfed.org/series/UNRATE")
        print("Y ejecuta este script nuevamente en el mismo directorio.")


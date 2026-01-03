"""
Calcular retorno anualizado con risk dinámico
"""
import pandas as pd
import numpy as np

trades = pd.read_csv('ny_orderblock_trades.csv')
trades['date'] = pd.to_datetime(trades['date'])
trades['year'] = trades['date'].dt.year

print("=" * 70)
print("RETORNO ANUALIZADO - RISK DINÁMICO (NY=2%, EU=1%)")
print("=" * 70)

capital_inicial = 100
capital_final = trades['capital'].iloc[-1]
fecha_inicial = trades['date'].iloc[0]
fecha_final = trades['date'].iloc[-1]

# Calcular años exactos
dias_totales = (fecha_final - fecha_inicial).days
años_exactos = dias_totales / 365.25

# CAGR (Compound Annual Growth Rate)
cagr = (((capital_final / capital_inicial) ** (1 / años_exactos)) - 1) * 100

print(f"\nCapital inicial: ${capital_inicial:,.2f}")
print(f"Capital final: ${capital_final:,.2f}")
print(f"Multiplicador: {capital_final/capital_inicial:.2f}x")
print(f"\nPeríodo: {fecha_inicial.strftime('%Y-%m-%d')} a {fecha_final.strftime('%Y-%m-%d')}")
print(f"Duración: {años_exactos:.2f} años")
print(f"\nRetorno total: +{((capital_final - capital_inicial) / capital_inicial * 100):.2f}%")
print(f"**RETORNO ANUALIZADO (CAGR): {cagr:.2f}% por año**")

# Comparar con S&P500
sp500_anual = 10  # Promedio histórico S&P500
print(f"\nVs. S&P500 (~{sp500_anual}%/año): {cagr/sp500_anual:.1f}x mejor")

# Proyección a futuro
print("\n" + "=" * 70)
print("PROYECCIÓN - Si mantiene este ritmo:")
print("=" * 70)

capital_actual = capital_final
for años_futuros in [1, 2, 3, 5]:
    capital_futuro = capital_actual * ((1 + cagr/100) ** años_futuros)
    print(f"En {años_futuros} {'año' if años_futuros == 1 else 'años'}: ${capital_futuro:,.2f} ({capital_futuro/capital_actual:.2f}x)")

# Desglose por año
print("\n" + "=" * 70)
print("RETORNO POR AÑO INDIVIDUAL")
print("=" * 70)
print(f"{'Año':<8} {'Capital Inicio':<15} {'Capital Fin':<15} {'Retorno %':<12}")
print("-" * 70)

for year in range(2018, 2026):
    year_trades = trades[trades['year'] == year]
    if len(year_trades) == 0:
        continue
    
    cap_inicio = year_trades.iloc[0]['capital'] - year_trades.iloc[0]['pnl']
    cap_fin = year_trades.iloc[-1]['capital']
    retorno_year = ((cap_fin - cap_inicio) / cap_inicio) * 100
    
    print(f"{year:<8} ${cap_inicio:<14,.2f} ${cap_fin:<14,.2f} {retorno_year:+<12.2f}")

# Promedio simple
retornos_anuales = []
for year in range(2018, 2026):
    year_trades = trades[trades['year'] == year]
    if len(year_trades) == 0:
        continue
    cap_inicio = year_trades.iloc[0]['capital'] - year_trades.iloc[0]['pnl']
    cap_fin = year_trades.iloc[-1]['capital']
    retorno_year = ((cap_fin - cap_inicio) / cap_inicio) * 100
    retornos_anuales.append(retorno_year)

promedio_simple = np.mean(retornos_anuales)
mediana = np.median(retornos_anuales)
mejor_año = max(retornos_anuales)
peor_año = min(retornos_anuales)

print("\n" + "=" * 70)
print("ESTADÍSTICAS")
print("=" * 70)
print(f"Promedio simple: {promedio_simple:.2f}% por año")
print(f"Mediana: {mediana:.2f}% por año")
print(f"Mejor año: {mejor_año:.2f}%")
print(f"Peor año: {peor_año:.2f}%")
print(f"CAGR (compuesto): {cagr:.2f}% por año")
print(f"\nConsistencia: {'Alta' if np.std(retornos_anuales) < 30 else 'Media' if np.std(retornos_anuales) < 50 else 'Baja'}")
print(f"Desviación estándar: {np.std(retornos_anuales):.2f}%")

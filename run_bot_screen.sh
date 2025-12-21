#!/bin/bash
# Script para lanzar el bot en background usando screen y que corra sin parar

cd /root/Botardo
source venv/bin/activate

while true; do
  screen -dmS botardo bash -c 'python botardo.py --data data/SYMBOL_15m_2018-01-01_to_2025-12-18.csv --assets BTCUSDT,SOLUSDT,ADAUSDT --use-binance-balance --start 2018-01-01 --end 2025-12-18'
  echo "Bot corriendo en screen. Usá 'screen -r botardo' para ver la sesión."
  # Esperar a que el proceso termine antes de reiniciar
  sleep 60
  # Si el proceso terminó, reinicia después de 1 minuto
  screen -ls | grep botardo && screen -S botardo -X quit
  sleep 5
  echo "Reiniciando bot..."
done

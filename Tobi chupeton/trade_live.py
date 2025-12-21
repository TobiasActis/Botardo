def set_leverage(symbol, leverage, api_key, api_secret):
    url = f"{TESTNET_URL}/fapi/v1/leverage"
    # Obtener timestamp del servidor de Binance
    try:
        server_time_resp = requests.get(f"{TESTNET_URL}/fapi/v1/time")
        server_time = server_time_resp.json()["serverTime"]
    except Exception as e:
        logger.warning(f"No se pudo obtener hora del servidor, usando hora local. Error: {e}")
        server_time = int(time.time() * 1000)
    params = {
        'symbol': symbol,
        'leverage': leverage,
        'timestamp': server_time,
        'recvWindow': 20000
    }
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature
    headers = {'X-MBX-APIKEY': api_key}
    response = requests.post(url, params=params, headers=headers)
    try:
        data = response.json()
    except Exception:
        data = response.text
    logger.info(f"Respuesta set_leverage: {response.status_code} {data}")
    return response.status_code, data
"""
trade_live.py - Ejemplo de bot en tiempo real para Binance Testnet
Utiliza la lógica de señales de Botardo para operar BTCUSDT en futuros testnet.
"""

import time
from binance.client import Client
from binance.enums import *
import pandas as pd
from loguru import logger
from botardo import Botardo
import requests
import hmac
import hashlib
import urllib.parse
import csv
import os

# Obtener balance USDT disponible en Binance Testnet
def get_available_balance(client, asset="USDT"):
    try:
        balance = client.futures_account_balance()
        for b in balance:
            if b['asset'] == asset:
                return float(b['availableBalance'])
    except Exception as e:
        logger.error(f"No se pudo obtener balance disponible: {e}")
    return None

# --- Trailing Stop Config ---
TRAILING_ENABLED = True
TRAILING_PCT = 0.5 / 100  # 0.5% trailing stop

# --- Mejora RR Ratio ---
RR_RATIO = 2.2  # Aumenta el take profit respecto al stop loss

# Utilidad para firmar y enviar orden de mercado con requests
def enviar_orden_market(symbol, side, quantity, api_key, api_secret):
    url = f"{TESTNET_URL}/fapi/v1/order"
    # Obtener timestamp del servidor de Binance
    try:
        server_time_resp = requests.get(f"{TESTNET_URL}/fapi/v1/time")
        server_time = server_time_resp.json()["serverTime"]
        logger.info(f"Timestamp de servidor Binance: {server_time}")
    except Exception as e:
        logger.warning(f"No se pudo obtener hora del servidor, usando hora local. Error: {e}")
        server_time = int(time.time() * 1000)
    params = {
        'symbol': symbol,
        'side': side,
        'type': 'MARKET',
        'quantity': quantity,
        'timestamp': server_time,
        'recvWindow': 20000  # 20 segundos de margen
    }
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature
    headers = {'X-MBX-APIKEY': api_key}
    response = requests.post(url, params=params, headers=headers)
    try:
        data = response.json()
    except Exception:
        data = response.text
    logger.info(f"Respuesta orden MARKET: {response.status_code} {data}")
    return response.status_code, data

API_KEY = "fVUABjXeCWY853Q6are1nXQGYc2h7DVVxmmD5uw6PfMaRfKqFQkB6lU8CcKPNdso"
API_SECRET = "rJ5PekO9paYsijLsGITdl6e6gbjV3akRuM3rtwpLVZWz9fZRaYFmGNn1hjB3Fguf"
TESTNET_URL = "https://testnet.binancefuture.com"


# Multi-asset config
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK = 300  # Cantidad de velas a analizar
LEVERAGE = 5
RISK = 0.04


def crear_cliente():
    client = Client(API_KEY, API_SECRET)
    client.FUTURES_URL = TESTNET_URL
    client.API_URL = TESTNET_URL
    client.FUTURES_API_VERSION = "v1"
    return client

# Utilidad para firmar y enviar orden de mercado con requests
    url = f"{TESTNET_URL}/fapi/v1/order"
    # Obtener timestamp del servidor de Binance
    try:
        server_time_resp = requests.get(f"{TESTNET_URL}/fapi/v1/time")
        server_time = server_time_resp.json()["serverTime"]
        logger.info(f"Timestamp de servidor Binance: {server_time}")
    except Exception as e:
        logger.warning(f"No se pudo obtener hora del servidor, usando hora local. Error: {e}")
        server_time = int(time.time() * 1000)
    params = {
        'symbol': symbol,
        'side': side,
        'type': 'MARKET',
        'quantity': quantity,
        'timestamp': server_time,
        'recvWindow': 20000  # 20 segundos de margen
    }
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature
    headers = {'X-MBX-APIKEY': api_key}
    response = requests.post(url, params=params, headers=headers)
    try:
        data = response.json()
    except Exception:
        data = response.text
    logger.info(f"Respuesta orden MARKET: {response.status_code} {data}")
    return response.status_code, data

def test_ping_exchangeinfo():
    logger.info("Probando conectividad con Binance Testnet...")
    try:
        ping = requests.get(f"{TESTNET_URL}/fapi/v1/ping")
        logger.info(f"Ping: {ping.status_code} {ping.text}")
    except Exception as e:
        logger.error(f"Error en ping: {e}")
    try:
        exch = requests.get(f"{TESTNET_URL}/fapi/v1/exchangeInfo")
        logger.info(f"exchangeInfo: {exch.status_code} {exch.text[:200]}")
    except Exception as e:
        logger.error(f"Error en exchangeInfo: {e}")



client = crear_cliente()
logger.info("Conectado a Binance Testnet")
import traceback
# Diccionario para manejar posiciones por símbolo
live_positions = {symbol: None for symbol in SYMBOLS}
bot_instances = {symbol: Botardo(capital=0, risk_per_trade=RISK, leverage=LEVERAGE) for symbol in SYMBOLS}
for bot in bot_instances.values():
    bot.rr_ratio = RR_RATIO
cycle = 0
while True:
    cycle += 1
    logger.info(f"\n--- CICLO {cycle} ---")
    for SYMBOL in SYMBOLS:
        try:
            logger.info(f"\n[{SYMBOL}] Descargando velas recientes...")
            url_klines = f"{TESTNET_URL}/fapi/v1/klines"
            params_klines = {
                'symbol': SYMBOL,
                'interval': '15m',
                'limit': LOOKBACK
            }
            response = requests.get(url_klines, params=params_klines)
            logger.info(f"[{SYMBOL}] Respuesta klines: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"[{SYMBOL}] Error HTTP al pedir klines: {response.status_code} {response.text}")
                continue
            klines = response.json()
            logger.info(f"[{SYMBOL}] Cantidad de velas recibidas: {len(klines)}")
            if not isinstance(klines, list):
                logger.error(f"[{SYMBOL}] Respuesta inesperada de klines: {klines}")
                continue
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'])
            logger.info(f"[{SYMBOL}] DataFrame creado, shape: {df.shape}")
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
            logger.info(f"[{SYMBOL}] Calculando indicadores y señales...")
            bot = bot_instances[SYMBOL]
            df = bot.calculate_indicators(df)
            logger.info(f"[{SYMBOL}] Indicadores calculados.")
            last = df.iloc[-1]
            logger.info(f"[{SYMBOL}] Último cierre: {last['close']}")

            signals_file = f"signals_detected_{SYMBOL}.csv"
            signals_exists = os.path.exists(signals_file)
            def log_signal(signal_type, price, timestamp):
                with open(signals_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not signals_exists and f.tell() == 0:
                        writer.writerow(["timestamp", "signal", "price"])
                    writer.writerow([timestamp, signal_type, price])

            # Si no hay posición abierta, buscar señal
            if live_positions[SYMBOL] is None:
                logger.info(f"[{SYMBOL}] Buscando señal de entrada...")
                set_leverage(SYMBOL, LEVERAGE, API_KEY, API_SECRET)
                # Lógica de señales como antes (ejemplo: usar columnas del DataFrame generadas por calculate_indicators)
                long_signal = df.iloc[-1]["long_signal"] if "long_signal" in df.columns else False
                short_signal = df.iloc[-1]["short_signal"] if "short_signal" in df.columns else False
                if long_signal:
                    logger.success(f"[{SYMBOL}] ¡SEÑAL LONG DETECTADA! Enviando orden...")
                    log_signal("LONG", last['close'], last['timestamp'])
                    entry_price = last['close']
                    sl_price = entry_price * (1 - bot.sl_pct)
                    tp_price = entry_price * (1 + bot.sl_pct * bot.rr_ratio)
                    trailing_active = False
                    real_capital = get_available_balance(client, asset="USDT")
                    if real_capital is None:
                        logger.error(f"[{SYMBOL}] No se pudo obtener el balance real, usando 500 USDT por defecto.")
                        real_capital = 500
                    bot.capital = real_capital / len(SYMBOLS)
                    risk_amount = bot.capital * bot.risk_per_trade
                    position_size = round((risk_amount * bot.leverage) / abs(entry_price - sl_price), 4)
                    # Ajustar precisión según el símbolo
                    SYMBOL_PRECISION = {
                        "BTCUSDT": 3,
                        "ETHUSDT": 3,
                        "SOLUSDT": 2,
                        "ADAUSDT": 0
                    }
                    qty = round(position_size, SYMBOL_PRECISION.get(SYMBOL, 3))
                    status, data = enviar_orden_market(SYMBOL, 'BUY', qty, API_KEY, API_SECRET)
                    if status == 200 and isinstance(data, dict) and 'orderId' in data:
                        logger.success(f"[{SYMBOL}] ORDEN LONG ENVIADA: {data['orderId']}")
                    else:
                        logger.error(f"[{SYMBOL}] Error al enviar ORDEN LONG: {data}")
                    if TRAILING_ENABLED:
                        logger.info(f"[{SYMBOL}] Trailing Stop activado: {TRAILING_PCT*100:.2f}%")
                        trailing_active = True
                        trailing_sl = sl_price
                    logger.info(f"[{SYMBOL}] SL/TP gestionados por el bot: SL={sl_price}, TP={tp_price}")
                    live_positions[SYMBOL] = {
                        'side': 'LONG',
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'size': position_size,
                        'trailing_active': trailing_active if TRAILING_ENABLED else False,
                        'trailing_sl': trailing_sl if TRAILING_ENABLED else None
                    }
                elif short_signal:
                    logger.success(f"[{SYMBOL}] ¡SEÑAL SHORT DETECTADA! Enviando orden...")
                    log_signal("SHORT", last['close'], last['timestamp'])
                    entry_price = last['close']
                    sl_price = entry_price * (1 + bot.sl_pct)
                    tp_price = entry_price * (1 - bot.sl_pct * bot.rr_ratio)
                    trailing_active = False
                    real_capital = get_available_balance(client, asset="USDT")
                    if real_capital is None:
                        logger.error(f"[{SYMBOL}] No se pudo obtener el balance real, usando 500 USDT por defecto.")
                        real_capital = 500
                    bot.capital = real_capital / len(SYMBOLS)
                    risk_amount = bot.capital * bot.risk_per_trade
                    position_size = round((risk_amount * bot.leverage) / abs(entry_price - sl_price), 4)
                    # Ajustar precisión según el símbolo
                    SYMBOL_PRECISION = {
                        "BTCUSDT": 3,
                        "ETHUSDT": 3,
                        "SOLUSDT": 2,
                        "ADAUSDT": 0
                    }
                    qty = round(position_size, SYMBOL_PRECISION.get(SYMBOL, 3))
                    status, data = enviar_orden_market(SYMBOL, 'SELL', qty, API_KEY, API_SECRET)
                    if status == 200 and isinstance(data, dict) and 'orderId' in data:
                        logger.success(f"[{SYMBOL}] ORDEN SHORT ENVIADA: {data['orderId']}")
                    else:
                        logger.error(f"[{SYMBOL}] Error al enviar ORDEN SHORT: {data}")
                    if TRAILING_ENABLED:
                        logger.info(f"[{SYMBOL}] Trailing Stop activado: {TRAILING_PCT*100:.2f}%")
                        trailing_active = True
                        trailing_sl = sl_price
                    logger.info(f"[{SYMBOL}] SL/TP gestionados por el bot: SL={sl_price}, TP={tp_price}")
                    live_positions[SYMBOL] = {
                        'side': 'SHORT',
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'size': position_size,
                        'trailing_active': trailing_active if TRAILING_ENABLED else False,
                        'trailing_sl': trailing_sl if TRAILING_ENABLED else None
                    }
                else:
                    logger.info(f"[{SYMBOL}] No se detectó señal de entrada en este ciclo.")
            else:
                logger.info(f"[{SYMBOL}] Monitoreando posición abierta: {live_positions[SYMBOL]}")
                current_price = last['close']
                logger.info(f"[{SYMBOL}] Precio actual: {current_price} | SL: {live_positions[SYMBOL]['sl_price']} | TP: {live_positions[SYMBOL]['tp_price']}")
                if live_positions[SYMBOL]['side'] == 'LONG':
                    if current_price <= live_positions[SYMBOL]['sl_price']:
                        logger.warning(f"[{SYMBOL}] ¡SL LONG ALCANZADO! Cerrando posición...")
                        qty = round(live_positions[SYMBOL]['size'], 3)
                        enviar_orden_market(SYMBOL, 'SELL', qty, API_KEY, API_SECRET)
                        live_positions[SYMBOL] = None
                    elif current_price >= live_positions[SYMBOL]['tp_price']:
                        logger.success(f"[{SYMBOL}] ¡TP LONG ALCANZADO! Cerrando posición...")
                        qty = round(live_positions[SYMBOL]['size'], 3)
                        enviar_orden_market(SYMBOL, 'SELL', qty, API_KEY, API_SECRET)
                        live_positions[SYMBOL] = None
                    else:
                        logger.info(f"[{SYMBOL}] LONG: posición sigue abierta.")
                elif live_positions[SYMBOL]['side'] == 'SHORT':
                    if current_price >= live_positions[SYMBOL]['sl_price']:
                        logger.warning(f"[{SYMBOL}] ¡SL SHORT ALCANZADO! Cerrando posición...")
                        qty = round(live_positions[SYMBOL]['size'], 3)
                        enviar_orden_market(SYMBOL, 'BUY', qty, API_KEY, API_SECRET)
                        live_positions[SYMBOL] = None
                    elif current_price <= live_positions[SYMBOL]['tp_price']:
                        logger.success(f"[{SYMBOL}] ¡TP SHORT ALCANZADO! Cerrando posición...")
                        qty = round(live_positions[SYMBOL]['size'], 3)
                        enviar_orden_market(SYMBOL, 'BUY', qty, API_KEY, API_SECRET)
                        live_positions[SYMBOL] = None
                    else:
                        logger.info(f"[{SYMBOL}] SHORT: posición sigue abierta.")
        except Exception as e:
            logger.critical(f"[{SYMBOL}] ERROR de conexión o API: {e}")
            logger.error(traceback.format_exc())
            logger.info(f"[{SYMBOL}] Reintentando en 15 segundos y re-conectando...")
            time.sleep(15)
            client = crear_cliente()
    logger.info("Esperando 60 segundos para el próximo ciclo...")
    time.sleep(60)

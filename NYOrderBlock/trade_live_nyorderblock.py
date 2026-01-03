"""
trade_live_nyorderblock.py - Bot en tiempo real para NY+EU OrderBlock en Binance Testnet
Opera BTCUSDT detectando orderblocks en sesiones EU (01:00-06:00) y NY (06:45-11:00) NY timezone
"""

import time
from binance.client import Client
from binance.enums import *
import pandas as pd
from loguru import logger
import requests
import hmac
import hashlib
import urllib.parse
import csv
import os
from datetime import datetime, timezone
import pytz

# ===================== CONFIGURACIÓN =====================
API_KEY = "79AxvQcuF1FybQ6FtSJQitwTJmH6BqA8u0i3He4Yf2qLSDAqYludZWpxCu13k42e"
API_SECRET = "i6yWefGMoy7cpMLaL52wk0MeHSburGP6OkjBfjsL2svyyWKcIwYhdZzWCz9pTzGt"
TESTNET_URL = "https://testnet.binancefuture.com"

SYMBOL = "BTCUSDT"
LEVERAGE = 5
RISK_PER_TRADE = 0.02  # 2% de riesgo base por trade
LOOKBACK_5M = 500  # Velas de 5m para detectar orderblocks
LOOKBACK_1H = 200  # Velas de 1h para tendencia

# Sesiones en hora NY (UTC-5 en invierno, UTC-4 en verano)
EU_PRE_START = "01:00"
EU_PRE_END = "02:00"
EU_SESSION_START = "02:00"
EU_SESSION_END = "06:00"

NY_PRE_START = "06:45"
NY_PRE_END = "07:30"
NY_SESSION_START = "07:30"
NY_SESSION_END = "11:00"

# Filtros de calidad
MIN_CANDLE_RANGE_PCT = 0.15 / 100  # 0.15% rango mínimo
MIN_BODY_PCT = 0.50  # 50% del rango debe ser body


# ===================== UTILIDADES BINANCE =====================
def crear_cliente():
    """Crea cliente de Binance configurado para testnet"""
    client = Client(API_KEY, API_SECRET)
    client.FUTURES_URL = TESTNET_URL
    client.API_URL = TESTNET_URL
    client.FUTURES_API_VERSION = "v1"
    return client


def set_leverage(symbol, leverage, api_key, api_secret):
    """Configura el apalancamiento para el símbolo"""
    url = f"{TESTNET_URL}/fapi/v1/leverage"
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


def enviar_orden_market(symbol, side, quantity, api_key, api_secret):
    """Envía orden de mercado firmada con HMAC"""
    url = f"{TESTNET_URL}/fapi/v1/order"
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
    logger.info(f"Respuesta orden MARKET: {response.status_code} {data}")
    return response.status_code, data


def get_available_balance(api_key, api_secret, asset="USDT"):
    """Obtiene balance USDT disponible en cuenta usando requests directo"""
    url = f"{TESTNET_URL}/fapi/v2/balance"
    try:
        server_time_resp = requests.get(f"{TESTNET_URL}/fapi/v1/time")
        server_time = server_time_resp.json()["serverTime"]
    except Exception as e:
        logger.warning(f"No se pudo obtener hora del servidor: {e}")
        server_time = int(time.time() * 1000)
    
    params = {
        'timestamp': server_time,
        'recvWindow': 20000
    }
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature
    headers = {'X-MBX-APIKEY': api_key}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            balance_list = response.json()
            for b in balance_list:
                if b['asset'] == asset:
                    return float(b['availableBalance'])
        else:
            logger.error(f"Error al obtener balance: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"No se pudo obtener balance disponible: {e}")
    return None


def descargar_klines(symbol, interval, limit):
    """Descarga velas históricas de Binance Testnet"""
    url_klines = f"{TESTNET_URL}/fapi/v1/klines"
    params_klines = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url_klines, params=params_klines)
    
    if response.status_code != 200:
        logger.error(f"Error HTTP al pedir klines: {response.status_code} {response.text}")
        return None
    
    klines = response.json()
    if not isinstance(klines, list):
        logger.error(f"Respuesta inesperada de klines: {klines}")
        return None
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    
    return df


# ===================== LÓGICA ORDERBLOCK =====================
def detectar_orderblock(df_5m, df_1h, session_name):
    """
    Detecta orderblocks válidos en la sesión especificada
    Retorna: dict con 'direction', 'entry_level', 'sl', 'candle_info' o None
    """
    # Convertir a timezone NY
    ny_tz = pytz.timezone('America/New_York')
    df_5m['timestamp_ny'] = df_5m['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ny_tz)
    
    # Calcular EMAs en 1h para tendencia
    df_1h = df_1h.copy()
    df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
    df_1h['ema50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
    df_1h['timestamp_ny'] = df_1h['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ny_tz)
    
    # Determinar rango de pre-sesión y sesión según session_name
    if session_name == "EU":
        pre_start, pre_end = EU_PRE_START, EU_PRE_END
        sess_start, sess_end = EU_SESSION_START, EU_SESSION_END
    else:  # NY
        pre_start, pre_end = NY_PRE_START, NY_PRE_END
        sess_start, sess_end = NY_SESSION_START, NY_SESSION_END
    
    # Filtrar velas de pre-sesión (últimas 24h máximo)
    now_ny = df_5m['timestamp_ny'].iloc[-1]
    yesterday = now_ny - pd.Timedelta(days=1)
    
    pre_session = df_5m[
        (df_5m['timestamp_ny'] >= yesterday) &
        (df_5m['timestamp_ny'].dt.time >= pd.to_datetime(pre_start).time()) &
        (df_5m['timestamp_ny'].dt.time < pd.to_datetime(pre_end).time())
    ]
    
    if pre_session.empty:
        return None
    
    # Encontrar la última vela de pre-sesión
    last_candle = pre_session.iloc[-1]
    
    # Aplicar filtros de calidad
    candle_range = last_candle['high'] - last_candle['low']
    candle_range_pct = candle_range / last_candle['close']
    if candle_range_pct < MIN_CANDLE_RANGE_PCT:
        return None
    
    body = abs(last_candle['close'] - last_candle['open'])
    body_pct = body / candle_range if candle_range > 0 else 0
    if body_pct < MIN_BODY_PCT:
        return None
    
    # Detectar tipo de vela
    is_bearish = last_candle['close'] < last_candle['open']
    is_bullish = last_candle['close'] > last_candle['open']
    
    if not is_bearish and not is_bullish:
        return None
    
    # Verificar tendencia 1h
    # Buscar la vela de 1h más cercana a la última vela de pre-sesión
    closest_1h = None
    min_diff = float('inf')
    for idx, row_1h in df_1h.iterrows():
        diff = abs((last_candle['timestamp_ny'] - row_1h['timestamp_ny']).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_1h = row_1h
    
    if closest_1h is None or min_diff > 3600:  # Máximo 1h de diferencia
        return None
    
    trend_bullish = closest_1h['ema20'] > closest_1h['ema50']
    trend_bearish = closest_1h['ema20'] < closest_1h['ema50']
    
    # Solo operar si la dirección coincide con la tendencia
    if is_bullish and not trend_bullish:
        return None
    if is_bearish and not trend_bearish:
        return None
    
    # Definir entry, SL y dirección
    if is_bearish:
        entry_level = last_candle['low']
        sl = last_candle['high'] + 0.001
        direction = 'SELL'
    else:  # is_bullish
        entry_level = last_candle['high']
        sl = last_candle['low'] - 0.001
        direction = 'BUY'
    
    # Verificar si ya estamos en sesión de trading
    current_time_ny = now_ny.time()
    session_start_time = pd.to_datetime(sess_start).time()
    session_end_time = pd.to_datetime(sess_end).time()
    
    if not (session_start_time <= current_time_ny <= session_end_time):
        return None  # Aún no es hora de operar
    
    # Verificar si el precio actual ya tocó el entry level
    current_price = df_5m['close'].iloc[-1]
    if direction == 'SELL' and current_price < entry_level:
        return None  # Ya pasó el nivel
    if direction == 'BUY' and current_price > entry_level:
        return None  # Ya pasó el nivel
    
    return {
        'direction': direction,
        'entry_level': entry_level,
        'sl': sl,
        'candle_info': last_candle,
        'session': session_name
    }


def calcular_tp(entry_price, sl, direction):
    """Calcula TP con ratio 1:1 del riesgo"""
    risk = abs(entry_price - sl)
    if direction == 'SELL':
        return entry_price - risk
    else:  # BUY
        return entry_price + risk


# ===================== GESTIÓN DE POSICIONES =====================
live_position = None
signal_log_file = f"signals_detected_{SYMBOL}.csv"

def log_signal(signal_type, price, timestamp):
    """Registra señales detectadas en CSV"""
    signals_exists = os.path.exists(signal_log_file)
    with open(signal_log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not signals_exists:
            writer.writerow(["timestamp", "signal", "price", "session"])
        writer.writerow([timestamp, signal_type, price, ""])


def abrir_posicion(client, signal, balance):
    """Abre posición basada en la señal detectada"""
    global live_position
    
    direction = signal['direction']
    entry_price = signal['entry_level']
    sl = signal['sl']
    session = signal.get('session', 'NY')  # Sesión de la señal
    
    # Calcular TP (1:1)
    tp = calcular_tp(entry_price, sl, direction)
    
    # RISK DINÁMICO: NY=2%, EU=1%
    base_risk = RISK_PER_TRADE if session == 'NY' else RISK_PER_TRADE * 0.5
    risk_amount = balance * base_risk
    
    logger.info(f"📊 Sesión {session}: Risk = {base_risk*100:.2f}% del capital")
    
    # Calcular tamaño de posición
    risk = abs(entry_price - sl)
    position_size = (risk_amount * LEVERAGE) / risk if risk > 0 else 0
    qty = round(position_size, 3)  # BTCUSDT usa 3 decimales
    
    if qty <= 0:
        logger.error(f"Tamaño de posición inválido: {qty}")
        return False
    
    # Enviar orden
    side = 'SELL' if direction == 'SELL' else 'BUY'
    status, data = enviar_orden_market(SYMBOL, side, qty, API_KEY, API_SECRET)
    
    if status == 200 and isinstance(data, dict) and 'orderId' in data:
        logger.success(f"✅ ORDEN {direction} ENVIADA: OrderID={data['orderId']}, Size={qty}, Entry={entry_price}")
        log_signal(direction, entry_price, signal['candle_info']['timestamp_ny'])
        
        live_position = {
            'side': direction,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'size': qty,
            'session': signal['session'],
            'order_id': data['orderId']
        }
        logger.info(f"📊 SL={sl:.2f} | TP={tp:.2f} | Risk={risk_amount:.2f} USDT | Size={qty} BTC")
        return True
    else:
        logger.error(f"❌ Error al enviar orden {direction}: {data}")
        return False


def monitorear_posicion(df_5m):
    """Monitorea posición abierta y ejecuta SL/TP si se alcanza"""
    global live_position
    
    if live_position is None:
        return
    
    current_price = df_5m['close'].iloc[-1]
    direction = live_position['side']
    sl = live_position['sl']
    tp = live_position['tp']
    
    logger.info(f"🔍 Monitoreando {direction}: Price={current_price:.2f} | SL={sl:.2f} | TP={tp:.2f}")
    
    cerrar = False
    motivo = ""
    
    if direction == 'SELL':
        if current_price >= sl:
            cerrar = True
            motivo = "SL"
        elif current_price <= tp:
            cerrar = True
            motivo = "TP"
    else:  # BUY
        if current_price <= sl:
            cerrar = True
            motivo = "SL"
        elif current_price >= tp:
            cerrar = True
            motivo = "TP"
    
    if cerrar:
        # Cerrar posición
        qty = live_position['size']
        close_side = 'BUY' if direction == 'SELL' else 'SELL'
        status, data = enviar_orden_market(SYMBOL, close_side, qty, API_KEY, API_SECRET)
        
        if status == 200:
            pnl = 0
            if direction == 'SELL':
                pnl = (live_position['entry_price'] - current_price) * qty
            else:
                pnl = (current_price - live_position['entry_price']) * qty
            
            symbol_msg = "🎯" if motivo == "TP" else "🛑"
            logger.success(f"{symbol_msg} {motivo} ALCANZADO! Posición {direction} cerrada. PnL estimado: {pnl:.2f} USDT")
        else:
            logger.error(f"❌ Error al cerrar posición: {data}")
        
        live_position = None


# ===================== MAIN LOOP =====================
def main():
    global live_position
    
    logger.info("🚀 Iniciando NY+EU OrderBlock Bot en Binance Testnet...")
    logger.info("📊 Risk Dinámico: NY=2%, EU=1%")
    
    # Configurar leverage
    set_leverage(SYMBOL, LEVERAGE, API_KEY, API_SECRET)
    
    client = crear_cliente()
    
    logger.info(f"✅ Conectado a Binance Testnet | Symbol: {SYMBOL} | Leverage: {LEVERAGE}x")
    
    cycle = 0
    
    while True:
        try:
            cycle += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 CICLO {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*60}")
            
            # Obtener balance
            balance = get_available_balance(API_KEY, API_SECRET, asset="USDT")
            if balance is None:
                logger.warning("⚠️ No se pudo obtener balance, usando 500 USDT por defecto")
                balance = 500
            else:
                logger.info(f"💰 Balance disponible: {balance:.2f} USDT")
            
            # Descargar datos
            logger.info("📥 Descargando datos de mercado...")
            df_5m = descargar_klines(SYMBOL, '5m', LOOKBACK_5M)
            df_1h = descargar_klines(SYMBOL, '1h', LOOKBACK_1H)
            
            if df_5m is None or df_1h is None:
                logger.error("❌ Error al descargar datos, reintentando en 60s...")
                time.sleep(60)
                continue
            
            logger.info(f"✅ Datos descargados: {len(df_5m)} velas 5m, {len(df_1h)} velas 1h")
            
            # Si no hay posición abierta, buscar señales
            if live_position is None:
                logger.info("🔎 Buscando señales de entrada...")
                
                # Buscar en ambas sesiones
                for session in ["EU", "NY"]:
                    signal = detectar_orderblock(df_5m, df_1h, session)
                    
                    if signal:
                        logger.success(f"🎯 ¡SEÑAL {signal['direction']} DETECTADA en sesión {session}!")
                        logger.info(f"   Entry: {signal['entry_level']:.2f} | SL: {signal['sl']:.2f}")
                        
                        # Abrir posición
                        if abrir_posicion(client, signal, balance):
                            break  # Solo una posición a la vez
                
                if live_position is None:
                    logger.info("😴 No se detectó señal de entrada en este ciclo")
            
            else:
                # Monitorear posición abierta
                logger.info(f"📍 Posición abierta: {live_position['side']} en sesión {live_position['session']}")
                monitorear_posicion(df_5m)
            
            # Esperar antes del próximo ciclo (adaptativo)
            if live_position is None:
                wait_time = 300  # 5 minutos sin posición (nueva vela 5m)
                logger.info("⏳ Sin posición abierta. Esperando 5 minutos para el próximo ciclo...")
            else:
                wait_time = 60  # 1 minuto con posición (monitoreo cercano de SL/TP)
                logger.info("⏳ Posición activa. Esperando 1 minuto para monitoreo...")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            logger.info("🛑 Bot detenido por el usuario")
            break
        except Exception as e:
            logger.critical(f"💥 ERROR CRÍTICO: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("🔄 Reconectando en 30 segundos...")
            time.sleep(30)
            client = crear_cliente()


if __name__ == "__main__":
    main()

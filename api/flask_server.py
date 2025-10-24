# src/api/flask_server.py
"""
Simple Flask API to control data fetching
POST /fetch_historical  -> start immediate historical fetch (sync)
POST /start_live       -> starts live poll (async thread)
POST /stop_live        -> stops live poll
GET  /status           -> list running polls
"""

from flask import Flask, request, jsonify
import threading
import os
import json
import time

from data.data_manager import (
    fetch_historical_binance,
    fetch_historical_yahoo,
    start_live_poll,
    stop_live_poll,
    list_live_polls,
    create_and_save_blocks,
    get_candles,
    read_settings
)

app = Flask(__name__)

# load default settings from config/settings.json if available
CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config", "settings.json")
CFG_PATH = os.path.normpath(CFG_PATH)
settings = {}
if os.path.exists(CFG_PATH):
    settings = read_settings(CFG_PATH)

@app.route("/fetch_historical", methods=["POST"])
def fetch_historical():
    """
    JSON body:
    {
        "source": "binance" | "yahoo",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "start": "2024-01-01",
        "end": "2024-06-01"
    }
    """
    p = request.json
    source = p.get("source", "binance")
    symbol = p.get("symbol", settings.get("default_symbol", "BTCUSDT"))
    interval = p.get("interval", settings.get("default_interval", "1m"))
    start = p.get("start", None)
    end = p.get("end", None)

    try:
        if source == "binance":
            df = fetch_historical_binance(symbol, interval, start, end)
        else:
            df = fetch_historical_yahoo(symbol, interval, start, end)
        return jsonify({"status": "ok", "rows": len(df)}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/start_live", methods=["POST"])
def start_live():
    p = request.json or {}
    source = p.get("source", "binance")
    symbol = p.get("symbol", settings.get("default_symbol", "BTCUSDT"))
    interval = p.get("interval", settings.get("default_interval", "1m"))
    try:
        res = start_live_poll(source, symbol, interval)
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/stop_live", methods=["POST"])
def stop_live():
    p = request.json or {}
    source = p.get("source", "binance")
    symbol = p.get("symbol", settings.get("default_symbol", "BTCUSDT"))
    interval = p.get("interval", settings.get("default_interval", "1m"))
    res = stop_live_poll(source, symbol, interval)
    return jsonify(res)

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"running": list_live_polls()})

@app.route("/create_blocks", methods=["POST"])
def create_blocks_endpoint():
    """
    Create training blocks from stored candles
    JSON:
    { "source":"binance","symbol":"BTCUSDT","interval":"1m","sequence_length":3 }
    """
    p = request.json or {}
    source = p.get("source","binance")
    symbol = p.get("symbol", settings.get("default_symbol","BTCUSDT"))
    interval = p.get("interval", settings.get("default_interval","1m"))
    seq_len = int(p.get("sequence_length", 3))
    out = create_and_save_blocks(source, symbol, interval, sequence_length=seq_len)
    return jsonify({"status":"ok","blocks_file": out})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

"""Risk-first live/dry-run bot for XAUUSD on MetaTrader 5.

The bot is intentionally conservative: it trades at most one position, avoids
martingale/grid recovery, enforces daily loss limits, and starts in dry-run mode
by default. It can use the trained model from ``model_trainer.py`` when present,
then confirms the signal with simple trend/RSI filters before sending an order.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import MetaTrader5 as mt5
import pandas as pd

import config
import data_handler


@dataclass(frozen=True)
class Signal:
    side: str
    confidence: float
    reason: str


TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

MAGIC = getattr(config, "GUARDIAN_MAGIC", 240507)
SYMBOL = getattr(config, "GUARDIAN_SYMBOL", config.SYMBOL)
ENTRY_TIMEFRAME = TIMEFRAMES[getattr(config, "GUARDIAN_TIMEFRAME", "M15")]
TREND_TIMEFRAME = TIMEFRAMES[getattr(config, "GUARDIAN_TREND_TIMEFRAME", "H1")]
DRY_RUN = getattr(config, "GUARDIAN_DRY_RUN", True)
MIN_MODEL_CONFIDENCE = getattr(config, "GUARDIAN_MIN_MODEL_CONFIDENCE", 0.64)
MIN_RULE_CONFIDENCE = getattr(config, "GUARDIAN_MIN_RULE_CONFIDENCE", 0.58)
RISK_PERCENT = getattr(config, "GUARDIAN_RISK_PERCENT", 0.35)
MAX_LOT = getattr(config, "GUARDIAN_MAX_LOT", 0.03)
MAX_SPREAD_POINTS = getattr(config, "GUARDIAN_MAX_SPREAD_POINTS", 180)
STOP_ATR_MULTIPLIER = getattr(config, "GUARDIAN_STOP_ATR_MULTIPLIER", 1.8)
TAKE_PROFIT_R_MULTIPLIER = getattr(config, "GUARDIAN_TAKE_PROFIT_R_MULTIPLIER", 1.4)
DAILY_LOSS_LIMIT_PERCENT = getattr(config, "GUARDIAN_DAILY_LOSS_LIMIT_PERCENT", 2.0)
COOLDOWN_MINUTES = getattr(config, "GUARDIAN_COOLDOWN_MINUTES", 45)
LOOP_SECONDS = getattr(config, "GUARDIAN_LOOP_SECONDS", 10)
BARS_NEEDED = getattr(config, "GUARDIAN_BARS_NEEDED", 240)


cooldown_until: datetime | None = None
last_closed_candle_time: int | None = None


def load_model():
    model_path = Path(config.MODEL_FILE_NAME)
    if not model_path.exists():
        print(f"Không thấy model {model_path}; bot sẽ chỉ dùng bộ lọc trend/RSI.")
        return None
    return joblib.load(model_path)


def ensure_symbol() -> bool:
    if not mt5.symbol_select(SYMBOL, True):
        print(f"Không chọn được symbol {SYMBOL}. Hãy kiểm tra tên mã vàng của broker.")
        return False
    return True


def fetch_rates(timeframe: int, count: int, start_pos: int = 1) -> pd.DataFrame | None:
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, start_pos, count)
    if rates is None or len(rates) < 80:
        print("Không đủ dữ liệu nến từ MT5.")
        return None
    df = pd.DataFrame(rates).rename(columns={"tick_volume": "volume"})
    return df[["time", "open", "high", "low", "close", "volume"]]


def add_guardian_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = data_handler.add_technical_indicators(df.copy())
    high_low = enriched["high"] - enriched["low"]
    high_close = (enriched["high"] - enriched["close"].shift()).abs()
    low_close = (enriched["low"] - enriched["close"].shift()).abs()
    enriched["ATR_14"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    enriched["EMA_50"] = enriched["close"].ewm(span=50, adjust=False).mean()
    enriched["EMA_200"] = enriched["close"].ewm(span=200, adjust=False).mean()
    enriched["ATR_14"] = enriched["ATR_14"].ffill().bfill()
    return enriched.ffill().bfill()


def closed_candle_time() -> int | None:
    rates = mt5.copy_rates_from_pos(SYMBOL, ENTRY_TIMEFRAME, 1, 1)
    if rates is None or len(rates) == 0:
        return None
    return int(rates[0]["time"])


def spread_points() -> float:
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if tick is None or info is None:
        return float("inf")
    return (tick.ask - tick.bid) / info.point


def open_bot_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return []
    return [position for position in positions if position.magic == MAGIC]


def today_profit() -> float:
    now = datetime.now()
    start = datetime(now.year, now.month, now.day)
    deals = mt5.history_deals_get(start, now)
    if not deals:
        return 0.0
    return sum(deal.profit + deal.swap + deal.commission for deal in deals if deal.magic == MAGIC)


def daily_loss_limit_hit() -> bool:
    account = mt5.account_info()
    if account is None or account.balance <= 0:
        return True
    loss_percent = abs(min(today_profit(), 0.0)) / account.balance * 100
    return loss_percent >= DAILY_LOSS_LIMIT_PERCENT


def model_signal(model, features: pd.DataFrame) -> Signal | None:
    if model is None or not hasattr(model, "predict_proba"):
        return None
    latest = features.iloc[-1:]
    probs = model.predict_proba(latest[model.feature_names_in_])[0]
    prob_sell, prob_buy = float(probs[0]), float(probs[1])
    if max(prob_buy, prob_sell) < MIN_MODEL_CONFIDENCE:
        return None
    side = "BUY" if prob_buy > prob_sell else "SELL"
    return Signal(side=side, confidence=max(prob_buy, prob_sell), reason="model")


def rule_signal(entry: pd.DataFrame, trend: pd.DataFrame) -> Signal | None:
    latest = entry.iloc[-1]
    trend_latest = trend.iloc[-1]
    bullish_trend = trend_latest["EMA_50"] > trend_latest["EMA_200"]
    bearish_trend = trend_latest["EMA_50"] < trend_latest["EMA_200"]
    rsi = float(latest["RSI_14"])

    if bullish_trend and latest["close"] > latest["EMA_50"] and 48 <= rsi <= 68:
        confidence = MIN_RULE_CONFIDENCE + min((rsi - 48) / 100, 0.12)
        return Signal("BUY", confidence, "trend_rsi")
    if bearish_trend and latest["close"] < latest["EMA_50"] and 32 <= rsi <= 52:
        confidence = MIN_RULE_CONFIDENCE + min((52 - rsi) / 100, 0.12)
        return Signal("SELL", confidence, "trend_rsi")
    return None


def combined_signal(model, entry: pd.DataFrame, trend: pd.DataFrame) -> Signal | None:
    model_view = model_signal(model, entry)
    rule_view = rule_signal(entry, trend)
    if model_view and rule_view and model_view.side == rule_view.side:
        confidence = (model_view.confidence * 0.65) + (rule_view.confidence * 0.35)
        return Signal(model_view.side, confidence, "model_confirmed_by_trend")
    if rule_view and model_view is None:
        return rule_view
    print("Không vào lệnh: model và trend/RSI chưa đồng thuận hoặc tín hiệu yếu.")
    return None


def lot_from_risk(stop_distance_price: float) -> float:
    account = mt5.account_info()
    info = mt5.symbol_info(SYMBOL)
    if account is None or info is None or stop_distance_price <= 0:
        return 0.01
    risk_money = account.balance * RISK_PERCENT / 100
    point_distance = stop_distance_price / info.point
    tick_size_in_points = max(info.trade_tick_size / info.point, 1.0)
    value_per_point_per_lot = info.trade_tick_value / tick_size_in_points
    if value_per_point_per_lot <= 0:
        return info.volume_min
    raw_lot = risk_money / max(point_distance * value_per_point_per_lot, 0.01)
    step = info.volume_step or 0.01
    lot = max(info.volume_min, min(raw_lot, MAX_LOT, info.volume_max))
    return round(math.floor(lot / step) * step, 2)


def order_request(signal: Signal, atr: float) -> dict | None:
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if tick is None or info is None:
        return None
    stop_distance = max(atr * STOP_ATR_MULTIPLIER, 300 * info.point)
    lot = lot_from_risk(stop_distance)
    if signal.side == "BUY":
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
        sl = price - stop_distance
        tp = price + stop_distance * TAKE_PROFIT_R_MULTIPLIER
    else:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
        sl = price + stop_distance
        tp = price - stop_distance * TAKE_PROFIT_R_MULTIPLIER
    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": MAGIC,
        "comment": f"Guardian {signal.reason} {signal.confidence:.2f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }


def place_order(request: dict, signal: Signal) -> None:
    if DRY_RUN:
        print(f"[DRY_RUN] {signal.side} {request['volume']} lot @ {request['price']:.2f}; "
              f"SL {request['sl']:.2f}; TP {request['tp']:.2f}; conf {signal.confidence:.2f}")
        return
    result = mt5.order_send(request)
    if result is None:
        print("MT5 không trả kết quả order_send.")
    elif result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f">>> ĐÃ VÀO {signal.side}: {request['volume']} lot | conf {signal.confidence:.2f}")
    else:
        print(f"Không vào được lệnh: {result.comment} (retcode {result.retcode})")


def run_bot() -> None:
    global cooldown_until, last_closed_candle_time
    if not mt5.initialize():
        print("Không kết nối được MT5.")
        return
    if not ensure_symbol():
        return
    model = load_model()
    mode = "DRY-RUN" if DRY_RUN else "LIVE"
    print(f"--- XAU GUARDIAN BOT đã bật ({mode}). Không grid, không martingale. ---")

    while True:
        try:
            now = datetime.now()
            if cooldown_until and now < cooldown_until:
                print(f"Bot đang nghỉ bảo vệ đến {cooldown_until:%H:%M:%S}.")
                time.sleep(LOOP_SECONDS)
                continue
            if daily_loss_limit_hit():
                cooldown_until = now + timedelta(minutes=COOLDOWN_MINUTES)
                print("Chạm giới hạn lỗ ngày; tạm dừng bot để bảo vệ vốn.")
                time.sleep(LOOP_SECONDS)
                continue
            candle_time = closed_candle_time()
            if candle_time is None or candle_time == last_closed_candle_time:
                time.sleep(LOOP_SECONDS)
                continue
            last_closed_candle_time = candle_time
            if spread_points() > MAX_SPREAD_POINTS:
                print("Spread đang cao; bỏ qua nến này.")
                continue
            if open_bot_positions():
                print("Đang có lệnh Guardian mở; không nhồi thêm.")
                continue

            entry_raw = fetch_rates(ENTRY_TIMEFRAME, BARS_NEEDED)
            trend_raw = fetch_rates(TREND_TIMEFRAME, BARS_NEEDED)
            if entry_raw is None or trend_raw is None:
                continue
            entry = add_guardian_indicators(entry_raw)
            trend = add_guardian_indicators(trend_raw)
            signal = combined_signal(model, entry, trend)
            if signal is None:
                continue
            request = order_request(signal, float(entry.iloc[-1]["ATR_14"]))
            if request is not None:
                place_order(request, signal)
        except Exception as exc:
            print(f"Lỗi vòng lặp Guardian: {exc}")
            time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    run_bot()

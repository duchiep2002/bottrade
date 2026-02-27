import MetaTrader5 as mt5
import pandas as pd
import joblib
import config
import data_handler
import time
from datetime import datetime, timedelta

# Biến toàn cục theo dõi thời gian nghỉ
cooldown_until = None

def get_lot_size(risk_percent, sl_points):
    account = mt5.account_info()
    if account is None: return 0.01
    risk_amount = account.balance * (risk_percent / 100)
    # Với Vàng 1 lot 1 point ~ 1$ (tùy sàn), công thức cơ bản:
    lot = risk_amount / sl_points 
    return round(max(0.01, min(lot, 0.5)), 2)

def check_circuit_breaker():
    """Hàm cầu chì: Kiểm tra lỗ 20% trong 20 phút"""
    global cooldown_until
    now = datetime.now()
    start_check = now - timedelta(minutes=config.CHECK_WINDOW_MINS)
    
    history = mt5.history_deals_get(start_check, now)
    if history is None: return False

    total_pl = sum(d.profit + d.swap + d.commission for d in history if d.magic == 123456)
    account = mt5.account_info()
    
    if account and total_pl < 0:
        loss_pct = (abs(total_pl) / account.balance) * 100
        if loss_pct >= config.LOSS_LIMIT_PERCENT:
            cooldown_until = datetime.now() + timedelta(minutes=config.COOLDOWN_MINUTES)
            return True
    return False

def close_everything(reason=""):
    """Xóa hết lệnh chờ và đóng vị thế đang chạy"""
    orders = mt5.orders_get(magic=123456)
    if orders:
        for o in orders: mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
    
    pos = mt5.positions_get(magic=123456)
    if pos:
        for p in pos:
            tick = mt5.symbol_info_tick(p.symbol)
            order_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
            mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume,
                "type": order_type, "position": p.ticket, "price": price, "magic": 123456,
                "comment": f"AI {reason}", "type_filling": mt5.ORDER_FILLING_IOC,
            })
    print(f"--- ĐÃ DỌN SẠCH TÀI KHOẢN. LÝ DO: {reason} ---")

def run_bot():
    global cooldown_until
    if not mt5.initialize(): return
    model = joblib.load(config.MODEL_FILE_NAME)
    print("--- BOT AI: ĐÃ KÍCH HOẠT CHẾ ĐỘ SINH TỒN ---")

    while True:
        try:
            # 1. Kiểm tra cầu chì
            if cooldown_until and datetime.now() < cooldown_until:
                print(f"Bot đang nghỉ bảo vệ... Còn {int((cooldown_until - datetime.now()).total_seconds())}s")
                time.sleep(30); continue

            if check_circuit_breaker():
                close_everything("CHÁY CẦU CHÌ 20%")
                continue

            # 2. Lấy dữ liệu và dự báo
            rates = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_M5, 0, 200)
            df = pd.DataFrame(rates)
            df = df.rename(columns={'tick_volume': 'volume'})
            df_features = data_handler.add_technical_indicators(df)
            latest = df_features.iloc[-1:]
            
            # AI phân tích độ tự tin
            probs = model.predict_proba(latest[model.feature_names_in_])[0]
            prob_sell, prob_buy = probs[0], probs[1]
            signal = "BUY" if prob_buy > prob_sell else "SELL"
            max_conf = max(prob_buy, prob_sell)

            # 3. Quản lý lệnh đang chạy (Đóng linh hoạt)
            pos_now = mt5.positions_get(magic=123456)
            if pos_now:
                p = pos_now[0]
                is_reverse = (p.type == mt5.ORDER_TYPE_BUY and signal == "SELL") or \
                             (p.type == mt5.ORDER_TYPE_SELL and signal == "BUY")
                # Đóng nếu đảo chiều hoặc AI không còn tự tin (>45% mới giữ)
                if is_reverse or max_conf < 0.45:
                    close_everything("AI THOÁT LINH HOẠT")

            # 4. Vào lệnh Limit mới (Chỉ vào khi tự tin > 60%)
            total_active = len(mt5.positions_get(magic=123456)) + len(mt5.orders_get(magic=123456))
            if total_active < config.MAX_OPEN_TRADES and max_conf > 0.60:
                tick = mt5.symbol_info_tick(config.SYMBOL)
                point = mt5.symbol_info(config.SYMBOL).point
                lot = get_lot_size(config.RISK_PERCENT, config.STOP_LOSS_POINTS)
                
                if signal == "BUY":
                    price = tick.ask - (config.OFFSET_POINTS * point)
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT
                    sl, tp = price - config.STOP_LOSS_POINTS * point, price + config.TAKE_PROFIT_POINTS * point
                else:
                    price = tick.bid + (config.OFFSET_POINTS * point)
                    order_type = mt5.ORDER_TYPE_SELL_LIMIT
                    sl, tp = price + config.STOP_LOSS_POINTS * point, price - config.TAKE_PROFIT_POINTS * point

                req = {
                    "action": mt5.TRADE_ACTION_PENDING, "symbol": config.SYMBOL, "volume": lot,
                    "type": order_type, "price": price, "sl": sl, "tp": tp, "magic": 123456,
                    "comment": f"Conf {max_conf:.2f}", "type_time": mt5.ORDER_TIME_DAY,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(req)
                print(f">>> ĐẶT {signal} LIMIT TẠI {price:.2f} | CONF: {max_conf:.2f}")

        except Exception as e:
            print(f"Lỗi hệ thống: {e}")
        
        time.sleep(5) # Tần suất quét 15 giây/lần

if __name__ == "__main__":
    run_bot()
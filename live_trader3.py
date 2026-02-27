import MetaTrader5 as mt5
import pandas as pd
import joblib
import config
import data_handler
import time

# --- CẤU HÌNH BỔ SUNG ---
OFFSET_POINTS = 200 # Khoảng cách đợi giá hồi về (ví dụ: đợi hồi 2 giá mới vào)

def get_lot_size(risk_percent, sl_points):
    account_info = mt5.account_info()
    if account_info is None: return 0.01
    balance = account_info.balance
    risk_amount = balance * (risk_percent / 100)
    lot = risk_amount / sl_points 
    return round(max(0.01, min(lot, 1.0)), 2)

def count_open_orders():
    # Đếm cả lệnh đang chạy (positions) và lệnh chờ (orders)
    pos = mt5.positions_get(magic=123456)
    orders = mt5.orders_get(magic=123456)
    return (len(pos) if pos else 0) + (len(orders) if orders else 0)

def close_all_orders_and_pending():
    """Đóng lệnh đang chạy và hủy các lệnh chờ cũ khi đảo chiều"""
    # 1. Hủy lệnh chờ (Pending Orders)
    pending = mt5.orders_get(magic=123456)
    if pending:
        for order in pending:
            req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
            mt5.order_send(req)
            
    # 2. Đóng vị thế đang chạy (Positions)
    positions = mt5.positions_get(magic=123456)
    if positions:
        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "magic": 123456,
                "comment": "AI Close Reverse",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            mt5.order_send(request)
        print("--- ĐÃ HỦY VÀ ĐÓNG TẤT CẢ LỆNH CŨ ---")

def run_bot():
    if not mt5.initialize(): return

    model = joblib.load(config.MODEL_FILE_NAME)
    print("--- BOT AI: CHIẾN THUẬT LIMIT ĐÓN GIÁ ĐÃ CHẠY ---")

    while True:
        try:
            rates = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_M5, 0, 200)
            df = pd.DataFrame(rates)
            df = df.rename(columns={'tick_volume': 'volume'})
            df_features = data_handler.add_technical_indicators(df)
            
            latest_features = df_features.iloc[-1:]
            prediction = model.predict(latest_features[model.feature_names_in_])
            signal = "BUY" if prediction[0] == 1 else "SELL"
            
            # KIỂM TRA ĐẢO CHIỀU (Nếu tín hiệu mới khác lệnh hiện tại -> Xóa sạch)
            # Code phần này tương tự nhưng áp dụng cho cả pending
            pos_now = mt5.positions_get(magic=123456)
            if pos_now:
                if (pos_now[0].type == mt5.ORDER_TYPE_BUY and signal == "SELL") or \
                   (pos_now[0].type == mt5.ORDER_TYPE_SELL and signal == "BUY"):
                    close_all_orders_and_pending()

            # VÀO LỆNH LIMIT MỚI
            if count_open_orders() < config.MAX_OPEN_TRADES:
                lot = get_lot_size(config.RISK_PERCENT, config.STOP_LOSS_POINTS)
                tick = mt5.symbol_info_tick(config.SYMBOL)
                point = mt5.symbol_info(config.SYMBOL).point
                
                if signal == "BUY":
                    # Đặt Buy Limit thấp hơn giá hiện tại (đợi giá giảm xuống mới mua)
                    limit_price = tick.ask - (OFFSET_POINTS * point)
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT
                    sl = limit_price - (config.STOP_LOSS_POINTS * point)
                    tp = limit_price + (config.TAKE_PROFIT_POINTS * point)
                else:
                    # Đặt Sell Limit cao hơn giá hiện tại (đợi giá hồi lên mới bán)
                    limit_price = tick.bid + (OFFSET_POINTS * point)
                    order_type = mt5.ORDER_TYPE_SELL_LIMIT
                    sl = limit_price + (config.STOP_LOSS_POINTS * point)
                    tp = limit_price - (config.TAKE_PROFIT_POINTS * point)

                request = {
                    "action": mt5.TRADE_ACTION_PENDING,
                    "symbol": config.SYMBOL,
                    "volume": lot,
                    "type": order_type,
                    "price": limit_price,
                    "sl": sl,
                    "tp": tp,
                    "magic": 123456,
                    "comment": "AI Limit Order",
                    "type_time": mt5.ORDER_TIME_DAY, # Lệnh tự hủy nếu hết ngày không khớp
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f">>> ĐÃ ĐẶT LỆNH CHỜ {signal} LIMIT TẠI {limit_price:.2f}")

        except Exception as e:
            print(f"Lỗi: {e}")

        #time.sleep(5) # Đợi nến 5p mới phân tích tiếp

if __name__ == "__main__":
    run_bot()
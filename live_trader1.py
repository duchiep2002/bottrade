import MetaTrader5 as mt5
import pandas as pd
import joblib
import config
import data_handler
import time

def get_lot_size(risk_percent, sl_points):
    """Tự động tính số Lot dựa trên % rủi ro tài khoản"""
    account_info = mt5.account_info()
    if account_info is None: return 0.01
    balance = account_info.balance
    risk_amount = balance * (risk_percent / 100)
    lot = risk_amount / sl_points 
    return round(max(0.01, min(lot, 1.0)), 2)

def count_open_orders():
    """Đếm số lệnh đang mở của Bot"""
    orders = mt5.positions_get(magic=123456)
    return len(orders) if orders else 0

def close_all_orders():
    """Hàm tự động đóng các lệnh đang mở khi có tín hiệu đảo chiều"""
    positions = mt5.positions_get(magic=123456)
    if positions:
        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            # Nếu lệnh đang BUY thì đóng bằng giá BID, nếu SELL đóng bằng giá ASK
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
        print("--- ĐÃ ĐÓNG LỆNH CŨ DO CÓ TÍN HIỆU ĐẢO CHIỀU ---")

def run_bot():
    if not mt5.initialize():
        print("Lỗi kết nối MT5")
        return

    model = joblib.load(config.MODEL_FILE_NAME)
    print("--- BOT AI ĐÃ SẴN SÀNG (CÓ TỰ ĐỘNG ĐẢO CHIỀU) ---")

    while True:
        try:
            # 1. Lấy dữ liệu và dự đoán
            rates = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_M5, 0, 200)
            df = pd.DataFrame(rates)
            df = df.rename(columns={'tick_volume': 'volume'})
            df_features = data_handler.add_technical_indicators(df)
            
            latest_features = df_features.iloc[-1:]
            prediction = model.predict(latest_features[model.feature_names_in_])
            signal = "BUY" if prediction[0] == 1 else "SELL"
            
            # 2. KIỂM TRA ĐẢO CHIỀU ĐỂ ĐÓNG LỆNH
            positions = mt5.positions_get(magic=123456)
            if positions:
                current_pos = positions[0]
                # Nếu lệnh đang mở là BUY mà AI báo SELL, hoặc ngược lại thì đóng lệnh ngay
                if (current_pos.type == mt5.ORDER_TYPE_BUY and signal == "SELL") or \
                   (current_pos.type == mt5.ORDER_TYPE_SELL and signal == "BUY"):
                    close_all_orders()

            # 3. VÀO LỆNH MỚI (Nếu chưa có lệnh nào mở)
            if count_open_orders() < config.MAX_OPEN_TRADES:
                lot = get_lot_size(config.RISK_PERCENT, config.STOP_LOSS_POINTS)
                tick = mt5.symbol_info_tick(config.SYMBOL)
                point = mt5.symbol_info(config.SYMBOL).point
                
                if signal == "BUY":
                    price, order_type = tick.ask, mt5.ORDER_TYPE_BUY
                    sl = price - (config.STOP_LOSS_POINTS * point)
                    tp = price + (config.TAKE_PROFIT_POINTS * point)
                else:
                    price, order_type = tick.bid, mt5.ORDER_TYPE_SELL
                    sl = price + (config.STOP_LOSS_POINTS * point)
                    tp = price - (config.TAKE_PROFIT_POINTS * point)

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": config.SYMBOL,
                    "volume": lot,
                    "type": order_type,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "magic": 123456,
                    "comment": f"AI Trade {signal}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f">>> ĐÃ VÀO LỆNH {signal} {lot} LOT.")
                else:
                    print(f"Lỗi: {result.comment}")

        except Exception as e:
            print(f"Lỗi: {e}")

        # Đợi nến tiếp theo (QUAN TRỌNG: Bạn cần bỏ comment dòng này để dữ liệu kịp cập nhật)
        time.sleep(10) 
       
if __name__ == "__main__":
    run_bot()
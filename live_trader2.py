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
    
    # Tính giá trị 1 point (với Vàng 1 lot 1 point ~ 1$)
    # Công thức: Lot = Tiền rủi ro / (Điểm cắt lỗ * giá trị point)
    lot = risk_amount / sl_points 
    return round(max(0.01, min(lot, 1.0)), 2) # Giới hạn tối thiểu 0.01, tối đa 1.0

def count_open_orders():
    """Đếm số lệnh đang mở của Bot (dựa trên Magic Number)"""
    orders = mt5.positions_get(magic=123456)
    return len(orders) if orders else 0

def run_bot():
    if not mt5.initialize():
        print("Lỗi kết nối MT5")
        return

    model = joblib.load(config.MODEL_FILE_NAME)
    print("--- BOT AI QUẢN LÝ VỐN ĐÃ KHỞI CHẠY ---")

    while True:
        try:
            # 1. Kiểm tra giới hạn số lệnh
            current_orders = count_open_orders()
            if current_orders >= config.MAX_OPEN_TRADES:
                print(f"Đang có {current_orders} lệnh mở. Tạm dừng vào thêm...")
                time.sleep(5)
                continue

            # 2. Lấy dữ liệu và tính toán chỉ báo
            rates = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_M5, 0, 200)
            df = pd.DataFrame(rates)
            df = df.rename(columns={'tick_volume': 'volume'})
            df_features = data_handler.add_technical_indicators(df)
            
            # 3. Dự đoán AI
            latest_features = df_features.iloc[-1:]
            prediction = model.predict(latest_features[model.feature_names_in_])
            print(f"DEBUG: Giá trị AI dự đoán thực tế là: {prediction[0]}")
            signal = "BUY" if prediction[0] == 1 else "SELL"
            
            # 4. Tính toán khối lượng vào lệnh tự động
            lot = get_lot_size(config.RISK_PERCENT, config.STOP_LOSS_POINTS)
            
           # 5. Đặt lệnh (Bản sửa lỗi minh bạch)
            tick = mt5.symbol_info_tick(config.SYMBOL)
            point = mt5.symbol_info(config.SYMBOL).point
            
            # Gán giá trị mặc định để tránh lỗi NameError
            request = None 

            if signal == "BUY":
                price = tick.ask
                sl = price - (config.STOP_LOSS_POINTS * point)
                tp = price + (config.TAKE_PROFIT_POINTS * point)
                order_type = mt5.ORDER_TYPE_BUY
                print(f"--- AI chọn BUY mã {config.SYMBOL} ---")
            
            elif signal == "SELL": # Dùng elif để tách biệt rõ ràng
                price = tick.bid
                sl = price + (config.STOP_LOSS_POINTS * point)
                tp = price - (config.TAKE_PROFIT_POINTS * point)
                order_type = mt5.ORDER_TYPE_SELL
                print(f"--- AI chọn SELL mã {config.SYMBOL} ---")
            
            else:
                print("--- AI phân vân, không vào lệnh ---")
                continue # Bỏ qua vòng lặp này

            # Chỉ gửi lệnh nếu signal hợp lệ
            if signal in ["BUY", "SELL"]:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": config.SYMBOL,
                    "volume": lot,
                    "type": order_type,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "magic": 123456,
                    "comment": f"AI Risk {config.RISK_PERCENT}%",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                # ... đoạn kiểm tra result phía sau giữ nguyên
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f">>> ĐÃ VÀO LỆNH {signal} {lot} LOT. SL: {sl:.2f}, TP: {tp:.2f}")
            else:
                print(f"Lỗi đặt lệnh: {result.comment}")

        except Exception as e:
            print(f"Lỗi: {e}")

        # Đợi nến tiếp theo (ví dụ nến 5p thì đợi 300s)
        #time.sleep(300)
       
if __name__ == "__main__":
    run_bot()
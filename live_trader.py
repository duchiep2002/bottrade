import MetaTrader5 as mt5
import pandas as pd
import joblib
import config
import data_handler
import time

# Biến để theo dõi nến
last_candle_time = 0

def count_open_orders():
    orders = mt5.positions_get(magic=123456)
    return len(orders) if orders else 0

def execute_trade(signal, prob):
    """Hàm thực hiện đặt lệnh chuyên nghiệp"""
    tick = mt5.symbol_info_tick(config.SYMBOL)
    symbol_info = mt5.symbol_info(config.SYMBOL)
    if tick is None or symbol_info is None:
        print("Lỗi lấy giá từ sàn!")
        return

    point = symbol_info.point
    # Với tài khoản nhỏ (~20$), mặc định đánh 0.01 để an toàn nhất
    lot = 0.01 
    
    if signal == "BUY":
        price = tick.ask
        sl = price - (config.STOP_LOSS_POINTS * point)
        tp = price + (config.TAKE_PROFIT_POINTS * point)
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = price + (config.STOP_LOSS_POINTS * point)
        tp = price - (config.TAKE_PROFIT_POINTS * point)
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": config.SYMBOL,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 123456,
        "comment": f"AI Conf: {prob:.2f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f">>> VÀO LỆNH THÀNH CÔNG: {signal} | SL: {sl:.2f} | TP: {tp:.2f}")
    else:
        print(f"Lỗi vào lệnh: {result.comment} (Code: {result.retcode})")

def run_bot():
    global last_candle_time
    if not mt5.initialize():
        print("Lỗi kết nối MT5")
        return

    try:
        model = joblib.load(config.MODEL_FILE_NAME)
        print("--- BOT AI ĐÃ KHỞI CHẠY: CHẾ ĐỘ NẾN ĐÓNG CỬA ---")
    except:
        print("Không tìm thấy file model AI!")
        return

    while True:
        try:
            # 1. KIỂM TRA NẾN MỚI
            # Lấy nến hiện tại (vị trí 0) để kiểm tra thời gian
            rates = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_M5, 0, 1)
            if rates is None or len(rates) == 0:
                time.sleep(2)
                continue
            
            current_time = rates[0]['time']

            # Nếu chưa có nến mới, bot sẽ đợi (không tính toán vô ích)
            if current_time == last_candle_time:
                time.sleep(5) 
                continue

            # --- ĐÃ CÓ NẾN MỚI ---
            last_candle_time = current_time
            print(f"\n[+] Đang phân tích nến mới lúc: {pd.to_datetime(current_time, unit='s')}")

            # 2. LẤY DỮ LIỆU 200 NẾN TRƯỚC ĐÓ (Bỏ qua nến 0 đang nhảy giá)
            rates = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_M5, 1, 200)
            df = pd.DataFrame(rates)
            df = df.rename(columns={'tick_volume': 'volume'})
            
            # 3. PANDAS TÍNH CHỈ BÁO & AI DỰ ĐOÁN
            df_features = data_handler.add_technical_indicators(df)
            latest_features = df_features.iloc[-1:]
            
            # Lấy xác suất tự tin của AI
            probs = model.predict_proba(latest_features[model.feature_names_in_])[0]
            prob_sell, prob_buy = probs[0], probs[1]
            
            print(f"Dự đoán AI: BUY {prob_buy*100:.1f}% | SELL {prob_sell*100:.1f}%")

            # Ngưỡng lọc lệnh rác (Tăng lên 60% để an toàn)
            signal = None
            conf = 0
            if prob_buy > 0.60:
                signal, conf = "BUY", prob_buy
            elif prob_sell > 0.60:
                signal, conf = "SELL", prob_sell

            # 4. KIỂM TRA ĐIỀU KIỆN VÀO LỆNH
            if signal:
                if count_open_orders() < 1: # CHỈ MỞ 1 LỆNH DUY NHẤT
                    execute_trade(signal, conf)
                else:
                    print("Đang có lệnh đang chạy, không vào thêm.")
            else:
                print("Tín hiệu yếu, không vào lệnh.")

        except Exception as e:
            print(f"Lỗi hệ thống: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_bot()
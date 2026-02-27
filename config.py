# Tên file: config.py
# Mô tả: File cấu hình trung tâm cho bot AI.

# --- Cấu hình Giao dịch ---
SYMBOL = "XAUUSDc"  # Cặp tiền hoặc sản phẩm muốn giao dịch
TIMEFRAME = "M5"   # Khung thời gian (M1, M5, H1, H4, D1...)
LOT_SIZE = 0.01    # Khối lượng mỗi lệnh

# --- Cấu hình Dữ liệu & Huấn luyện ---
# Tên file dữ liệu lịch sử xuất từ MT5
HISTORICAL_DATA_FILE = "XAUUSD_M5_2025-2026.csv"
# Tên file để lưu mô hình AI đã được huấn luyện
MODEL_FILE_NAME = "xgb_trading_model.joblib"

# --- Cấu hình Kết nối (ZMQ) ---
# Cổng để Python giao tiếp với MT5
# Phải trùng với cổng trong file EA .mq5
ZMQ_PORT = 5555

# --- Cấu hình Mô hình AI ---
# Số cây nến trong tương lai để AI dự đoán
# Ví dụ: 5 nghĩa là "Dự đoán giá trong 5 nến tới sẽ TĂNG hay GIẢM?"
PREDICTION_HORIZON = 12

# Thêm vào config.py
RISK_PERCENT = 2.0       # Chấp nhận rủi ro 3% tài khoản cho mỗi lệnh
MAX_OPEN_TRADES = 4      # Tối đa chỉ mở 5 lệnh cùng lúc
STOP_LOSS_POINTS = 1000  # 1.5 giá Vàng
TAKE_PROFIT_POINTS = 2000 # 2 giá Vàng
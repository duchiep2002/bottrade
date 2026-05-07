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
# --- XAU Guardian Bot: cấu hình an toàn mặc định ---
# Bot mới luôn chạy DRY-RUN trước. Chỉ đổi thành False sau khi đã demo đủ lâu.
GUARDIAN_SYMBOL = "XAUUSDc"
GUARDIAN_TIMEFRAME = "M15"
GUARDIAN_TREND_TIMEFRAME = "H1"
GUARDIAN_DRY_RUN = True
GUARDIAN_MAGIC = 240507
GUARDIAN_MIN_MODEL_CONFIDENCE = 0.64
GUARDIAN_MIN_RULE_CONFIDENCE = 0.58
GUARDIAN_RISK_PERCENT = 0.35
GUARDIAN_MAX_LOT = 0.03
GUARDIAN_MAX_SPREAD_POINTS = 180
GUARDIAN_STOP_ATR_MULTIPLIER = 1.8
GUARDIAN_TAKE_PROFIT_R_MULTIPLIER = 1.4
GUARDIAN_DAILY_LOSS_LIMIT_PERCENT = 2.0
GUARDIAN_COOLDOWN_MINUTES = 45
GUARDIAN_LOOP_SECONDS = 10
GUARDIAN_BARS_NEEDED = 240

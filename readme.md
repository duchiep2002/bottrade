Tổng quan Kiến trúcMô hình hoạt động như sau: Bạn sẽ chạy file train_model.py một lần để tạo ra file trading_model.joblib. Sau đó, bạn gắn EA AI_Trader_EA.mq4 vào chart trên MT4 và chạy file run_live_trading.py. Script này sẽ liên tục gửi tín hiệu cho EA để thực hiện giao dịch.Bước 1: Huấn luyện và Lưu trữ Mô hình AI (train_model.py)File này có nhiệm vụ đọc dữ liệu lịch sử (bạn cần xuất từ MT4 ra file .csv), tính toán các chỉ báo kỹ thuật, huấn luyện mô hình và lưu lại để sử dụng sau.# Tên file: train_model.py

import pandas as pd
from ta.add_all_ta_features import add_all_ta_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib # Thư viện để lưu và tải mô hình

print("Bắt đầu quá trình huấn luyện AI...")

# 1. TẢI VÀ XỬ LÝ DỮ LIỆU
# !!! THAY THẾ 'XAUUSD_H1.csv' BẰNG TÊN FILE DỮ LIỆU CỦA BẠN
try:
    df = pd.read_csv('XAUUSD_H1.csv',
                     names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    # Kết hợp cột ngày và giờ thành một cột datetime chuẩn
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.set_index('datetime')
    df = df[['open', 'high', 'low', 'close', 'volume']]
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'XAUUSD_H1.csv'. Hãy chắc chắn bạn đã xuất dữ liệu từ MT4.")
    exit()

print("Tải dữ liệu thành công. Bắt đầu tính toán chỉ báo kỹ thuật...")

# 2. KỸ THUẬT ĐẶC TRƯNG (FEATURE ENGINEERING)
# Thêm các chỉ báo kỹ thuật làm "đặc trưng" cho AI học
# Thư viện 'ta' sẽ tự động tính toán hàng chục chỉ báo
df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)

print(f"Đã thêm {len(df.columns) - 5} chỉ báo. Tổng số đặc trưng: {len(df.columns)}")

# 3. TẠO MỤC TIÊU DỰ ĐOÁN (LABEL)
# Ta muốn AI dự đoán: "Giá trong 5 cây nến tới sẽ TĂNG hay GIẢM?"
# Nếu giá đóng cửa 5 cây nến sau > giá đóng cửa hiện tại -> 1 (Tăng)
# Ngược lại -> 0 (Giảm)
future_candles = 5
df['target'] = (df['close'].shift(-future_candles) > df['close']).astype(int)

# Loại bỏ các hàng có dữ liệu bị thiếu (do tính toán shift và chỉ báo)
df = df.dropna()

# 4. HUẤN LUYỆN MÔ HÌNH
# Chọn dữ liệu để huấn luyện (X: đặc trưng, y: mục tiêu)
features = df.drop('target', axis=1)
labels = df['target']

# Chia dữ liệu thành 80% để huấn luyện và 20% để kiểm thử
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=False)

print(f"Huấn luyện mô hình trên {len(X_train)} mẫu dữ liệu...")

# Sử dụng mô hình RandomForest - một lựa chọn mạnh mẽ và phổ biến
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 để dùng tất cả CPU
model.fit(X_train, y_train)

# Đánh giá độ chính xác trên tập dữ liệu kiểm thử
accuracy = model.score(X_test, y_test)
print(f"Đánh giá hoàn tất. Độ chính xác của mô hình trên dữ liệu mới: {accuracy:.2%}")

# 5. LƯU MÔ HÌNH
# Lưu lại "bộ não" AI đã được huấn luyện vào file
joblib.dump(model, 'trading_model.joblib')
print("Đã lưu mô hình thành công vào file 'trading_model.joblib'.")
print("Quá trình huấn luyện hoàn tất!")

Bước 2: Chạy Giao dịch và Gửi Tín hiệu (run_live_trading.py)File này sẽ tải mô hình đã lưu, kết nối với MT4 và gửi tín hiệu "BUY" hoặc "SELL".# Tên file: run_live_trading.py

import zmq
import time
import joblib
import pandas as pd
from ta.add_all_ta_features import add_all_ta_features

print("Bắt đầu khởi tạo hệ thống giao dịch AI...")

# 1. TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN
try:
    model = joblib.load('trading_model.joblib')
    print("Tải mô hình 'trading_model.joblib' thành công.")
except FileNotFoundError:
    print("LỖI: Không tìm thấy file mô hình 'trading_model.joblib'. Vui lòng chạy file 'train_model.py' trước.")
    exit()

# 2. THIẾT LẬP KẾT NỐI VỚI MT4 QUA ZMQ
context = zmq.Context()
socket = context.socket(zmq.REQ) # REQ = Request
socket.connect("tcp://localhost:5555")
print("Đã kết nối với MT4 qua cổng 5555. Đang chờ tín hiệu...")

# 3. VÒNG LẶP GIAO DỊCH
while True:
    try:
        # TRONG THỰC TẾ: Bạn cần có một cơ chế để lấy dữ liệu mới nhất từ MT4
        # Ở đây, ta sẽ giả lập dữ liệu mới để minh họa
        # Ví dụ: bạn có thể viết một hàm để đọc 100 cây nến gần nhất từ MT4
        print("\nĐang chờ dữ liệu mới...")
        time.sleep(10) # Giả lập việc chờ nến mới

        # --- Giả lập lấy dữ liệu mới và tính toán đặc trưng ---
        # Đây là phần bạn cần tùy chỉnh để lấy dữ liệu thật
        # Ví dụ: df_new = get_latest_data_from_mt4(symbol="XAUUSD", timeframe="H1", count=100)
        # df_features = add_all_ta_features(...)
        # latest_features = df_features.iloc[-1:] # Lấy dòng cuối cùng
        # --------------------------------------------------------

        # Giả sử ta có dữ liệu mới nhất (để code chạy được)
        latest_features = pd.DataFrame([range(94)], columns=model.feature_names_in_) # Tạo dataframe giả lập

        # 4. ĐƯA RA DỰ ĐOÁN
        prediction = model.predict(latest_features)
        signal = "BUY" if prediction[0] == 1 else "SELL"

        # 5. GỬI TÍN HIỆU TỚI MT4
        print(f"AI dự đoán: {signal}. Gửi tín hiệu đến MT4...")
        socket.send_string(signal)

        # Chờ và nhận phản hồi từ MT4
        message = socket.recv_string()
        print(f"Phản hồi từ MT4: {message}")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        time.sleep(10)
Bước 3: Expert Advisor Nhận Lệnh trên MT4 (AI_Trader_EA.mq4)Đây là code MQL4. Bạn cần tạo một EA mới trong MetaEditor của MT4 và dán code này vào. Bạn cũng cần tải thư viện MQL-ZMQ và đặt đúng thư mục.// Tên file: AI_Trader_EA.mq4

#property copyright "Copyright 2025, Gemini"
#property link      "https://google.com"
#property version   "1.00"
#property strict

// Cần tải và include thư viện MQL-ZMQ từ: https://github.com/dingmaotu/mql-zmq
#include <Zmq.mqh>

int zmq_handle;
int magic_number = 12345; // Số magic để EA nhận diện lệnh của chính nó

//+------------------------------------------------------------------+
//| Hàm khởi tạo EA                                                  |
//+------------------------------------------------------------------+
int OnInit()
{
    // Khởi tạo ZMQ server ở chế độ REP (Reply)
    zmq_handle = Zmq_init(ZMQ_REP);
    if (zmq_handle < 0)
    {
        Alert("Lỗi khởi tạo ZMQ!");
        return(INIT_FAILED);
    }

    // Mở cổng 5555 để lắng nghe tín hiệu từ Python
    int result = Zmq_bind(zmq_handle, "tcp://*:5555");
    if (result < 0)
    {
        Alert("Lỗi bind ZMQ tới cổng 5555!");
        return(INIT_FAILED);
    }
    
    Print("EA AI Trader đã khởi động. Đang lắng nghe tín hiệu từ Python trên cổng 5555...");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Hàm chạy với mỗi tick giá mới                                    |
//+------------------------------------------------------------------+
void OnTick()
{
    // Nhận tín hiệu từ Python, không chờ đợi (ZMQ_DONTWAIT)
    string signal = Zmq_recv(zmq_handle, ZMQ_DONTWAIT);

    // Nếu có tín hiệu mới
    if (signal != "")
    {
        Print("Nhận được tín hiệu từ Python: ", signal);

        double lot = 0.01; // Khối lượng giao dịch
        double slippage = 3;
        double stop_loss = 150; // 150 points
        double take_profit = 300; // 300 points

        // Thực thi lệnh dựa trên tín hiệu
        if (signal == "BUY")
        {
            // Gửi lệnh MUA
            OrderSend(Symbol(), OP_BUY, lot, Ask, slippage, Ask - stop_loss * _Point, Ask + take_profit * _Point, "AI BUY", magic_number, 0, clrGreen);
        }
        else if (signal == "SELL")
        {
            // Gửi lệnh BÁN
            OrderSend(Symbol(), OP_SELL, lot, Bid, slippage, Bid + stop_loss * _Point, Bid - take_profit * _Point, "AI SELL", magic_number, 0, clrRed);
        }

        // Gửi lại phản hồi cho Python để xác nhận đã xử lý
        Zmq_send(zmq_handle, "Tín hiệu " + signal + " đã được xử lý.");
    }
}

//+------------------------------------------------------------------+
//| Hàm hủy EA                                                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("Đang đóng EA AI Trader và giải phóng cổng ZMQ...");
    Zmq_close(zmq_handle);
}
Cách Chạy:Chuẩn bị: Cài Python, các thư viện, tải MQL-ZMQ, xuất dữ liệu từ MT4.Huấn luyện: Chạy file python train_model.py trong terminal. Nó sẽ tạo ra file trading_model.joblib.Thiết lập MT4: Mở MetaEditor, tạo EA mới, dán code MQL4 vào, compile và gắn EA vào chart bạn muốn giao dịch.Giao dịch: Chạy file python run_live_trading.py. Script này sẽ bắt đầu gửi tín hiệu đến EA trên MT4.QUAN TRỌNG: Luôn luôn chạy trên tài khoản Demo trong nhiều tuần hoặc nhiều tháng để kiểm thử trước khi nghĩ đến việc dùng tiền thật.
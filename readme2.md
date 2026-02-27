Tổng quan Kiến trúc Dự ánDự án của chúng ta sẽ bao gồm các file sau, mỗi file có một nhiệm vụ riêng biệt:config.py: File cấu hình trung tâm để dễ dàng thay đổi các tham số như cặp tiền, khung thời gian, cổng kết nối...data_handler.py: Module chịu trách nhiệm tải, xử lý dữ liệu và tính toán các chỉ báo kỹ thuật.model_trainer.py: Script để huấn luyện "bộ não" AI từ dữ liệu lịch sử và lưu lại.live_trader.py: Script chính, chạy liên tục để tải mô hình AI, phân tích và gửi tín hiệu giao dịch đến MT5.AI_Trader_EA.mq5: Expert Advisor (EA) trên MT5, hoạt động như "đôi tay" nhận lệnh từ Python và thực thi giao dịch.File 1: Cấu hình (config.py)File này chứa tất cả các tham số bạn có thể muốn thay đổi.# Tên file: config.py

# --- Cấu hình Giao dịch ---
SYMBOL = "XAUUSD"  # Cặp tiền hoặc sản phẩm muốn giao dịch
TIMEFRAME = "H1"   # Khung thời gian (M1, M5, H1, H4, D1...)
LOT_SIZE = 0.01    # Khối lượng mỗi lệnh

# --- Cấu hình Dữ liệu & Huấn luyện ---
# Tên file dữ liệu lịch sử xuất từ MT5
HISTORICAL_DATA_FILE = "XAUUSD_H1_2020_2024.csv"
# Tên file để lưu mô hình AI đã được huấn luyện
MODEL_FILE_NAME = "xgb_trading_model.joblib"

# --- Cấu hình Kết nối (ZMQ) ---
# Cổng để Python giao tiếp với MT5
# Phải trùng với cổng trong file EA .mq5
ZMQ_PORT = 5555

# --- Cấu hình Mô hình AI ---
# Số cây nến trong tương lai để AI dự đoán
# Ví dụ: 5 nghĩa là "Dự đoán giá trong 5 nến tới sẽ TĂNG hay GIẢM?"
PREDICTION_HORIZON = 5
File 2: Xử lý Dữ liệu (data_handler.py)Module này chuẩn bị "thức ăn" cho AI.# Tên file: data_handler.py

import pandas as pd
import pandas_ta as ta # Thư viện tính chỉ báo kỹ thuật mạnh mẽ

def load_data(filepath):
    """Tải dữ liệu lịch sử từ file CSV xuất ra bởi MT5."""
    try:
        df = pd.read_csv(
            filepath,
            sep='\t', # Dữ liệu từ MT5 thường phân cách bằng tab
            names=['date', 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        )
        # Kết hợp cột ngày và giờ, đặt làm chỉ mục
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.set_index('datetime')
        # Giữ lại các cột cần thiết
        df = df[['open', 'high', 'low', 'close', 'real_volume']]
        df.rename(columns={'real_volume': 'volume'}, inplace=True)
        print(f"Tải thành công {len(df)} dòng dữ liệu từ {filepath}")
        return df
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu '{filepath}'.")
        return None

def add_technical_indicators(df):
    """Thêm các chỉ báo kỹ thuật vào dataframe làm đặc trưng cho AI."""
    print("Đang tính toán các chỉ báo kỹ thuật...")
    # Sử dụng pandas_ta để thêm một bộ chỉ báo phổ biến
    # Bạn có thể tùy chỉnh thêm/bớt các chỉ báo ở đây
    df.ta.strategy("common") # Thêm các chỉ báo thông dụng như RSI, MACD, Bollinger Bands...
    df.fillna(method='bfill', inplace=True) # Điền các giá trị NaN
    print(f"Đã thêm các chỉ báo. Tổng số cột đặc trưng: {len(df.columns)}")
    return df
File 3: Huấn luyện Mô hình AI (model_trainer.py)Script này chạy một lần để tạo ra file mô hình.# Tên file: model_trainer.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import config
import data_handler

def train():
    """Hàm chính để huấn luyện và lưu mô hình AI."""
    print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN AI ---")

    # 1. Tải và chuẩn bị dữ liệu
    df = data_handler.load_data(config.HISTORICAL_DATA_FILE)
    if df is None:
        return

    df = data_handler.add_technical_indicators(df)

    # 2. Tạo mục tiêu dự đoán (Label)
    df['target'] = (df['close'].shift(-config.PREDICTION_HORIZON) > df['close']).astype(int)
    df.dropna(inplace=True)

    # 3. Chuẩn bị dữ liệu cho mô hình
    X = df.drop('target', axis=1)
    y = df['target']

    # Chia dữ liệu: 80% huấn luyện, 20% kiểm thử
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"Kích thước tập huấn luyện: {len(X_train)} mẫu")
    print(f"Kích thước tập kiểm thử: {len(X_test)} mẫu")

    # 4. Huấn luyện mô hình XGBoost
    print("Đang huấn luyện mô hình XGBoost...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,      # Số lượng cây
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1              # Sử dụng tất cả CPU
    )
    model.fit(X_train, y_train)

    # 5. Đánh giá mô hình
    print("--- ĐÁNH GIÁ MÔ HÌNH ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác trên tập kiểm thử: {accuracy:.2%}")
    print("Báo cáo chi tiết:")
    print(classification_report(y_test, y_pred, target_names=['SELL', 'BUY']))

    # 6. Lưu mô hình
    joblib.dump(model, config.MODEL_FILE_NAME)
    print(f"Đã lưu mô hình thành công vào file: '{config.MODEL_FILE_NAME}'")
    print("--- QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT ---")

if __name__ == "__main__":
    train()
File 4: Giao dịch Trực tiếp (live_trader.py)Script chính chạy song song với MT5.# Tên file: live_trader.py

import zmq
import time
import joblib
import pandas as pd
import config
import data_handler

class LiveTrader:
    def __init__(self):
        self.model = self.load_model()
        self.socket = self.setup_zmq_connection()

    def load_model(self):
        """Tải mô hình AI đã được huấn luyện."""
        print("Đang tải mô hình AI...")
        try:
            model = joblib.load(config.MODEL_FILE_NAME)
            print("Tải mô hình thành công.")
            return model
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file '{config.MODEL_FILE_NAME}'. Vui lòng chạy 'model_trainer.py' trước.")
            exit()

    def setup_zmq_connection(self):
        """Thiết lập kết nối ZMQ để giao tiếp với MT5."""
        print(f"Đang kết nối với MT5 qua cổng: {config.ZMQ_PORT}...")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{config.ZMQ_PORT}")
        print("Kết nối thành công. Sẵn sàng giao dịch.")
        return socket

    def get_latest_data(self):
        """Yêu cầu và nhận dữ liệu mới nhất từ MT5."""
        print("Đang yêu cầu dữ liệu mới từ MT5...")
        self.socket.send_string("GET_DATA")
        data_str = self.socket.recv_string()

        # Chuyển đổi chuỗi dữ liệu CSV từ MT5 thành DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(data_str), sep=',')
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('datetime')
        df = df[['open', 'high', 'low', 'close', 'volume']]
        return df

    def run(self):
        """Vòng lặp giao dịch chính."""
        print("--- BOT GIAO DỊCH AI BẮT ĐẦU HOẠT ĐỘNG ---")
        while True:
            try:
                # 1. Lấy dữ liệu mới
                df_latest = self.get_latest_data()
                print(f"Đã nhận {len(df_latest)} cây nến mới nhất.")

                # 2. Tính toán đặc trưng
                df_features = data_handler.add_technical_indicators(df_latest.copy())
                
                # Lấy dòng dữ liệu cuối cùng để dự đoán
                latest_features = df_features.iloc[-1:]

                # 3. Đưa ra dự đoán
                prediction = self.model.predict(latest_features[self.model.feature_names_in_])
                signal = "BUY" if prediction[0] == 1 else "SELL"
                
                print(f"==> AI ĐƯA RA QUYẾT ĐỊNH: {signal} cho {config.SYMBOL} ==")

                # 4. Gửi tín hiệu đến MT5
                print("Gửi tín hiệu đến MT5 để thực thi...")
                self.socket.send_string(f"TRADE,{signal},{config.LOT_SIZE}")
                
                # Nhận phản hồi từ MT5
                response = self.socket.recv_string()
                print(f"Phản hồi từ MT5: {response}")
                
                # Chờ cho nến tiếp theo
                print("\nĐang chờ chu kỳ tiếp theo...")
                time.sleep(60) # Chờ 1 phút, bạn có thể điều chỉnh

            except Exception as e:
                print(f"Đã xảy ra lỗi trong vòng lặp chính: {e}")
                # Thử kết nối lại nếu có lỗi
                self.socket.close()
                self.socket = self.setup_zmq_connection()
                time.sleep(10)

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run()
File 5: Expert Advisor trên MT5 (AI_Trader_EA.mq5)"Đôi tay" thực thi lệnh trong MetaTrader 5.// Tên file: AI_Trader_EA.mq5

#property copyright "Copyright 2025, Gemini AI"
#property link      "https://google.com"
#property version   "1.00"
#property strict

#include <Zmq.mqh> // Cần tải và include thư viện MQL-ZMQ
#include <Trade\Trade.mqh>

// --- Cấu hình ---
#define ZMQ_PORT 5555
#define MAGIC_NUMBER 67890
#define SL_POINTS 2000 // Stop loss 200 pips (cho XAUUSD)
#define TP_POINTS 4000 // Take profit 400 pips

int zmq_handle;
CTrade trade;

//+------------------------------------------------------------------+
//| Hàm khởi tạo EA                                                  |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MAGIC_NUMBER);
   trade.SetTypeFillingBySymbol(Symbol());
   
   zmq_handle = Zmq_init(ZMQ_REP);
   if(zmq_handle < 0)
   {
      Alert("Lỗi khởi tạo ZMQ!");
      return(INIT_FAILED);
   }
   
   int result = Zmq_bind(zmq_handle, "tcp://*:" + IntegerToString(ZMQ_PORT));
   if(result < 0)
   {
      Alert("Lỗi bind ZMQ tới cổng ", ZMQ_PORT, "!");
      return(INIT_FAILED);
   }
   
   Print("EA AI Trader đã khởi động. Đang lắng nghe tín hiệu từ Python trên cổng ", ZMQ_PORT);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Hàm chạy với mỗi tick giá mới                                    |
//+------------------------------------------------------------------+
void OnTick()
{
   // Nhận tín hiệu từ Python, không chờ đợi
   string request = Zmq_recv(zmq_handle, ZMQ_DONTWAIT);

   if(request != "")
   {
      string response = "UNKNOWN_COMMAND";
      string parts[];
      StringSplit(request, ',', parts);

      // Xử lý yêu cầu từ Python
      if(parts[0] == "GET_DATA")
      {
         response = GetHistoricalData();
      }
      else if(parts[0] == "TRADE")
      {
         string signal = parts[1];
         double lots = StringToDouble(parts[2]);
         response = ExecuteTrade(signal, lots);
      }
      
      // Gửi lại phản hồi cho Python
      Zmq_send(zmq_handle, response);
   }
}

//+------------------------------------------------------------------+
//| Hàm thực thi giao dịch                                           |
//+------------------------------------------------------------------+
string ExecuteTrade(string signal, double lots)
{
   Print("Nhận lệnh giao dịch: ", signal, " với ", DoubleToString(lots, 2), " lots");
   
   // Đóng tất cả các lệnh đang mở trước khi mở lệnh mới để tránh nhiều lệnh cùng lúc
   CloseAllPositions();
   
   double price = 0;
   double sl = 0;
   double tp = 0;
   
   if(signal == "BUY")
   {
      price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
      sl = price - SL_POINTS * _Point;
      tp = price + TP_POINTS * _Point;
      if(!trade.Buy(lots, Symbol(), price, sl, tp, "AI BUY"))
      {
         return "Lỗi khi đặt lệnh BUY: " + IntegerToString(trade.ResultRetcode());
      }
   }
   else if(signal == "SELL")
   {
      price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
      sl = price + SL_POINTS * _Point;
      tp = price - TP_POINTS * _Point;
      if(!trade.Sell(lots, Symbol(), price, sl, tp, "AI SELL"))
      {
         return "Lỗi khi đặt lệnh SELL: " + IntegerToString(trade.ResultRetcode());
      }
   }
   
   return "Lệnh " + signal + " đã được thực thi thành công.";
}

//+------------------------------------------------------------------+
//| Hàm lấy dữ liệu lịch sử và gửi cho Python                        |
//+------------------------------------------------------------------+
string GetHistoricalData()
{
   MqlRates rates[];
   // Lấy 200 cây nến gần nhất của khung thời gian hiện tại
   if(CopyRates(Symbol(), Period(), 0, 200, rates) < 0)
   {
      return "ERROR,Cannot copy rates";
   }
   
   // Chuyển dữ liệu thành định dạng CSV
   string csv_data = "time,open,high,low,close,volume\n";
   for(int i = 0; i < ArraySize(rates); i++)
   {
      csv_data += StringFormat("%d,%.5f,%.5f,%.5f,%.5f,%d\n",
         rates[i].time,
         rates[i].open,
         rates[i].high,
         rates[i].low,
         rates[i].close,
         rates[i].tick_volume
      );
   }
   return csv_data;
}

//+------------------------------------------------------------------+
//| Hàm đóng tất cả các vị thế đang mở                               |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == MAGIC_NUMBER)
         {
            trade.PositionClose(ticket);
         }
      }
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
Hướng dẫn Chạy Hệ thốngCài đặt Môi trường Python:Cài đặt Python 3.9+.Chạy lệnh sau trong terminal: pip install pandas pandas-ta xgboost scikit-learn joblib pyzmqChuẩn bị Dữ liệu:Mở MT5, vào Tools -> History Center.Tìm cặp tiền của bạn (ví dụ: XAUUSD), chọn khung thời gian (ví dụ: H1).Export dữ liệu ra file .csv. Đổi tên file và đặt vào cùng thư mục với các file Python. Cập nhật tên file trong config.py.Huấn luyện AI:Mở terminal trong thư mục dự án.Chạy lệnh: python model_trainer.pyChờ quá trình hoàn tất. File xgb_trading_model.joblib sẽ được tạo ra.Thiết lập MetaTrader 5:Tải thư viện MQL-ZMQ từ GitHub. Giải nén và chép vào thư mục MQL5/Include của MT5.Mở MetaEditor, tạo một Expert Advisor mới, dán code từ file AI_Trader_EA.mq5 vào.Nhấn Compile. Nếu không có lỗi, bạn đã sẵn sàng.Kéo EA vừa tạo vào chart XAUUSD H1 trên MT5. Nhớ tick vào ô "Allow DLL imports".Bắt đầu Giao dịch:Chạy script Python chính: python live_trader.pyScript sẽ kết nối với EA trên MT5. Bạn sẽ thấy các log xuất hiện trên cả terminal của Python và tab "Experts" của MT5. Bot sẽ bắt đầu yêu cầu dữ liệu, phân tích và gửi tín hiệu giao dịch.QUAN TRỌNG: Hãy luôn chạy hệ thống này trên tài khoản Demo trong một thời gian dài để kiểm thử và đánh giá hiệu quả trước khi mạo hiểm bất kỳ khoản vốn nào.
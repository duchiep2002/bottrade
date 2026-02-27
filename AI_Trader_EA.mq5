// Tên file: AI_Trader_EA.mq5
// Mô tả: Expert Advisor trên MT5, hoạt động như "đôi tay" nhận lệnh từ Python và thực thi giao dịch.

#property copyright "Copyright 2025, Gemini AI"
#property link      "https://google.com"
#property version   "1.00"
#property strict

#include <Zmq.mqh> // Cần tải và include thư viện MQL-ZMQ
#include <Trade\Trade.mqh>

// --- Cấu hình ---
#define ZMQ_PORT 5555
#define MAGIC_NUMBER 67890
#define SL_POINTS 500 // Stop loss 50 pips (cho XAUUSD)
#define TP_POINTS 1000 // Take profit 100 pips

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
         (long)rates[i].tick_volume
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

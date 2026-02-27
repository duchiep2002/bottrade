# Tên file: model_trainer.py
# Mô tả: Script để huấn luyện "bộ não" AI từ dữ liệu lịch sử và lưu lại.

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

import pandas as pd
import numpy as np

def load_data(filepath):
    try:
        df = pd.read_csv(
            filepath,
            sep='\t',
            skiprows=1,
            names=['date', 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'vol_skip', 'spread']
        )
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.set_index('datetime')
        df = df[['open', 'high', 'low', 'close', 'tick_volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except Exception as e:
        print(f"LỖI đọc file: {e}")
        return None

def add_technical_indicators(df):
    print("Đang tính toán chỉ báo bằng Pandas thuần...")
    # Tính SMA (Trung bình động)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    # Tính RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # Điền giá trị trống
    # Sửa lỗi AttributeError: 'NoneType'
    df = df.ffill()
    df = df.bfill()
    return df
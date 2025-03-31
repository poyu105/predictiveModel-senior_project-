import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# 📌 1. 載入模型
print("🔄 載入模型中...")
model = load_model("sales_forecast_model.h5")
print("✅ 模型載入成功！")

# 📌 2. 讀取資料
df = pd.read_excel("Data_sorted.xlsx")
df["order_date"] = pd.to_datetime(df["order_date"])
df["order_day"] = df["order_date"].dt.dayofweek
df["order_month"] = df["order_date"].dt.month
festival_dates = ["2024-01-01", "2024-02-10", "2024-12-25"]
df["is_festival"] = df["order_date"].isin(pd.to_datetime(festival_dates)).astype(int)

# 確保使用正確欄位名稱（請依據實際資料名稱確認）
noodle_id_col = "泡麵口味_id"
quantity_col = "quantity"

# 預測函數：針對特定泡麵口味
def predict_sales(date, noodle_id):
    date = pd.to_datetime(date)
    day = np.array([[date.dayofweek]])
    month = np.array([[date.month]])
    is_festival = np.array([[1]]) if date in pd.to_datetime(festival_dates) else np.array([[0]])

    df_noodle = df[df[noodle_id_col] == noodle_id]

    last_year_sales_df = df_noodle[df_noodle["order_date"] == date - pd.DateOffset(years=1)][quantity_col]
    last_year_sales = np.array([[last_year_sales_df.sum() if not last_year_sales_df.empty else 0]])

    past_7_days = []
    for i in range(1, 8):
        past_sales_df = df_noodle[df_noodle["order_date"] == date - pd.DateOffset(days=i)][quantity_col]
        past_7_days.append(past_sales_df.sum() if not past_sales_df.empty else 0)
    past_7_days = np.array(past_7_days).reshape(1, 7, 1)

    pred = model.predict([day, month, is_festival, last_year_sales, past_7_days], verbose=0)
    return round(pred[0][0])

# 📌 3. 執行預測
future_date = input("📅 請輸入要預測的日期（格式：YYYY-MM-DD）：")

flavor_map = {
    1: "辛辣",
    2: "海鮮",
    3: "牛肉",
    4: "素食",
    5: "清湯"
}

print(f"\n📊 {future_date} 預測銷量如下：")
for noodle_id, name in flavor_map.items():
    pred = predict_sales(future_date, noodle_id)
    print(f"🧂 {name}：{pred} 碗")

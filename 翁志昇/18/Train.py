import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, concatenate
import matplotlib

# 避免中文亂碼
matplotlib.rc("font", family="Microsoft YaHei")
plt.rcParams['axes.unicode_minus'] = False

print("🔄 讀取數據並進行預處理...")
df = pd.read_excel("Data_sorted.xlsx").copy()

# 轉換日期格式
df["order_date"] = pd.to_datetime(df["order_date"])
df["order_day"] = df["order_date"].dt.dayofweek
df["order_month"] = df["order_date"].dt.month

# 是否為特定節日（銷量降低的日子）
festival_dates = [
    "2024-01-01", "2024-02-08", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14",
    "2024-02-28", "2024-04-04", "2024-04-05", "2024-04-06", "2024-04-07",
    "2024-06-08", "2024-06-09", "2024-06-10",
    "2024-09-17", "2024-10-10",
    "2024-12-25"
]
df["is_festival"] = df["order_date"].isin(pd.to_datetime(festival_dates)).astype(int)

# 將每日每種泡麵總量聚合成每筆紀錄
df_grouped = df.groupby(["order_date", "泡麵_name"]).agg({
    "quantity": "sum",
    "order_day": "first",
    "order_month": "first",
    "is_festival": "first"
}).reset_index()

# 建立過去七天銷售量特徵
noodle_types = df_grouped["泡麵_name"].unique()
full_data = []

for noodle in noodle_types:
    temp = df_grouped[df_grouped["泡麵_name"] == noodle].copy()
    temp = temp.sort_values("order_date")
    for i in range(1, 8):
        temp[f"sales_day_{i}"] = temp["quantity"].shift(i).fillna(0)
    temp["last_year_sales"] = 0  # 因為只有一年資料
    full_data.append(temp)

df_model = pd.concat(full_data)

# 🔧 輸入輸出資料
features = [
    "order_day", "order_month", "is_festival", "last_year_sales",
    "sales_day_1", "sales_day_2", "sales_day_3", "sales_day_4",
    "sales_day_5", "sales_day_6", "sales_day_7"
]

X = df_model[features].values
y = df_model["quantity"].values
X_day7 = df_model[[f"sales_day_{i}" for i in range(1, 8)]].values.reshape(-1, 7, 1)

# 建立模型輸入層
input_day = Input(shape=(1,), name='input_day')
input_month = Input(shape=(1,), name='input_month')
input_festival = Input(shape=(1,), name='input_festival')
input_last_year = Input(shape=(1,), name='input_last_year')
input_day7 = Input(shape=(7, 1), name='input_day7')

# 模型結構
x1 = Dense(5, activation='relu')(input_day)
x2 = Dense(5, activation='relu')(input_month)
x3 = Dense(5, activation='relu')(input_festival)
x4 = Dense(5, activation='relu')(input_last_year)
x5 = LSTM(5, return_sequences=True)(input_day7)
x5 = Flatten()(x5)

c = concatenate([x1, x2, x3, x4, x5])
layer1 = Dense(64, activation='relu')(c)
outputs = Dense(1, activation='linear')(layer1)

model = Model(inputs=[input_day, input_month, input_festival, input_last_year, input_day7], outputs=outputs)
model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()

print("🚀 開始訓練 LSTM 模型...")
history = model.fit(
    x=[df_model["order_day"].values, df_model["order_month"].values,
       df_model["is_festival"].values, df_model["last_year_sales"].values, X_day7],
    y=y,
    batch_size=16,
    epochs=15,
    verbose=1,
    shuffle=False
)

# 📈 顯示訓練損失
plt.plot(history.history['loss'], label='Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("訓練損失")
plt.legend()
plt.show()

model.save("sales_forecast_model.h5")
print("✅ 模型已儲存為 sales_forecast_model.h5")
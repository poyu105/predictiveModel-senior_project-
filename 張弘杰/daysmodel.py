import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pyodbc
import uuid
from datetime import datetime, timedelta

# ========================
# 建立 SQL Server 連線
conn = pyodbc.connect(
    r'DRIVER={SQL Server};SERVER=HJROG;DATABASE=ordersystem;Trusted_Connection=yes;'
)
cursor = conn.cursor()
# ========================
# 輸入日期與預測天數
start_date_input = input("請輸入預測起始日期 (格式：YYYY-MM-DD)：")
predict_days_input = input("請輸入要預測的天數（例如 1、3 或 5）：")

# 驗證與轉換輸入格式
try:
    start_date = datetime.strptime(start_date_input, "%Y-%m-%d")
    predict_days = int(predict_days_input)
except:
    print("輸入格式錯誤，請輸入正確的日期格式和天數")
    exit()

# 根據日期產生對應的季節（簡易邏輯）
def get_season(month):
    if month in [3, 4, 5]:
        return '春'
    elif month in [6, 7, 8]:
        return '夏'
    elif month in [9, 10, 11]:
        return '秋'
    else:
        return '冬'

# 建立未來日期與假設天氣（這裡示範用輪流替代）
weather_list = ['晴天', '陰天', '雨天']
future_dates = []
for i in range(predict_days):
    date = start_date + timedelta(days=i)
    season = get_season(date.month)
    weather = weather_list[i % len(weather_list)]  # 模擬天氣
    future_dates.append({
        '日期': date.strftime("%Y-%m-%d"),
        '季節': season,
        '天氣': weather
    })

future_df = pd.DataFrame(future_dates)

# ========================
# 讀取與訓練模型
# ========================
sales_df = pd.read_excel(r"C:\Users\lichu\OneDrive\桌面\vs\深度\泡麵銷量.xlsx")
sales_df['季節'] = sales_df['季節'].replace({'春天': '春', '夏天': '夏', '秋天': '秋', '冬天': '冬'})

le_season = LabelEncoder()
le_weather = LabelEncoder()
le_noodle = LabelEncoder()

sales_df['季節編碼'] = le_season.fit_transform(sales_df['季節'])
sales_df['天氣編碼'] = le_weather.fit_transform(sales_df['天氣'])
sales_df['泡麵編碼'] = le_noodle.fit_transform(sales_df['泡麵名稱'])

X = sales_df[['季節編碼', '天氣編碼', '泡麵編碼']]
y = sales_df['銷量']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n測試集MAE：{mae:.2f}")
print(f"測試集RMSE：{rmse:.2f}")
print(f"測試集R²分數：{r2:.2f}\n")

# 先查出泡麵對應的 meal_id（假設 Meal 表有泡麵名稱欄位）
meal_id_map = {}
cursor.execute("SELECT meal_id, name FROM Meal")
for row in cursor.fetchall():
    meal_id_map[row.name] = row.meal_id
# ========================
# 預測未來泡麵銷量
# ========================
noodle_names = le_noodle.classes_
predictions = []

for _, row in future_df.iterrows():
    for noodle in noodle_names:
        temp = {
            '日期': row['日期'],
            '季節': row['季節'],
            '天氣': row['天氣'],
            '泡麵名稱': noodle
        }
        temp['季節編碼'] = le_season.transform([row['季節']])[0]
        temp['天氣編碼'] = le_weather.transform([row['天氣']])[0]
        temp['泡麵編碼'] = le_noodle.transform([noodle])[0]
        
        input_features = [[temp['季節編碼'], temp['天氣編碼'], temp['泡麵編碼']]]
        predicted_sales = model.predict(input_features)[0]
        
        temp['備貨量'] = round(predicted_sales)
        predictions.append(temp)

future_sales_df = pd.DataFrame(predictions)
# 將預測結果寫入 Prediction 表
for _, row in future_sales_df.iterrows():
    meal_id = meal_id_map.get(row['泡麵名稱'])
    if not meal_id:
        print(f"找不到泡麵『{row['泡麵名稱']}』對應的 meal_id，已略過")
        
        continue

    prediction_id = str(uuid.uuid4())  # 產生唯一ID
    cursor.execute("""
        INSERT INTO Prediction (prediction_id, date, predicted_sales, weather_condition, season, meal_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        prediction_id,
        row['日期'],
        int(row['備貨量']),
        row['天氣'],
        row['季節'],
        meal_id
    ))

# 提交變更並關閉連線
conn.commit()
cursor.close()
conn.close()
print("預測結果已成功寫入 SQL Server 的 Prediction 資料表")

# 輸出結果
output_filename = f"預測結果_{start_date.strftime('%Y%m%d')}_{predict_days}天.xlsx"
future_sales_df[['日期', '季節', '天氣', '泡麵名稱', '備貨量']].to_excel(output_filename, index=False)
print(f"預測結果已儲存為：{output_filename}")

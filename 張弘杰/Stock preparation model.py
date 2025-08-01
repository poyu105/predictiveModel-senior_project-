import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pyodbc
import uuid
from datetime import datetime, timedelta

# 建立 SQL Server 連線
conn = pyodbc.connect(
    r'DRIVER={SQL Server};SERVER=HJROG;DATABASE=ordersystem;Trusted_Connection=yes;'
)
cursor = conn.cursor()

# 輸入預測參數 
start_date_input = input("請輸入預測起始日期 (格式：YYYY-MM-DD)：")
predict_days_input = input("請輸入要預測的天數（例如 1、3 或 5）：")

try:
    start_date = datetime.strptime(start_date_input, "%Y-%m-%d")
    predict_days = int(predict_days_input)
except:
    print("輸入格式錯誤，請輸入正確的日期格式和天數")
    exit()

# 根據月份轉換季節
def get_season(month):
    if month in [3, 4, 5]:
        return '春'
    elif month in [6, 7, 8]:
        return '夏'
    elif month in [9, 10, 11]:
        return '秋'
    else:
        return '冬'

# 模擬天氣與未來日期
weather_list = ['晴天', '陰天', '雨天']
future_dates = []
for i in range(predict_days):
    date = start_date + timedelta(days=i)
    season = get_season(date.month)
    weather = weather_list[i % len(weather_list)]
    future_dates.append({
        '日期': date.strftime("%Y-%m-%d"),
        '季節': season,
        '天氣': weather
    })

future_df = pd.DataFrame(future_dates)

# 建立模型
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
print(f"\n模型效能：")
print(f"MAE：{mean_absolute_error(y_test, y_pred):.2f}, RMSE：{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}, R²：{r2_score(y_test, y_pred):.2f}\n")

# 檢查 Meal 表
meal_id_map = {}
cursor.execute("SELECT meal_id, name FROM Meal")
for row in cursor.fetchall():
    meal_id_map[row.name] = row.meal_id

# 若為空，自動插入泡麵名稱與庫存
if not meal_id_map:
    print("Meal 資料表為空，開始建立...")
    for noodle in le_noodle.classes_:
        meal_id = str(uuid.uuid4())
        inventory_id = str(uuid.uuid4())
        now = datetime.now()

        cursor.execute("""
            INSERT INTO Inventory (inventory_id, quantity, create_at, update_at)
            VALUES (?, ?, ?, ?)
        """, (inventory_id, 0, now, now))

        meal_type = "泡麵"
        img_path = "default.jpg"
        description = "暫無描述"
        price = 50
        cost = 30

        cursor.execute("""
            INSERT INTO Meal (meal_id, name, inventory_id, type, img_path, description, price, cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (meal_id, noodle, inventory_id, meal_type, img_path, description, price, cost))

        meal_id_map[noodle] = meal_id

    conn.commit()
    print("泡麵與庫存已建立\n")

# 預測並寫入 Prediction 表 
predictions = []

for _, row in future_df.iterrows():
    for noodle in le_noodle.classes_:
        temp = {
            '日期': row['日期'],
            '季節': row['季節'],
            '天氣': row['天氣'],
            '泡麵名稱': noodle
        }
        temp['季節編碼'] = le_season.transform([row['季節']])[0]
        temp['天氣編碼'] = le_weather.transform([row['天氣']])[0]
        temp['泡麵編碼'] = le_noodle.transform([noodle])[0]

        input_df = pd.DataFrame([[
            temp['季節編碼'],
            temp['天氣編碼'],
            temp['泡麵編碼']
        ]], columns=['季節編碼', '天氣編碼', '泡麵編碼'])

        predicted_sales = model.predict(input_df)[0]
        temp['備貨量'] = round(predicted_sales)
        predictions.append(temp)

future_sales_df = pd.DataFrame(predictions)

for _, row in future_sales_df.iterrows():
    meal_id = meal_id_map.get(row['泡麵名稱'])
    if not meal_id:
        print(f"找不到泡麵『{row['泡麵名稱']}』對應的 meal_id，已略過")
        continue

    prediction_date = datetime.strptime(row['日期'], "%Y-%m-%d").date()

    # 查詢前將參數轉為字串以避免 HYC00 錯誤
    cursor.execute("""
        SELECT COUNT(*) FROM Prediction WHERE date = ? AND meal_id = ?
    """, (str(prediction_date), str(meal_id)))
    
    exists = cursor.fetchone()[0] > 0
    if exists:
        print(f"已存在 {row['泡麵名稱']} 的 {prediction_date} 預測，略過插入")
        continue

    prediction_id = str(uuid.uuid4())
    cursor.execute("""
    INSERT INTO Prediction (prediction_id, date, predicted_sales, weather_condition, season, meal_id)
    VALUES (?, ?, ?, ?, ?, ?)
""", (
    str(prediction_id),
    prediction_date.strftime('%Y-%m-%d'),
    int(row['備貨量']),
    str(row['天氣']),
    str(row['季節']),
    str(meal_id)
))


conn.commit()
cursor.close()
conn.close()

# 匯出 Excel
output_filename = f"預測結果_{start_date.strftime('%Y%m%d')}_{predict_days}天.xlsx"
future_sales_df[['日期', '季節', '天氣', '泡麵名稱', '備貨量']].to_excel(output_filename, index=False)
print(f"\n✅ 預測結果已寫入 SQL 並匯出為 Excel：{output_filename}")

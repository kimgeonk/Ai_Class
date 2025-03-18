import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\3_week\abalone\abalone.csv"
df = pd.read_csv(file_path)

encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])
    
X = df.drop(columns=['Rings'])
y = df['Rings']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rl_model = RandomForestRegressor(random_state=42)
rl_model.fit(x_train, y_train)

y_pred = rl_model.predict(x_test)

print('평균제곱근오차', mean_squared_error(y_pred, y_test)) # 3.8048405502392346
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\2_week\quiz\mobile.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=['price_range']) # 입력값
y = df['price_range'] # 출력값

x_train, x_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 42) # train, test를 분리
lr_model = LogisticRegression(random_state = 42) # 로지스틱회기 모델 생성
lr_model.fit(x_train, y_train) # 로지스틱회기 모델로 학습

y_pred = lr_model.predict(x_test) # 로지스틱회기 모델로 y값 예측

accuarcy = accuracy_score(y_test, y_pred) # 예측한 y값과 실제 y값 비교
print(f"Accuarcy : {accuarcy:.2f}") # 0.63 정확도가 너무 낮음 --> 실제 데이터 값이 너무 퍼져 있음음
print(y_pred[0:5]) # 예측값 상위 5개
print(y_test[0:5]) # 실제값 상위 5개 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\2_week\quiz\mobile.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=['price_range']) # 입력값
y = df['price_range'] # 출력값(인코딩 생략)

x_trina, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state = 42)
dt_model = DecisionTreeClassifier(random_state = 42) # DT모델 생성
dt_model.fit(x_trina, y_train) # DT모델을 학습 

y_pred = dt_model.predict(x_test) # 학습을 통해 도출된 y_test값

accuracy = accuracy_score(y_test, y_pred) # 실제 y_test값과 학습을 통해 도출된 y_pred값을 비교
print(f"Accuracy: {accuracy:.2f}") # 0.83
print(y_pred[:5]) # y_pred 값 5개 출력
print(y_test[:5]) # y_test 값 5개 출력
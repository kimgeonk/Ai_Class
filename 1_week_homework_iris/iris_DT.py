import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\1_week_homework_iris\iris.csv" 
df = pd.read_csv(file_path)

# 2️⃣ 입력(X)과 출력(y) 분리
X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]  # 입력값
y = df['Name']  # 출력값 

# 3️⃣ 라벨 인코딩 (품종을 숫자로 변환)
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # 예: 'setosa' → 0, 'versicolor' → 1, 'virginica' → 2

# 4️⃣ 학습(train) & 테스트(test) 데이터 분할 (75%: 학습, 25%: 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 5️⃣ 결정트리(Decision Tree) 모델 생성 및 학습
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# 6️⃣ 테스트 데이터 예측
y_pred = dt_model.predict(X_test)

# 7️⃣ 모델 평가 (정확도 확인)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ 모델 정확도: {accuracy:.4f}")  # 1.0000 (100%) 정확도 기대 가능

# 8️⃣ 테스트 데이터 중 5개 샘플을 선택하여 입력값과 예측값 출력
sample_idx = np.random.choice(len(X_test), 5, replace=False)  # 5개 랜덤 선택
sample_X = X_test.iloc[sample_idx]  # 입력값 (꽃받침, 꽃잎 데이터)
sample_y_true = y_test[sample_idx]  # 실제 품종
sample_y_pred = dt_model.predict(sample_X)  # 예측 품종

print("\n🎯 테스트 데이터 예측 결과 (5개 샘플)")
print("----------------------------------------------------------------------------------------")
print(" SepalLength | SepalWidth | PetalLength | PetalWidth |   실제 품종 (True)   |   예측 품종 (Predicted)")
print("----------------------------------------------------------------------------------------")
for i in range(5):
    sepal_length = sample_X.iloc[i, 0]
    sepal_width = sample_X.iloc[i, 1]
    petal_length = sample_X.iloc[i, 2]
    petal_width = sample_X.iloc[i, 3]
    true_label = encoder.inverse_transform([sample_y_true[i]])[0]  # 실제 품종 이름
    pred_label = encoder.inverse_transform([sample_y_pred[i]])[0]  # 예측 품종 이름

    print(f"  {sepal_length:^10.2f} | {sepal_width:^10.2f} | {petal_length:^10.2f} | {petal_width:^10.2f} | {true_label:^15} | {pred_label:^15}")

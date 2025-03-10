import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\1_week_homework_iris\iris.csv"
df = pd.read_csv(file_path)
sns.set_style("whitegrid")

X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]  # 입력값
y = df['Name']  # 출력값 (품종)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # (75%: 학습, 25%: 테스트)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LR모델 생성 및 학습
lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train_scaled, y_train)

y_pred = lr_model.predict(X_test_scaled)

# 정확도 & 보고서 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.4f}")  
actual_classes = sorted(set(y_test))  
print("\n분류 리포트(Classification Report)")
print(classification_report(y_test, y_pred, target_names=encoder.inverse_transform(actual_classes)))

# 시각화
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d",
            xticklabels=encoder.inverse_transform(actual_classes), 
            yticklabels=encoder.inverse_transform(actual_classes))
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("logistic_regression_confusion_matrix.png") 
plt.show()

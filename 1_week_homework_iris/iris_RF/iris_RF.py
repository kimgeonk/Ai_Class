import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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

# RL모델 생성 및 학습
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)

# 정확도 & 보고서 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.4f}")  
actual_classes = sorted(set(y_test))  
print("\n분류 리포트(Classification Report)")
print(classification_report(y_test, y_pred, target_names=encoder.inverse_transform(actual_classes)))

# 시각화
feature_importances = rf_model.feature_importances_  
feature_names = X.columns  
plt.figure(figsize=(7, 5))
sns.barplot(x=feature_importances, y=feature_names, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance (Random Forest)")
plt.savefig("random_forest_feature_importance.png") 
plt.show()

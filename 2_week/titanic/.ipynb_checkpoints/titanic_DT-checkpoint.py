import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\2_week\titanic\titanic.csv"
df = pd.read_csv(file_path)

df = df.assign(Age=df['Age'].fillna(df['Age'].mean())) # Age의 평균값으로 새로운 컬럼(Age) 대체
df = df.assign(Embarked=df['Embarked'].fillna(df['Embarked'].mode()[0])) # Embarked는 최빈값으로 대체
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True) # inplace=True 가 있어야지 변경된 사항이 적용됨

encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns: # 문자로 이루어진 컬럼을 순회 
    df[col] = encoder.fit_transform(df[col]) # encoder를 이용하여 문자를 숫자로 바꿈 

X = df.drop(columns=['Survived'])
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
dt_model = DecisionTreeClassifier(random_state = 42)
dt_model.fit(x_train, y_train)

y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) # 0.7933의 정확도
print(f"Accuracy : {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Died (0)", "Survived (1)"], yticklabels=["Died (0)", "Survived (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred))
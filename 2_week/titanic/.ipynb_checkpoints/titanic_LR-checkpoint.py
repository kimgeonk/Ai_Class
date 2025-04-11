import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\2_week\titanic\titanic.csv"
df = pd.read_csv(file_path)

df = df.assign(Age = df['Age'].fillna(df['Age'].mean()))
df = df.assign(Embarked = df['Embarked'].fillna(df['Embarked'].mode()[0]))
df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace = True)

encoder = LabelEncoder()
for col in df.select_dtypes(include = ['object']).columns:
    df[col] = encoder.fit_transform(df[col])
    
X = df.drop(columns = ['Survived'])
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
lr_model = LogisticRegression(random_state = 42)
lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_test)

accuarcy = accuracy_score(y_test, y_pred) # 0.8101의 정확도
print(f"Accuracy : {accuarcy:.4f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Died (0)", "Survived (1)"], yticklabels=["Died (0)", "Survived (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred))
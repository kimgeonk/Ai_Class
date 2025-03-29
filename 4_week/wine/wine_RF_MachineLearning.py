import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\4_week\wine\wine.csv"
df = pd.read_csv(file_path)

wine_classes = df['Wine'].unique()
wine_classes.sort() 

X = df.drop(columns=['Wine'])
y = df['Wine']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt_model = RandomForestClassifier(random_state=42)
dt_model.fit(x_train, y_train)

y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) # 1.000
print(f"Accuracy : {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=wine_classes, yticklabels=wine_classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Wine Classification")
plt.show()

print(classification_report(y_test, y_pred))
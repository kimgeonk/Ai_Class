import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\3_week\quiz\car_evaluation.csv"
df = pd.read_csv(file_path)

encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])

X = df.drop(columns='unacc')
y = df['unacc']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_model = RandomForestClassifier(random_state=42)
dt_model.fit(x_train, y_train)

y_pred = dt_model.predict(x_test)

accuarcy = accuracy_score(y_test, y_pred) # 0.9624의 정확도
print(f"Accuarcy : {accuarcy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
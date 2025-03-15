import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\2_week\quiz\mobile.csv"
df = pd.read_csv(file_path)

X = df.drop(columns=('price_range'))
y = df['price_range']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
svm_model = SVC(kernel = 'linear')
svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) # 0.97
print(f"SVM_Accuracy = {accuracy:.2f}")
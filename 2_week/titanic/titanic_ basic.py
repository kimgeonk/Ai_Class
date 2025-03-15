import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\2_week\titanic\titanic.csv"
df = pd.read_csv(file_path)

print("결측치 개수 (처리 전):")
print(df.isnull().sum())  # Age, Cabin, Embarked에서 결측치 발생

if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].mean(), inplace=True)  # 나이의 경우 평균값으로 대체

if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # 탑승지는 최빈값으로 대체

if 'Survived' in df.columns:
    print("\n생존 여부 분포:")
    print(df['Survived'].value_counts())

df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True) # inplace=True 가 있어야지 변경된 사항이 적용됨

encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])
    
print("결측치 개수 (처리 후):")
print(df.isnull().sum())
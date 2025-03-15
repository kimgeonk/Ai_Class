import pandas as pd

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\2_week\titanic\titanic.csv" # 로컬을 사용한 방법
url = "https://raw.githubusercontent.com/MyungKyuYi/AI-class/main/titanic.csv" # git.com 대신 raw.git.com을 사용& blom을 삭제 하여 url을 사용!
df = pd.read_csv(url) 
print(df.head())
    
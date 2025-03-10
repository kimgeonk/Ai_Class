import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\1_week_homework_iris\iris.csv"
df = pd.read_csv(file_path)
sns.set_style("whitegrid")

X_train, _, y_train, _ = train_test_split(df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']], 
                                          df['Name'], test_size=0.25, random_state=42)
palette = sns.color_palette("husl", n_colors=df['Name'].nunique())

# 시각화 : 꽃받침(Sepal) 너비 vs 길이
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_train['SepalWidth'], y=X_train['SepalLength'], hue=y_train, palette=palette, s=70, edgecolor='black')
plt.title("Sepal Width vs Sepal Length (Train Data)")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Sepal Length (cm)")
plt.legend(title="Species")
plt.savefig("sepal_visualization.png")  # 꽃받침 데이터 저장
plt.show()

# 시각화 : 꽃잎(Petal) 너비 vs 길이
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_train['PetalWidth'], y=X_train['PetalLength'], hue=y_train, palette=palette, s=70, edgecolor='black')
plt.title("Petal Width vs Petal Length (Train Data)")
plt.xlabel("Petal Width")
plt.ylabel("Petal Length")
plt.legend(title="Species")
plt.savefig("petal_visualization.png")  
plt.show()

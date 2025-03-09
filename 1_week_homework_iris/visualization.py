import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\1_week_homework_iris\iris.csv" 
df = pd.read_csv(file_path)

sns.set_style("whitegrid")
palette = sns.color_palette("husl", n_colors=df['Name'].nunique())

# 시각화 1: 꽃받침 (Sepal) 길이 vs 넓이
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='SepalWidth', y='SepalLength', hue='Name', palette=palette, s=70, edgecolor='black')
plt.title("Sepal Width vs Sepal Length")
plt.xlabel("Sepal Width")
plt.ylabel("Sepal Length")

# 시각화 2: 꽃잎 (Petal) 길이 vs 넓이
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='PetalWidth', y='PetalLength', hue='Name', palette=palette, s=70, edgecolor='black')
plt.title("Petal Width vs Petal Length")
plt.xlabel("Petal Width")
plt.ylabel("Petal Length")

plt.tight_layout()
plt.show()

![Image](https://github.com/user-attachments/assets/5fb3ff15-8abe-4c26-aa1c-ab09bfa7dd9f)

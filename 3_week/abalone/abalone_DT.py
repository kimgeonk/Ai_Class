import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor # regressor(회귀)를 사용함
from sklearn.metrics import mean_squared_error # 분류와 다르게 accuarcy를 사용하지 않음
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"C:\Users\kimge\OneDrive\문서\Desktop\김 건\가천대학교\2025년 4학년 1학기_시간표\인공지능개론\3_week\abalone\abalone.csv"
df = pd.read_csv(file_path)

encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])
    
X = df.drop(columns=['Rings'])
y = df['Rings']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train, y_train)

y_pred = dt_model.predict(x_test)

print('평균제곱근오차', mean_squared_error(y_pred, y_test)) # 6.076555023923445

#히트맵
df_corr=df.corr()
plt.figure(figsize=(7,7))
sns.set(font_scale=0.8)
sns.heatmap(df_corr, annot=True, cbar=False)
plt.show()

#산점도
effect_thing = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
plt.figure(figsize=(8, 6))
for i, feature in enumerate(effect_thing):
    plt.subplot(3, 3, i+1)
    plt.scatter(x_test[feature], y_test, color='blue', alpha=0.6, label=f'test({feature})')
    plt.scatter(x_test[feature], y_pred, color='yellow', alpha=0.6, label=f'pred({feature})')
    plt.xlabel(feature)
    plt.ylabel('Rings')
    plt.legend()
plt.tight_layout()
plt.show()
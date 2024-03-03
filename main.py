import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Создание таблицы данных
data = {
    'Электрическая активность сетчатки': [0, 38.5, 59, 97.4, 119.2, 129.5, 198.7, 248.7, 318, 438.5],
    'Проницаемость сосудов сетчатки': [19.5, 15, 13.5, 23.3, 6.3, 2.5, 13, 1.8, 6.5, 1.8]
}

df = pd.DataFrame(data)

# Вычисление коэффициента корреляции
correlation, t_value = pearsonr(df['Электрическая активность сетчатки'], df['Проницаемость сосудов сетчатки'])
print('Коэффициент корреляции:', correlation)
print('t-критерий:', t_value)

# Вывод выводов о значимости связи
if -0.05 < t_value <= 0.05:
    strength = 'связь статистически значима'
else:
    strength = 'связь статистически не значима'

print('Согласно t-критерию Стьюдента :', strength)

# Регрессионный анализ
X = df[['Электрическая активность сетчатки']]
y = df['Проницаемость сосудов сетчатки']

reg_model = LinearRegression()
reg_model.fit(X, y)

# Построение графика зависимости и линии регрессии
plt.scatter(df['Электрическая активность сетчатки'], df['Проницаемость сосудов сетчатки'], label='Данные')
plt.plot(df['Электрическая активность сетчатки'], reg_model.predict(df[['Электрическая активность сетчатки']]), color='red', label='Линия регрессии')
plt.xlabel('Электрическая активность сетчатки')
plt.ylabel('Проницаемость сосудов сетчатки')
plt.title('Связь между электрической активностью сетчатки и проницаемостью сосудов сетчатки')
plt.legend()
plt.show()

# Вывод уравнения регрессии
slope = reg_model.coef_[0]
intercept = reg_model.intercept_
equation = f'Проницаемость сосудов сетчатки = {slope:.2f} * Электрическая активность сетчатки + {intercept:.2f}'
print('Уравнение регрессии:', equation)

# Вывод выводов о силе и направлении зависимости
if correlation > 0:
    strength = 'положительная'
elif correlation < 0:
    strength = 'отрицательная'
else:
    strength = 'нет'

print('Сила зависимости:', strength)

# Вывод выводов о значимости корреляции
if t_value < 0.05:
    significance = 'статистически значима'
else:
    significance = 'не статистически значима'

print('Значимость корреляции:', significance)
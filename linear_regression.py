import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')
X = data['YearsExperience'].values
y = data['Salary'].values
n = len(X)

m = 0.0
b = 0.0
learning_rate = 0.01
epochs = 1000

for i in range(epochs):
    y_predicted = m * X + b
    D_m = (-2/n) * sum(X * (y - y_predicted))
    D_b = (-2/n) * sum(y - y_predicted)
    m = m - learning_rate * D_m
    b = b - learning_rate * D_b
    

print(m)
print(b)

plt.scatter(X, y, color='blue', label='real data')
plt.plot(X, m * X + b, color='red', label='linear regression')
plt.title('Salary from experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

experience_to_predict = 10
predicted_salary = m * experience_to_predict + b
print(f"predicted {experience_to_predict} years of experience: ${predicted_salary:,.2f}")
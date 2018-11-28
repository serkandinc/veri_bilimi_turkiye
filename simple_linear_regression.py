# Simple Linear Regression

# Kütüphanelerin import edilmesi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Datasetin import edilmesi
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Datasetin Test ve eğitim datasetlerine ayrıştırılması
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# 'Simple Linear Regression' un eğitim setine fit edilmesi 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Test Dataları ile tahmin
y_tahmin = regressor.predict(X_test)

# Eğitim datalarının sonuclarını görselleştirme
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Maaş & Deneyim(Training set)')
plt.xlabel('Deneyim')
plt.ylabel('Maaş')
plt.show()

# Test datalarının sonuclarını görselleştirme
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title('Maaş & Deneyim (Test set)')
plt.xlabel('Deneyim')
plt.ylabel('Maaş')
plt.show()
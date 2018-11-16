import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataFrame = pd.read_csv('D:/GitHub/GitTesis/SVR/14112018D2.csv')
# print(dataFrame)

X = dataFrame.iloc[:, :6].values
y = dataFrame.iloc[:, 6].values
# print(X)

# ----- Fitting Linear Regression to the datasets
# lin_reg = LinearRegression()
# lin_reg.fit(X,y)
# print(lin_reg.fit(X,y))

for k in range(1):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=k)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	for i in [0.1, 1.0, 10, 0.15, 0.2]:
		classifier = SVR(kernel='rbf', C=i, gamma=0.125)
		classifier.fit(X_train, y_train)
		print(i,"=",classifier.score(X_test, y_test))

# ----- Plot visual result LinearModel
# plt.scatter(X,y,color='red')
# plt.plot(X, lin_reg.predict(X), color='blue')
# plt.title('Support Vector Regression')
# plt.xlabel('X Lebel')
# plt.ylabel('Y Label')
# plt.show()
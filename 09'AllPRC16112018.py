import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataLatihan = pd.read_csv('D:/00AllData/DataLatihan.csv')
# dataFrame = pd.read_csv('D:/GitHub/GitTesis/SVR/14112018D2.csv')
# print(dataLatihan)
X = dataLatihan.iloc[:, 0:1].values
y = dataLatihan.iloc[:, 0]. values
print(X,y)

# for k in range(2):
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=k)

# 	sc = StandardScaler()
# 	X_train = sc.fit_transform(X_train)
# 	X_test = sc.transform(X_test)

# 	for i in [1, 0.2, 0.5, 10, 0.1]:
# 		classifier_rbf = SVR(kernel='rbf', C=i, gamma=0.1)
# 		# classifier_lin = SVR(kernel='linear', C=i)
# 		classifier_rbf.fit(X_train, y_train)
# 		# classifier_lin.fit(X_train, y_train)
		
# 		# y_rbf = classifier_rbf.fit(X,y).predict(X)
# 		# y_lin = classifier_lin.fit(X,y).predict(X)
# 		# print(y_rbf)
# 		print('RBF:', classifier_rbf.score(X_test, y_test))
# 		# print('LIN:', classifier_lin.score(X_test, y_test))
# 		# print('+++++++++++++++++++++++++++++++++++++++++')
# 		# print('LIN:',y_lin)

	# 	lw = 2

	# plt.scatter(X,y, color='darkorange', label='Data')
	# plt.plot(X,y_rbf, color='navy', lw=lw, label='RBF Model')
	# plt.plot(X,y_lin, color='c', lw=lw, label='Linear Model')
	# plt.xlabel('data')
	# plt.ylabel('target')
	# plt.title('Support Vector Regression')
	# plt.legend()
	# plt.show()

# plt.scatter(X,y, color='darkorange', label='Data')
# plt.plot(X,y_rbf, color='navy', lw=lw, label='RBF Model')
# plt.plot(X,y_lin, color='c', lw=lw, label='Linear Model')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()
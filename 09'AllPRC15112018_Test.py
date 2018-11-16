import pandas as pd
import numpy as np
from matplotlib import cm, pyplot
from matplotlib.mlab import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from mpl_toolkits.mplot3d import Axes3D

#loading the dataset
dataset = pd.read_csv('D:/00AllData/dataTest.csv')

S = dataset.iloc[:,0:2].values
t = dataset.iloc[:,2].values
print(S, t)

#feature scaling
sc_S = StandardScaler()
sc_t = StandardScaler()
S2 = sc_S.fit_transform(S)
t2 = sc_t.fit_transform(t)

# #fitting the SVR to the dataset
regressor = SVR(kernel = 'rbf')
regressor.fit(S2, t2)
print(regressor.fit(S2, t2))

# #displaying the 3D graph
# x = S[:, 0]
# y = S[:, 1]
# z = t
# zp = sc_t.inverse_transform(regressor.predict(sc_S.transform(S))) #the predictions

# xi = np.linspace(min(x), max(x))
# yi = np.linspace(min(y), max(y))
# X, Y = np.meshgrid(xi, yi)
# ZP = griddata(x, y, zp, xi, yi)

# fig = pyplot.figure()
# ax = Axes3D(fig)
# surf = ax.plot_surface(X, Y, ZP, rstride=1, cstride=1, facecolors=cm.jet(ZP/3200), linewidth=0, antialiased=True)
# ax.scatter(x, y, z)
# ax.set_zlim3d(np.min(z), np.max(z))
# colorscale = cm.ScalarMappable(cmap=cm.jet)
# colorscale.set_array(z)
# fig.colorbar(colorscale)
# pyplot.show()
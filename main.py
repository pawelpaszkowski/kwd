""" Linear regression """
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

housingData = datasets.load_boston()

#linear regression
housingData_X = housingData.data[:, 2]  #vector 442
housingData_X = housingData_X[:, np.newaxis] #matrix 442x1

housingData_X_train, housingData_X_test, housingData_y_train, housingData_y_test = \
    train_test_split(housingData_X, housingData.target, test_size=0.01)

plt.figure(1)
plt.subplot('221')
plt.scatter(housingData_X_train, housingData_y_train)
plt.title('Training dataset')
plt.xlabel('x3')
plt.ylabel('target')
plt.grid()

plt.subplot('222')
plt.scatter(housingData_X_test, housingData_y_test)
plt.title('Testing dataset')
plt.xlabel('x3')
plt.ylabel('target')
plt.grid()

regr = linear_model.LinearRegression()
regr.fit(housingData_X_train, housingData_y_train)
housingData_y_pred = regr.predict(housingData_X_test)

plt.subplot('223')
plt.scatter(housingData_X_train, housingData_y_train)
plt.plot(housingData_X_train, regr.predict(housingData_X_train), color='green',
         linewidth=3)
plt.title('Learned linear regression')
plt.xlabel('x3')
plt.ylabel('target')
plt.tight_layout()
plt.grid()

plt.subplot('224')
plt.scatter(housingData_X_test, housingData_y_test)
plt.plot(housingData_X_test, regr.predict(housingData_X_test), color='green', linewidth=3)
plt.title('Learned linear regression')
plt.xlabel('x3')
plt.ylabel('target')
plt.tight_layout()
plt.grid()

plt.show()
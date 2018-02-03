""" Linear regression """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#housingData = datasets.load_boston()
#TEST
housingData = pd.read_csv('boston_house_prices.csv')

#linear regression
housingData_X = np.array(housingData['LSTAT'].tolist())
#print(type(housingData.data[:, 12]))
#housingData_X = housingData.data[:, 12]  #vector 442
housingData_X = housingData_X[:, np.newaxis] #matrix 442x1
#print(type(housingData_X))
#print(housingData_X)
housingDataTarget =np.array(housingData['MEDV'].tolist())

housingData_X_train, housingData_X_test, housingData_y_train, housingData_y_test = \
    train_test_split(housingData_X, housingDataTarget, test_size=0.3)



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

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(housingData_y_test, regr.predict(housingData_X_test)))
print('r2: %.2f' % r2_score(housingData_y_test, regr.predict(housingData_X_test)))

#test
# regression with all features
housingData.drop('MEDV', axis=1, inplace=True)
housingData_all_features = np.array(housingData)
print(housingData_all_features)
housingData_all_features_target = housingDataTarget
housingData_all_features_train, housingData_all_features_test, housingData_all_features_target_train, housingData_all_features_target_test = train_test_split(
    housingData_all_features, housingData_all_features_target)

linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(housingData_all_features_train, housingData_all_features_target_train)

predicts = linear_regression_model.predict(housingData_all_features_test)
print("\nRegression with all features:")
print("coefficients with all features:", linear_regression_model.coef_)
print("Mean error over test data: %.2f" % mean_squared_error(housingData_all_features_target_test, predicts))
print("R2  test data: %.2f" % r2_score(housingData_all_features_target_test, predicts))

plt.show()
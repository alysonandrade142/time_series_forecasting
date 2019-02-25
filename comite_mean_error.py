from pandas import Series
from matplotlib import pyplot
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose, seasonal_mean
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import accuracy_score
from scipy.stats import mstats

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import pandas
import numpy

def normalize(values):
    values = values.reshape((len(values), 1))
    # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    return normalized

def create_dataset(data,  n_in=1, n_out=1, dropnan=True):
	n_vars = 1 
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg 

dataframe = pandas.read_csv("./datasets/annual-total-melanoma-incidence-.csv", sep=",", header=0)
dataset = dataframe['value']

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = create_dataset(dataset[0:train_size],1), create_dataset(dataset[0:test_size],1)
trainX, trainY = train['var1(t-1)'], train['var1(t)'] 
testX, testY = test['var1(t-1)'], test['var1(t)']

clfMLP = MLPRegressor(max_iter=400)
clfMLP.fit(normalize(numpy.array(trainX)), numpy.array(trainY))
result1 = clfMLP.predict(normalize(numpy.array(testX)))
mlpMAE = mean_absolute_error(result1, testY)
winMlpMAE = mean_absolute_error(result1, testY)

clfMLP2 = MLPRegressor(max_iter=400)
clfMLP2.fit(normalize(numpy.array(trainX)), numpy.array(trainY))
result2 = clfMLP2.predict(normalize(numpy.array(testX)))
mlp2MAE = mean_absolute_error(result2, testY)
winMlp2MAE = mean_absolute_error(result2, testY)

clfTree = DecisionTreeRegressor()
clfTree.fit(normalize(numpy.array(trainX)), numpy.array(trainY))
result3 = clfTree.predict(normalize(numpy.array(testX)))
treeMAE = mean_absolute_error(result3, testY)
winTreeMAE = mean_absolute_error(result3, testY)

clfLinear = LinearRegression()
clfLinear.fit(normalize(numpy.array(trainX)), numpy.array(trainY))
result4 = clfLinear.predict(normalize(numpy.array(testX)))
linearMAE = mean_absolute_error(result4, testY)
winLinearMAE = mean_absolute_error(result4, testY)

clfSVR = SVR()
clfSVR.fit(normalize(numpy.array(trainX)), numpy.array(trainY))
result5 = clfSVR.predict(normalize(numpy.array(testX)))
svrMAE = mean_absolute_error(result5, testY)
winSvrMAE = mean_absolute_error(result5, testY)

print('MLP MAE: %.3f' % (mlpMAE))
print('MLP2 MAE: %.3f' % (mlp2MAE))
print('TREE MAE: %.3f' % (treeMAE))
print('SVR MAE: %.3f' % (svrMAE))
print('Linear MAE: %.3f' % (linearMAE))

totalMAE = [svrMAE, linearMAE, treeMAE, mlp2MAE, mlpMAE]
print('\n')
print('-----------------------------------')
print('---------------TOTAL---------------')
print('\n')
print('TOTAL MEAN MAE: %.3f' % ((sum(totalMAE))/5))

MAES = numpy.array([svrMAE, linearMAE, treeMAE, mlp2MAE, mlpMAE])
print('TOTAL WINSORIZED MAE: %.3f' % ((sum(mstats.winsorize(MAES, limits=[0.50, 0.50]))/5)))
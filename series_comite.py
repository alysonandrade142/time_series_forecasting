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

ONLY_MLP = []
ONLY_MLP2 = []
ONLY_TREE = []
ONLY_LINEAR = []
ONLY_SVR = []

RESULTS_MEDIA = []
RESULTS_WIN = []

clfMLP = MLPRegressor(max_iter=400)
clfMLP.fit(normalize(numpy.array(trainX)), numpy.array(trainY))
clfMLP2 = MLPRegressor(max_iter=400)
clfMLP2.fit(normalize(numpy.array(trainX)), numpy.array(trainY))
clfTree = DecisionTreeRegressor()
clfTree.fit(normalize(numpy.array(trainX)), numpy.array(trainY))
clfLinear = LinearRegression()
clfLinear.fit(normalize(numpy.array(trainX)), numpy.array(trainY))
clfSVR = SVR()
clfSVR.fit(normalize(numpy.array(trainX)), numpy.array(trainY))

for element in testX:

	result1 = clfMLP.predict(numpy.array(element).reshape(-1, 1))
	result2 = clfMLP2.predict(numpy.array(element).reshape(-1, 1))
	result3 = clfTree.predict(numpy.array(element).reshape(-1, 1))
	result4 = clfLinear.predict(numpy.array(element).reshape(-1, 1))
	result5 = clfSVR.predict(numpy.array(element).reshape(-1, 1))

	ONLY_MLP.append(result1)
	ONLY_MLP2.append(result2)
	ONLY_TREE.append(result3)
	ONLY_LINEAR.append(result4)
	ONLY_SVR.append(result5)

	TOTAL = [result1, result2, result3, result4, result5]

	media = sum(TOTAL)/5
	win = (sum(mstats.winsorize(numpy.array(TOTAL), limits=[0.50, 0.50]))/5)
	RESULTS_MEDIA.append(media[0])
	RESULTS_WIN.append(win[0])

media = mean_absolute_error(RESULTS_MEDIA, testY)
win = mean_absolute_error(RESULTS_WIN, testY)
print('RESULTS MEDIA COMITE: %.3f' % (media))
print('RESULTS WINSORIZED COMITE: %.3f' % (win))

MAE1 = mean_absolute_error(ONLY_MLP, testY)
MAE2 = mean_absolute_error(ONLY_MLP2, testY)
MAE3 = mean_absolute_error(ONLY_TREE, testY)
MAE4 = mean_absolute_error(ONLY_LINEAR, testY)
MAE5 = mean_absolute_error(ONLY_SVR, testY)

print('RESULTS MLP: %.3f' % (MAE1))
print('RESULTS MLP2: %.3f' % (MAE2))
print('RESULTS TREE: %.3f' % (MAE3))
print('RESULTS LINEAR: %.3f' % (MAE4))
print('RESULTS SVR: %.3f' % (MAE5))


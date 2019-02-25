from pandas import Series
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose, seasonal_mean
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from pandas import concat
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import accuracy_score


import pandas
import numpy

scaler = MinMaxScaler(feature_range=(0, 1))
dataframe = pandas.read_csv("./datasets/annual-total-melanoma-incidence-.csv", sep=",", header=0)
dataset = dataframe['value']
y = dataframe['year']

def normalize(values):
    values = values.reshape((len(values), 1))
    scaler_values = scaler.fit(values)
    normalized = scaler.transform(values)
    return normalized

def undo(values):
	values = values.reshape((len(values), 1))
	scaler_values = scaler.fit(values)
	values = scaler.inverse_transform(values)
	return values

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

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = create_dataset(dataset[0:train_size],1), create_dataset(dataset[0:test_size],1)
trainX, trainY = train['var1(t-1)'], train['var1(t)'] 
testX, testY = test['var1(t-1)'], test['var1(t)']

direct = create_dataset(dataset[0:test_size], 1, 11, True)

clf = MLPRegressor()
clf.fit(normalize(numpy.array(trainX)), numpy.array(trainY))
aux = testX[1]
recursive_predictions = []
direct_prediction = []

# DIRETO
for column in direct:
	direct_prediction.append(direct[column][1])

# RECURSIVO
for i in range(0, 12):
	aux = clf.predict(numpy.array(aux).reshape(-1, 1))
	recursive_predictions.append(aux)

print(mean_absolute_error(direct_prediction, testY))
print(mean_absolute_error(recursive_predictions, testY))

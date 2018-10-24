from pandas import Series
from matplotlib import pyplot
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose, seasonal_mean
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import os
import random

asma = Series.from_csv(
    "./datasets/uk-deaths-from-bronchitis-emphys.csv", sep=",")

values = asma.values
size = int(len(values) * 0.66)
train, test = values[0:size], values[size:len(values)]
results = []


def arima_method(train, test, ar, i, ma):
    predictions = list()
    history = []
    history = [x for x in train]

    # print("AR %i , I %i , MA %i " % (ar, i, ma))

    for v in range(len(test)):
        model = ARIMA(history, order=(ar, i, ma)).fit(disp=0)
        output = model.forecast()
        predictions.append(output[0])
        history.append(test[v])
        # print('predicated=%f, expected=%f' % (output[0], test[v]))

    return predictions


fig, axes = pyplot.subplots(nrows=2, ncols=3)

for i in range(0, 5):
    predictions = arima_method(
        train=train, test=test, ar=5, i=2, ma=0)
    error = mean_squared_error(test, predictions)
    print('Iteration %d MSE: %.3f' % (i, error))
    results.append(predictions)

i = 0
for x in range(0, 2):
    for y in range(0, 3):
        if(i < len(results)):
            axes[x][y].plot(results[i], color='red')
            axes[x][y].plot(test)
            axes[x][y].set_title('Result %d' % i)
            i += 1
        else:
            break

pyplot.show()

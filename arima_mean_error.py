from pandas import Series
from matplotlib import pyplot
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose, seasonal_mean
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


import os
import random

asma = Series.from_csv(
    "./datasets/uk-deaths-from-bronchitis-emphys.csv", sep=",")
os.system('cls')
values = asma.values
size = int(len(values) * 0.66)
train, test = values[0:size], values[size:len(values)]

def arimaMethod(train, test, AR, I, MA):

    predictions = list()
    history = []
    history = [x for x in train]

    for v in range(len(test)):
        model = ARIMA(history, order=(AR,I,MA)).fit(disp=0)
        output = model.forecast()
        predictions.append(output[0])
        history.append(test[v])
        print('predicated=%f, expected=%f' % (output[0], test[v]))

    return predictions


for x in range(0, 4):
    
    ar = random.randint(1, 5)
    i = random.randint(0, 1)
    ma = random.randint(0, 1)

    predictions = arimaMethod(train=train, test=test,
    AR=ar,
    I=i,
    MA=ma)

    error = mean_squared_error(test, predictions)
    error2 = mean_absolute_error(test, predictions)
    print('Test %f MSE: %.3f' % (x, error))
    print('Test %f RMSE: %.3f' % (x, sqrt(error)))
    print('Test %f MAE: %.3f' % (x, error2))
    print("AR={}, I={}, MA={}".format(ar, i, ma))

    # PLOT
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()

#
    # O parametro que retornou o menor MAE foi o AR=2 I=0 MA=1, levando em consideração que o random é entre os valores 
    # 1-5 para o AR, 0-1 para o I e 0-1 para o MA.
#

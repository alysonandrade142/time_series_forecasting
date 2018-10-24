import os

#region INSTALL DEPENDENCIES
os.system('python -m pip install pandas')
os.system('python -m pip install matplotlib')
os.system('python -m pip install statsmodels')
os.system('python -m pip install sklearn')
os.system('cls')
#endregion 

from pandas import Series
from matplotlib import pyplot
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose, seasonal_mean
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

asma = Series.from_csv("./datasets/uk-deaths-from-bronchitis-emphys.csv", sep=",")

# SIMPLE PLOT =======
asma.plot()
pyplot.show()

# LAG PLOT =========
lag_plot(asma)
pyplot.show()

# AUTOCORRELATION PLOT =========
autocorrelation_plot(asma)
pyplot.show()

# DECOMPOSE SERIE PLOT =========
result1 = seasonal_decompose(asma, model="additive")
result1.plot()
pyplot.show()

# DICKEY FUNCTION LOG =========
asmaValues = asma.values
asmaDickeyFuller = adfuller(asmaValues)
print("===========================")
print('ASMA ADF Statistic: %f' % asmaDickeyFuller[0])
print('ASMA p-value: %f' % asmaDickeyFuller[1])
print("===========================")
from pandas import Series
from matplotlib import pyplot
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose, seasonal_mean
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import os

asma = Series.from_csv("./datasets/uk-deaths-from-bronchitis-emphys.csv", sep=",")
suicide = Series.from_csv("./datasets/annual-us-suicide-rate-per-10010.csv", sep=",")
melanoma = Series.from_csv("./datasets/annual-total-melanoma-incidence-.csv", sep=",", header=0)

os.system('cls')

result1 = seasonal_decompose(asma, model="additive")
result1.plot()
pyplot.show()
asmaValues = asma.values
asmaDickeyFuller = adfuller(asmaValues)
print("===========================")
print('ASMA ADF Statistic: %f' % asmaDickeyFuller[0])
print('ASMA p-value: %f' % asmaDickeyFuller[1])
print("===========================")


result2 = seasonal_decompose(suicide, model="additive")
result2.plot()
pyplot.show()
suicideValues = suicide.values
suicideDickeyFuller = adfuller(suicideValues)
print("===========================")
print('SUICIDE ADF Statistic: %f' % suicideDickeyFuller[0])
print('SUICIDE p-value: %f' % suicideDickeyFuller[1])
print("===========================")


result3 = seasonal_decompose(melanoma, model="additive")
result3.plot()
pyplot.show()
melanomaValues = melanoma.values
melanomaDickeyFuller = adfuller(melanomaValues)
print("===========================")
print('MELANOMA ADF Statistic: %f' % melanomaDickeyFuller[0])
print('MELANOMA p-value: %f' % melanomaDickeyFuller[1])
print("===========================")

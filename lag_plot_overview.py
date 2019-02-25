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

asma.plot()
pyplot.show()
lag_plot(asma)
pyplot.show()

# Essa serie possui um comportamento regular, com sazonalidade e sem tendência.

suicide.plot()
pyplot.show()
lag_plot(suicide)
pyplot.show()

# Essa serie possui um comportamento irregular, inicialmente sem sazonalidade e com tendência.

melanoma.plot()
pyplot.show()
lag_plot(melanoma)
pyplot.show()

# Essa serie possui um comportamento irregular, inicialmente com sazonalidade e com tendência.

####   LAG PLOT ####

## O lag plot nos auxilia em vários momentos, mas destes gostaria de explicitar alguns que são:

## * Outliers, facilmente conseguimos descobrir outliers na TS.
## * Correlação, ou seja, o impacto (influência) de um valor com o outro.
## * Randomização dos dados, existem outros plots que pode nos auxiliar a respeito disto, mas com o lag plot também é possível.

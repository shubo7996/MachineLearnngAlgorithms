import pandas as pd
import quandl
import math, datetime 
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

#using graph style
style.use('ggplot')

#loading the dataset
df = quandl.get('WIKI/GOOGL')

#useful features from the dataset
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#percentage volatility
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

#daily percentage change
df['PCT_CHNG'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#relevant dataframes 
df = df[['Adj. Close', 'HL_PCT', 'PCT_CHNG', 'Adj. Volume']]

forecast_col = 'Adj. Close'
#replacing the missing data
df.fillna(-99999, inplace=True)

#predicting 10% of the dataframe(data which came days ago to predict today)
forecast_out = int(math.ceil(0.01*len(df)))

#predicting the future close price of the stock
df['label'] = df[forecast_col].shift(-forecast_out)

#print (forecast_out)

#features col(everything excld. label col)
x = np.array(df.drop(['label'],1))
#scaling your data which is being fed
x = preprocessing.scale(x)
#predicting against these values
xLately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
#df.dropna(inplace=True)
#labels col
y = np.array(df['label'])
print (len(x), len(y))

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

clf = LinearRegression()
#using SVM Algorith
#clf = svm.SVR()
clf.fit(x_train, y_train)
with open('linearregression.pickle', 'wb') as file:
	pickle.dump(clf,file)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(x_test, y_test)

#predicted value for the next 30 days
forecast_set = clf.predict(xLately)
print (forecast_set,accuracy,forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day_in_sec = 86400
next_unix = last_unix + one_day_in_sec

#making dates as x axis in the graph
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day_in_sec
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("price")
plt.show()

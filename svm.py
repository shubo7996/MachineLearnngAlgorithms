import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm

df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['Class'], 1))
#xs = x.astype(np.float64)
y = np.array(df['Class'])
#ys = y.astype(np.float64)

x_test, x_train, y_test, y_train = cross_validation.train_test_split(x,y,test_size = 0.2)

clf = svm.SVC()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test, y_test)
print (accuracy)
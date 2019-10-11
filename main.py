import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import numpy as np


start = datetime.date(2018, 1, 1)
end = datetime.date(2019, 8, 14)
stock = web.DataReader("AAPL", 'yahoo', start, end)
y = np.asarray(stock['Close'])
y = y.reshape(len(y), 1)
X = np.arange(0, len(y))
X = X.reshape(len(X), 1)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
y = sc.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

def tuning():
    from sklearn.svm import SVR
    svr = SVR(kernel = 'rbf')
    svr.fit(X_train, y_train)
    
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = svr, X = X_train, y = y_train, cv = 10)
    accuracies.mean()
    accuracies.std()
    
    from sklearn.model_selection import GridSearchCV
    parameters = [{'C': [0.1, 1, 10, 50, 100], 'gamma': [1000, 100, 10, 1, 0.5, 0.1, 0.01, 0.001, 0.0005, 0.0001], 'epsilon': [0.001, 0.1, 0.5, 0, 1, 2, 4, 5, 10, 50, 100]}]
    gridsearch = GridSearchCV(estimator = svr, param_grid = parameters, scoring = 'neg_mean_squared_error', cv = 10, n_jobs = 1)
    gridsearch = gridsearch.fit(X_train, y_train)
    best_params = gridsearch.best_params_
    return best_params

from sklearn.svm import SVR
tuning = tuning()
model = SVR(kernel = 'rbf', C = tuning['C'], epsilon = tuning['epsilon'], gamma = tuning['gamma'])
model.fit(X_train, y_train)
X_pred = model.predict(X_train)
y_pred = model.predict(X_test)


X_train = sc.inverse_transform(X_train)
y_train = sc.inverse_transform(y_train)
X_test = sc.inverse_transform(X_test)
y_test = sc.inverse_transform(y_test)
y_pred = y_pred.reshape(len(y_pred), 1)
y_pred = sc.inverse_transform(y_pred)
X_pred = X_pred.reshape(len(X_pred), 1)
X_pred = sc.inverse_transform(X_pred)

class predictions():
    def getX_pred(self):
        return X_pred
    def gety_pred(self):
        return y_pred

plt.scatter(X_train, y_train, color = 'black')
plt.scatter(X_test, y_test)
plt.plot(X_train, X_pred)
plt.plot(X_test, y_pred, color = 'red')
plt.show()




    
    
    
    
    
    


    
    
        
    
    







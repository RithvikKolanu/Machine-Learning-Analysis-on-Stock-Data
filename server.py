from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/prediction")
def runmodel():
    import pandas_datareader.data as web
    import datetime
    import matplotlib.pyplot as plt
    import numpy as np
    
    stuff = request.args.get("text")
    start = datetime.date(2018, 1, 1)
    end = datetime.date(2019, 8, 15)
    stock = web.DataReader(stuff, 'yahoo', start, end)
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

    from sklearn.svm import SVR
    svr = SVR(kernel = 'rbf')    

    from sklearn.model_selection import GridSearchCV
    parameters = [{'C': [0.1, 1, 10, 100], 'gamma': [100, 10, 1, 0.1], 'epsilon': [0.1, 0, 1, 10, 100]}]
    gridsearch = GridSearchCV(estimator = svr, param_grid = parameters, scoring = 'neg_mean_squared_error', cv = 3, n_jobs =1)
    gridsearch = gridsearch.fit(X_train, y_train)
    tuning = gridsearch.best_params_

    print(tuning['C'], tuning['epsilon'], tuning['gamma'])
    model = SVR(kernel = 'rbf', C = tuning['C'], epsilon = tuning['epsilon'], gamma = tuning['gamma'])
    model.fit(X_train, y_train)
    X_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    X_pred = X_pred.reshape(len(X_pred), 1)
    X_pred = sc.inverse_transform(X_pred)
    X_pred = list(X_pred.reshape(len(X_pred), ))

    y_pred = y_pred.reshape(len(y_pred), 1)
    y_pred = sc.inverse_transform(y_pred)
    y_pred = list(y_pred.reshape(len(y_pred), ))

    X_train = sc.inverse_transform(X_train)
    X_train = list(X_train.reshape(len(X_train), ))

    y_train = sc.inverse_transform(y_train)
    y_train = list(y_train.reshape(len(y_train), ))

    X_test = sc.inverse_transform(X_test)
    X_test = list(X_test.reshape(len(X_test), ))

    y_test = sc.inverse_transform(y_test)
    y_test = list(y_test.reshape(len(y_test), ))  

    return render_template('prediction.html', Xpred=X_pred, ypred = y_pred, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0')

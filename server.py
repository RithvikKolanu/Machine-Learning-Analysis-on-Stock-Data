from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/prediction")
def runmodel():
    stuff = request.args.get("text")
    from main import getdata
    X_pred = getdata(stuff).X_pred
    X_train = getdata(stuff).X_train
    X_test = getdata(stuff).X_test
    y_pred = getdata(stuff).y_pred
    y_test = getdata(stuff).y_test
    y_train = getdata(stuff).y_train
 
    return render_template('prediction.html', Xpred=X_pred, ypred = y_pred, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0')

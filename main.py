from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load('iris_model.pkl')

app = Flask(__name__, template_folder='templates')


@app.route('/')
def main():
    return render_template('homepage.html')


@app.route('/guess', methods=['POST'])
def home():
    data1 = request.form['data1']
    data2 = request.form['data2']
    data3 = request.form['data3']
    data4 = request.form['data4']

    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)

    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)

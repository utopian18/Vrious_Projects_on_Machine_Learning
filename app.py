
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
linear_regression = pickle.load(open('/home/toufique/mysite/static/lr_model_prediction.pkl', 'rb'))
random_forest = pickle.load(open('/home/toufique/mysite/static/random_forest_regressor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method =='POST':
        epi = request.form['epi']
        ppi = request.form['ppi']
        warp_count = request.form['warp_count']
        weft_count = request.form['weft_count']
        shrinkage = request.form['shrinkage']
        finish_order = request.form['finish_order']
        remark = request.form['remark']
        algorithm = request.form['model']
        sample_data = [warp_count, weft_count, epi, ppi, remark, finish_order, shrinkage]
        sample_data = [float(int(i)) for i in sample_data]
        if algorithm =='linear_regression':
            predict = round(linear_regression.predict([sample_data])[0],3)
        elif algorithm =='random_forest':
            predict = round(random_forest.predict([sample_data])[0], 3)

    return render_template('predict.html', algorithm=algorithm, predict= predict )

@app.route('/dataset')
def dataset():
    dataset = pd.read_csv('short_version_weaving.csv')
    dataset = dataset.iloc[:, 1:]
    return render_template('dataset.html', dataset=dataset)
if __name__ =='__main__':
    app.run()


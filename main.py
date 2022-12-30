from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from script.func import predictTweet
from kmeans import predict_label, predict_label_svm

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hi, It\'s me. I\'m the problem it\'s me.'

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    prediction = predictTweet(tweet)
    # return the result in json format
    predict_using_svm = predict_label_svm(tweet)
    return jsonify({'prediction': prediction, 'predict_using_svm': predict_using_svm})

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    tweet = request.form['tweet']
    prediction = predict_label_svm(tweet)
    # return the result in json format
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))

    
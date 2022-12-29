from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from script.func import predictTweet

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    prediction = predictTweet(tweet)
    # return the result in json format
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8999, debug=True)

    
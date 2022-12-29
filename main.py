from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from script.func import predictTweet

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return 'Hi, It\'s me. I\'m the problem it\'s me.'

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    prediction = predictTweet(tweet)
    # return the result in json format
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))

    
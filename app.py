# importing required libraries to create an api
import numpy as np
from flask import Flask, jsonify, request, render_template

from joblib import load

app = Flask(__name__)
model = load('house_price_pred_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"]) # routing api to 127.0.0.1/pred and using POST method
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    price = model.predict(final_features) # using pre-trained model for prediction
    
    # return jsonify({"Price of the House: ": price })
    
    return render_template('index.html', prediction_text='Predicted house price ${}'.format(price))


if __name__ == "__main__":
    app.run(debug=True)
    
    
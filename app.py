import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
with open(r"C:\Users\99210\Downloads\mohan\trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    moisture = float(request.form['moisture'])

    # Make prediction
    prediction = model.predict([[temperature, humidity, moisture]])[0]

    # Prepare response
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

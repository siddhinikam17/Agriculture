from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
  
  
# Load the trained model (model1.pkl) with detailed error handling
try:
    with open('model1.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading the model:", e)
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model not loaded"
    
    try:
        # Get the input data from the form
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        temperature = float(request.form['temperature'])
        ph = float(request.form['ph'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])

        # Make prediction using the loaded model (model1.pkl)
        prediction = model.predict([[N, P, K, temperature, ph, humidity, rainfall]])

        # Convert the prediction result to a string (class label)
        predicted_class = str(prediction[0])

        # Return the predicted class label
        return predicted_class
    except Exception as e:
        print("Error during prediction:", e)
        return "Error during prediction: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)

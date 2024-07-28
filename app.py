from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
from io import StringIO
 
app = Flask(__name__)    
 
# URL to the dataset
DATA_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

# Load and preprocess the dataset
data = pd.read_csv(DATA_URL, delimiter=';') 

# Define features and target 
X = data.drop('quality', axis=1)  
y = data['quality'] 
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model 
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate the model
predictions = model.predict(X_test)
# mse = mean_squared_error(y_test, predictions)
# print(f'Mean Squared Error: {mse}') 

@app.route('/') 
def home(): 
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model from the file
    with open('model.pkl', 'rb') as f: 
        model = pickle.load(f)
    
    # Get the input data from the request
    data = request.get_json(force=True)  
    
    # Extract features and convert to the appropriate format
    features = [data['fixed acidity'], data['volatile acidity'], data['citric acid'], data['residual sugar'], 
                data['chlorides'], data['free sulfur dioxide'], data['total sulfur dioxide'], data['density'], 
                data['pH'], data['sulphates'], data['alcohol']]
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features) 
     
    # Return the prediction as JSON
    return jsonify({'quality': prediction[0]}) 

if __name__ == '__main__':
    app.run(debug=True)

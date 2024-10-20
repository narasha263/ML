from flask import Flask, request, jsonify
import joblib  # for loading your trained model

app = Flask(__name__)
model = joblib.load('/model_predictions.xlsx')  # Load your trained model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from request
    # Process data and make prediction
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

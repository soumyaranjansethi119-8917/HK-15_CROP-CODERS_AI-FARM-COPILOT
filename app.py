from flask import Flask, request
import numpy as np
from joblib import load

app = Flask(__name__)

# Load trained model & label encoder
model = load("model.pkl")
le = load("label_encoder.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    
    prediction_text = ""

    if request.method == 'POST':
        try:
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            prediction = model.predict(input_data)
            crop_name = le.inverse_transform(prediction)

            prediction_text = f"🌾 Recommended Crop: {crop_name[0]}"

        except:
            prediction_text = "⚠ Please enter valid input values"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crop Recommendation System</title>
        <style>
            body {{
                font-family: Arial;
                background: linear-gradient(to right, #4CAF50, #2E7D32);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                width: 400px;
                text-align: center;
                box-shadow: 0px 5px 20px rgba(0,0,0,0.3);
            }}
            input {{
                width: 100%;
                padding: 10px;
                margin: 8px 0;
                border-radius: 5px;
                border: 1px solid #ccc;
            }}
            button {{
                width: 100%;
                padding: 12px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
            }}
            button:hover {{
                background: #388E3C;
            }}
            h1 {{
                color: #2E7D32;
            }}
            h2 {{
                margin-top: 20px;
                color: #1B5E20;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌱 Crop Recommendation System</h1>
            <form method="POST">
                <input type="number" step="any" name="N" placeholder="Nitrogen (N)" required>
                <input type="number" step="any" name="P" placeholder="Phosphorus (P)" required>
                <input type="number" step="any" name="K" placeholder="Potassium (K)" required>
                <input type="number" step="any" name="temperature" placeholder="Temperature (°C)" required>
                <input type="number" step="any" name="humidity" placeholder="Humidity (%)" required>
                <input type="number" step="any" name="ph" placeholder="pH Value" required>
                <input type="number" step="any" name="rainfall" placeholder="Rainfall (mm)" required>
                <button type="submit">Predict Crop</button>
            </form>
            <h2>{prediction_text}</h2>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)
 
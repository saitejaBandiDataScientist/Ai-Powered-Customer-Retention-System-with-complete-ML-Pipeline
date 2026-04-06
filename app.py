from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature columns in exact order expected by the model
FEATURE_COLUMNS = [
    'SeniorCitizen', 'tenure_yeo_tri', 'MonthlyCharges_box_CapMS',
    'TotalCharges_rep_yeo_CapMS', 'gender_Male', 'Partner_Yes',
    'Dependents_Yes', 'PhoneService_ordinal', 'MultipleLines_ordinal',
    'InternetService_ordinal', 'OnlineSecurity_ordinal',
    'OnlineBackup_ordinal', 'DeviceProtection_ordinal',
    'TechSupport_ordinal', 'StreamingTV_ordinal', 'StreamingMovies_ordinal',
    'Contract_ordinal', 'PaperlessBilling_ordinal', 'PaymentMethod_ordinal',
    'Sim_ordinal'
]

SIM_ORDINAL_MAP = {
    'jio': 0,
    'airtel': 1,
    'vi': 2,
    'bsnl': 3
}

def load_model():
    try:
        if os.path.exists('Model.pkl'):
            with open('Model.pkl', 'rb') as f:
                return pickle.load(f)
        logger.warning("Model.pkl not found. Using dummy model for demo.")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def load_scaler():
    try:
        if os.path.exists('standard_scaler.pkl'):
            with open('standard_scaler.pkl', 'rb') as f:
                return pickle.load(f)
        logger.warning("standard_scaler.pkl not found.")
        return None
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        return None

model = load_model()
scaler = load_scaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")

        # Build feature array in correct order
        features = []
        for col in FEATURE_COLUMNS:
            val = data.get(col, 0)
            features.append(float(val))

        features_array = np.array(features).reshape(1, -1)

        if model is not None:
            if scaler is not None:
                features_array = scaler.transform(features_array)
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0].tolist()
            churn = bool(prediction == 1)
            churn_prob = probability[1]
        else:
            # Demo mode: simple rule-based mock
            tenure = float(data.get('tenure_yeo_tri', 0))
            monthly = float(data.get('MonthlyCharges_box_CapMS', 0))
            contract = float(data.get('Contract_ordinal', 0))
            churn_prob = max(0.05, min(0.95, 0.7 - (tenure * 0.05) - (contract * 0.15) + (monthly * 0.003)))
            churn = churn_prob > 0.5

        sim_name = data.get('sim_name', 'unknown').capitalize()
        logger.info(f"Prediction: churn={churn}, probability={churn_prob:.4f}, sim={sim_name}")

        return jsonify({
            'success': True,
            'churn': churn,
            'churn_probability': round(churn_prob * 100, 2),
            'retain_probability': round((1 - churn_prob) * 100, 2),
            'sim': sim_name,
            'risk_level': 'High' if churn_prob > 0.7 else ('Medium' if churn_prob > 0.4 else 'Low'),
            'message': f"Customer is {'likely to CHURN' if churn else 'likely to STAY'} with {round(churn_prob*100,1)}% churn probability."
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
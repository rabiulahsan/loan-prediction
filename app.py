from flask import Flask,request, jsonify
import numpy as np
import pandas as pd 

from src.exception import CustomException
from src.logger import logging
from src.pipelines.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

# Route for a home page
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Loan Eligibility Prediction API!"})

# Route for making predictions
@app.route('/predictdata', methods=['GET','POST'])
def predictdata():
    try:
        #log the route hit
        print("hitting the route successfully...")  
        # Retrieve data from the request JSON
        data = request.json
        
        
        # Create a CustomData instance with the input data
        custom_data = CustomData(
            gender=data.get('gender'),
            age=int(data.get('age')),
            income=int(data.get('income')),
            income_stability=data.get('income_stability'),
            loan_amount_request=int(data.get('loan_amount_request')),
            current_loan=int(data.get('current_loan')),
            dependents=int(data.get('dependents')),
            credit_score=int(data.get('credit_score')),
            co_applicant=int(data.get('co_applicant')),
            property_price=int(data.get('property_price'))
        )

        
        # Convert the input data to a DataFrame
        pred_df = custom_data.make_data_frame()
        
        # Initialize the prediction pipeline and make a prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict_approval(pred_df)
        
        # Return the result as a JSON response
        return jsonify({"prediction": results[0],'results':results})

    except Exception as e:
        raise CustomException(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
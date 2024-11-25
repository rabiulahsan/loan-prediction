import pandas as pd 
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass;

    def predict_approval(self, features):
        try:
            model_path = 'artifacts\clf_models\GradientBoostingClassifier_model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Loading model and preprocessor file...")
            # print(model)

            data_scaled = preprocessor.transform(features)
            # print(data_scaled)
            preds = model.predict(data_scaled)
            

            logging.info("Prediction Completed...")

            return preds

        except Exception as e:
            raise CustomException(e)


class CustomData:
    def __init__(  self,
        gender: str,
        age: str,
        income_stability: str,
        co_applicant: int,
        income: int,
        current_loan: int,
        dependents: int,
        credit_score: int,
        loan_amount_request: int,
        property_price: int):

        self.gender = gender
        self.age = age
        self.income_stability = income_stability
        self.co_applicant = co_applicant
        self.income = income
        self.current_loan = current_loan
        self.dependents = dependents
        self.credit_score = credit_score
        self.loan_amount_request = loan_amount_request
        self.property_price = property_price

    def make_data_frame(self):
        try:
            custom_data_input_dict = {
            "gender": [self.gender],
            "age": [self.age],
            "income_stability": [self.income_stability],
            "co_applicant": [self.co_applicant],
            "income": [self.income],
            "current_loan": [self.current_loan],
            "dependents": [self.dependents],
            "credit_score": [self.credit_score],
            "loan_amount_request": [self.loan_amount_request],
            "property_price": [self.property_price],
        }


            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e)


# Main function for testing
if __name__ == "__main__":
    try:
        # Create a CustomData instance with sample data
        custom_data = CustomData(
            gender="M",
            age='Middle aged',
            income_stability="High",
            co_applicant=1,
            income=20000,
            current_loan=500,
            dependents=2,
            credit_score=500,
            loan_amount_request=20000,
            property_price=50000
        )

        # Convert the custom data to a DataFrame
        pred_df = custom_data.make_data_frame()
        print("Generated DataFrame:")
        print(pred_df)

        # Initialize the PredictPipeline
        predict_pipeline = PredictPipeline()

        # Make predictions
        predictions = predict_pipeline.predict_approval(pred_df)

        # Print the predictions
        print("Predictions:")
        print(predictions[0])

    except Exception as e:
        print(f"Error during execution: {e}")
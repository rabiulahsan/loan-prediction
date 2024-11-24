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

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e)


class CustomData:
    def __init__(  self,
        gender: str,
        age: int,
        income: int,
        income_stability: str,
        loan_amount_request: int,
        current_loan: int,
        dependents: int,
        credit_score: int,
        co_applicant: int,
        property_price: int):

        self.gender = gender
        self.age = age
        self.income = income
        self.income_stability = income_stability
        self.loan_amount_request = loan_amount_request
        self.current_loan = current_loan
        self.dependents = dependents
        self.credit_score = credit_score
        self.co_applicant = co_applicant
        self.property_price = property_price

    def make_data_frame(self):
        try:
            custom_data_input_dict = {
            "gender": [self.gender],
            "age": [self.age],
            "income": [self.income],
            "income_stability": [self.income_stability],
            "loan_amount_request": [self.loan_amount_request],
            "current_loan": [self.current_loan],
            "dependents": [self.dependents],
            "credit_score": [self.credit_score],
            "co_applicant": [self.co_applicant],
            "property_price": [self.property_price],
        }


            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e)
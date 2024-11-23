import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_file_path = 'artifacts\preprocessor.pkl'

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformer_pipeline(self):
        try:
            # Define numerical columns
            num_cols = ['Income', 'Loan Amount Request', 'Current Loan', 'Credit Score', 'Property Price']

            # Define categorical columns
            cat_cols = ['Income Stability', 'Age', 'Approved']

            # Numerical pipeline (filling missing values, replacing -999/? to 0, scaling)
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Replace missing with mean
                ('replace_negatives', FunctionTransformer(
                    lambda x: np.where(np.isin(x, [-999, '?']), 0, x), validate=False)),  # Replace -999/? with 0
                ('scaler', MinMaxScaler())  # Scale values between 0 and 1
            ])

            # Categorical pipeline (filling missing values, encoding)
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing with mode
                ('label_encoder', FunctionTransformer(
                    lambda x: pd.get_dummies(x, drop_first=True), validate=False))  # Label encoding
            ])

            # Combine pipelines for preprocessing
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e)

        
    def initiate_data_transformation(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e)
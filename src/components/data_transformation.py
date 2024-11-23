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


    def add_new_features(self, df):
        """
        Add new features to the dataset, including the 'Approved' column and transforming 'Age'.
        """
        try:
            # Create 'Approved' column: 1 if 'Loan Amount' > 0, else 0
            df['Approved'] = df['Loan Amount'].apply(lambda x: 1 if x > 0 else 0)

            # Transform 'Age' column: Categorize age ranges
            def age_category(age):
                if age <= 25:
                    return "Young"
                elif 26 <= age <= 50:
                    return "Middle Aged"
                else:
                    return "Senior"

            df['Age'] = df['Age'].apply(age_category)

            return df
        except Exception as e:
            raise CustomException(e)

        
    def initiate_data_transformation(self):
        try:
            logging.info("Loading training and testing datasets...")

            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Adding new features to datasets...")

            # Add new features (Approved, Age transformations)
            train_df = self.add_new_features(train_df)
            test_df = self.add_new_features(test_df)

            logging.info("Applying preprocessing pipeline...")

            # Create preprocessing pipeline
            preprocessor = self.data_transformer_pipeline()

            # Define target column
            target_column = "Loan Amount"

            # Split features and target
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            # Apply preprocessing
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Saving the preprocessor object...")

            # Save the preprocessor to a pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor
            )

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomException(e)
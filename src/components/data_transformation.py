import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, FunctionTransformer
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
            cat_cols = ['Income Stability', 'Age','Co-Applicant']

            # Numerical pipeline (filling missing values, replacing -999/? to 0, scaling)
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Replace missing values with mean
                ('scaler', MinMaxScaler())  # Scale values between 0 and 1
            ])

            # Categorical pipeline (filling missing values, encoding)
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing with mode
                ('ordinal_encoder', OrdinalEncoder())  # Label encoding
            ])

            # Combine pipelines for preprocessing
            preprocessor = ColumnTransformer(transformers=[
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols),
                ], remainder = 'passthrough'
            )

            # Build complete pipeline
            pre_processing_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor)  # Apply preprocessing steps
            ])
            logging.info("Data transformation has been completed...")

            return pre_processing_pipeline

        except Exception as e:
            raise CustomException(e)


    def add_new_features(self, df):
        """
        Add new features to the dataset, including the 'Approved' column and transforming 'Age'.
        """
        try:

            # Handle missing values in 'Age'
            age_imputer = SimpleImputer(strategy='mean')
            df['Age'] = age_imputer.fit_transform(df[['Age']])

            # Drop unnecessary columns
            drop_col = ['Gender', 'Dependents']
            df.drop(columns=drop_col, inplace=True)


            # Transform 'Age' into categories
            def age_category(age):
                if age <= 25:
                    return "Young"
                elif 26 <= age <= 50:
                    return "Middle Aged"
                else:
                    return "Senior"

            def approved_loan(amount):
                return 1 if amount > 0 else 0

            df['Age'] = df['Age'].apply(age_category)

            # Handle 'Loan Amount' column
            df['Loan Amount'] = df['Loan Amount'].replace(['-999', '?'], 0).astype(float)

            loan_imputer = SimpleImputer(strategy='mean')
            df['Loan Amount'] = loan_imputer.fit_transform(df[['Loan Amount']])

            # Create 'Approved' column
            df['Approved'] = df['Loan Amount'].apply(approved_loan)

            # Debugging: Check the final dataframe
            # print("Final Data After Adding Features:\n", df.head())

            # Numerical columns to preprocess
            num_cols = ['Income', 'Loan Amount Request', 'Current Loan', 'Credit Score', 'Property Price']

            # Replace -999 and '?' with 0 in numerical columns
            for col in num_cols:
                df[col] = df[col].replace(['-999', '?'], 0).astype(float)

            # print("adding feature complete sucessfully")
            return df
        except Exception as e:
            raise CustomException(e)

        
    def initiate_data_transformation(self, raw_data_path):
        try:
            logging.info("Loading dataset...")
            data = pd.read_csv(raw_data_path)

            # Debugging: Check raw data
            # print("Raw Data Head:\n", data.head())

            # Define target columns
            target_regression_column = "Loan Amount"
            target_classification_column = "Approved"

            logging.info("Adding new features...")
            data = self.add_new_features(data)

            # Debugging: Check data after adding features
            print(data.shape)

            # Create preprocessing pipeline
            preprocessor = self.data_transformer_pipeline()

            # Split features and targets
            X = data.drop(columns=[target_regression_column, target_classification_column], axis=1)
            y_regression = data[target_regression_column]
            y_classification = data[target_classification_column]

            print(X.head(5))

            feature_columns = X.columns
            X_transformed = preprocessor.fit_transform(X)

            # Define numerical columns
            num_cols = ['Income', 'Loan Amount Request', 'Current Loan', 'Credit Score', 'Property Price']

            # Define categorical columns
            cat_cols = ['Income Stability', 'Age','Co-Applicant']

            # Combine the transformed data into a DataFrame
            transformed_columns = num_cols + cat_cols  # Combine the column order
            X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)

            # Rearrange columns to match the original order
            X_transformed_df = X_transformed_df[X.columns]  # Ensure same order as original

            print(X_transformed_df.head(5))

            # Add targets back to the processed DataFrame
            X_transformed_df['Loan Amount'] = y_regression.values
            X_transformed_df['Approved'] = y_classification.values

            print(X_transformed.shape)

            logging.info("Saving preprocessor object...")
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor
            )

            return X_transformed, self.data_transformation_config.preprocessor_file_path
        except Exception as e:
            raise CustomException(e)



if __name__ =='__main__':
    raw_data_path = 'notebook\data\loan-prediction-dataset.csv'
    try:
        # Initialize the DataTransformation class
        data_transformation = DataTransformation()

        # Call the initiate_data_transformation method and get the results
        processed_data, preprocessor_file_path = data_transformation.initiate_data_transformation(raw_data_path)

        # Print the processed data and the path to the saved pipeline
        # print("\nProcessed Data Head:")
        # print(processed_data.head(4))  # Print the first few rows of the processed data

        # print("\nSaved Preprocessor Path:")
        # print(preprocessor_file_path)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
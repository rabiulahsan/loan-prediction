from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

class ModelConfig:
    trained_model_path:str = 'artifacts\model.pkl'

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelConfig()

    def initiate_model_trainer(self, data):
        try:
            # Define target columns
            target_regression_column = "Loan Amount"
            target_classification_column = "Approved"

            # Split features and targets
            X_classification = data.drop(columns=[target_classification_column], axis=1)
            X_regression = data.drop(columns=[target_regression_column], axis=1)
            y_regression = data[target_regression_column]
            y_classification = data[target_classification_column]

            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

            # Step 2: Train a classification model
            classification_models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
                "RandomForestClassifier": RandomForestClassifier(random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            }

        except Exception as e:
            raise CustomException(e)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split


from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_classification_models, evaluate_regression_models

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
            X_classification = data.drop(columns=[target_classification_column, target_regression_column], axis=1)
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

            # Step 4: Train regression models
            regression_models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
                "RandomForestRegressor": RandomForestRegressor(random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                "XGBRegressor": XGBRegressor(random_state=42, verbosity=0)
            }

            classification_model_report = evaluate_classification_models(X_train_clf, X_test_clf, y_train_clf, y_test_clf, classification_models)
            # Find the best model based on the test score (second element in the tuple)
            best_clf_model_name, (best_clf_train_score, best_clf_test_score) = max(
                classification_model_report.items(), key=lambda x: x[1][1]  # Use test score for max
            )

            best_clf_model = classification_models[best_clf_model_name]

            # Assign best_test_score as best_model_score for comparison
            best_clf_model_score = best_clf_test_score

            # Check if the best test score is above the threshold
            if best_clf_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            # Return the report, best model name, and R2 score for the best model


            evaluate_regression_models( X_train_reg, X_test_reg, y_train_reg, y_test_reg, regression_models)
            return classification_model_report, best_clf_model_name, best_clf_model_score

        except Exception as e:
            raise CustomException(e)





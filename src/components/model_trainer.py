import os
import sys
from dataclasses import dataclass

import yaml
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

# common utility functions
from src.utils import save_object, evaluate_models
import pandas as pd


@dataclass
class ModelTrainerConfig:
    """
    Model trainer config with path of the trained model file
    """

    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    hyper_paramter_file_path = os.path.join("src/components/config", "hyperparameters.yml")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            logging.info("Read hyperparameter search space from yml file")
            #read params from yml file
            with open(self.model_trainer_config.hyper_paramter_file_path, 'r') as stream:
                params = yaml.safe_load(stream)
            
            logging.info(f"Hyperparameter search space: {params}")
    
            model_report, models= evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )
            
            logging.info(f"Model report: {model_report}")
            
            #create dataframe from list of dict
            model_report_df = pd.DataFrame(model_report)

            #get index of the max by column
            best_model_name = model_report_df.T['r2_score'].idxmax()
            
            #get max value of model r2_Score
            best_model_score = model_report_df.T['r2_score'].max()
            
            #get best model
            best_model = models[best_model_name]['model']
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model: {best_model_name} with score: {best_model_score} and hyperparameters: {models[best_model_name]['best_hyperparameters']} found and saved")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return best_model_score
        
        except Exception as e:
            logging.error(f"Error in ModelTrainer: {e}")
            raise CustomException(e, sys)

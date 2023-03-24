import dill
from src.exception import CustomException
from src.logger import logging
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
 

def save_object(file_path, obj):
    """
    This function is responsible for saving python object to a file in pickle format
    """
    try:
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        logging.error(f"Error occurred while saving object to file: {e}")
        raise CustomException(e, sys)


def load_object(file_path):
    """
    This function is responsible for loading python object from a file in pickle format
    """
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        logging.error(f"Error occurred while loading object from file: {e}")
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    '''
    This function is responsible for evaluating the models and returning the best model with best hyperparameters 
    '''
    model_evaluations = {}
    best_models = {}
    
    for model_name, model in models.items():
        try:
            logging.info(f"Evaluating model: {model_name}")
            
            parameter_grid = params[model_name] if model_name in params else {}
            # Evaluate the model on each of the hyperparameter values in the grid
            grid = GridSearchCV(model, parameter_grid, cv=5, n_jobs=-1, verbose=1)
            # Train the model on the best hyperparameter values
            grid.fit(X_train, y_train)
            # Select the best hyperparameter values based on the results of the evaluation
            best_hyperparameters = grid.best_params_
            # create a new model instance with the best hyperparameter values
            model =  grid.best_estimator_
            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            
            model_evaluations[model_name] = {
                "r2_score": r2_score(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
            }
            
            best_models[model_name] = {"model": model, "best_hyperparameters": best_hyperparameters}
            
        except Exception as e:
            logging.error(f"Error occurred while evaluating model {model_name}: {e}")
            raise CustomException(e, sys)
        
    return model_evaluations, best_models
    
    
    
    
    
    
    
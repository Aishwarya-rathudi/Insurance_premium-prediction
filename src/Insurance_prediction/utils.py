import os
import sys
from src.Insurance_prediction.exception import CustomException
from src.Insurance_prediction.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import pickle
import numpy as np


load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")


def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info(f"Connection Established: {mydb}")

        df=pd.read_sql_query('Select * from insurance',mydb)
        print(df.head())

        return df


    except Exception as ex:
        raise CustomException(ex)
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
        
        report = {}

        for model_name, model in models.items():  # Iterate through models (dictionary)

            try:
                 
                if model_name not in params:  # Handle missing configuration (optional)
                   continue

                para = params[model_name]


                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_model_score

    
            except Exception as e:
               raise CustomException(e, sys)
            
        return report
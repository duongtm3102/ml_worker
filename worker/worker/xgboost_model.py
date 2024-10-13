import logging
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

from worker.db.db import get_session
from worker.db.model import HousePredXGBoost
import utils

FEATURES = ['minute', 'hour', 'day', 'lag1', 'lag2']
TARGET = 'avg'

def create_features(df):
    df = df.copy()
    df['minute'] = df.index.minute
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['lag1'] = df['avg'].shift(1)
    df['lag2'] = df['avg'].shift(2)
    return df

def build_model(house_id, slice_gap):
    train_data = utils.get_house_train_data(house_id=house_id, slice_gap=slice_gap)
    
    # Preprocess
    
    train_data.index = train_data['reg_date']
    train_data.drop(['slice_index', 'reg_date'], axis=1, inplace=True)
    
    train = train_data.copy()
    train = create_features(train)
    X_train = train[FEATURES]
    y_train = train[TARGET]
    
    reg = xgb.XGBRegressor(n_estimators=1000, 
                           early_stopping_rounds=50,
                           learning_rate=0.01,
                           objective='reg:absoluteerror')

    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train),],
            verbose=100)
    
    model_dump = pickle.dumps(reg)
    utils.save_model(house_id=house_id, slice_gap=slice_gap, model_name="XGBoost", model_dump=model_dump)
    
    return model_dump
    
def forecast_house_data(house_id, slice_gap):
    model_dump = utils.get_model(house_id, "XGBoost", slice_gap)
    if not model_dump:
        model_dump = build_model(house_id, slice_gap)
        
    reg = pickle.loads(model_dump)
    
    recent_data = utils.get_recent_house_data(house_id=house_id, slice_gap=slice_gap)
    recent_data.index = recent_data['reg_date']
    recent_data.drop(['slice_index', 'reg_date'], axis=1, inplace=True)
    
    
    df_to_predict = recent_data
    next_index = df_to_predict.index[-1] + pd.Timedelta(minutes=5)
    df_to_predict = pd.concat([df_to_predict, pd.DataFrame(index=[next_index])])
    print(df_to_predict)
    df_to_predict = create_features(df_to_predict)
    forecast = reg.predict(df_to_predict[FEATURES].iloc[-1].to_frame().T)
    
    avg_forecast = forecast[0]
    return avg_forecast
    with get_session() as db:
        house_pred = HousePredXGBoost(
                        house_id=house_id,
                        reg_date=None,
                        slice_gap=slice_gap,
                        slice_index=None,
                        avg=avg_forecast
        )
        db.merge(house_pred)
        db.commit()
    
    

print(forecast_house_data(0, 5))

import os
import logging
import pandas as pd
import pickle
import xgboost as xgb

from dotenv import load_dotenv

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

dotenv_path = os.path.join(base_dir, '.env')
print(dotenv_path)
load_dotenv(dotenv_path)

from db.db import get_session, engine
from db.model import HousePredXGBoost
import utils
import constants as const
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, wait
import time


log_file_handler = logging.FileHandler("xgboost.log")
log_file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

error_file_handler = logging.FileHandler("xgboost.log.error")
error_file_handler.setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[log_file_handler, stream_handler, error_file_handler]
)
logger = logging.getLogger(__name__)

FEATURES = ['minute', 'hour', 'day', 'lag2', 'lag3']
TARGET = 'avg'

def create_features(df):
    df = df.copy()
    df['minute'] = df.index.minute
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['lag2'] = df['avg'].shift(1)
    df['lag3'] = df['avg'].shift(2)
    return df

def build_model(house_id, slice_gap):
    train_data = utils.get_house_train_data(house_id=house_id, slice_gap=slice_gap)
    updated_at = train_data.iloc[-1]['start_time']
    # Preprocess
    train_data.index = train_data['start_time']
    train_data.drop(['slice_index', 'start_time', 'year', 'month', 'day'], axis=1, inplace=True)
    
    train = train_data.copy()
    train = create_features(train)
    X_train = train[FEATURES]
    y_train = train[TARGET]
    
    start_time = time.time()
    
    reg = xgb.XGBRegressor(n_estimators=1000, 
                           early_stopping_rounds=50,
                           learning_rate=0.01,
                           objective='reg:absoluteerror')

    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train),],
            verbose=100)
    model_dump = pickle.dumps(reg, protocol=pickle.HIGHEST_PROTOCOL)
    utils.save_model(house_id=house_id, slice_gap=slice_gap, model_name="XGBoost", model_dump=model_dump, updated_at=updated_at)
    
    logger.info(
        f'---XGBoost model for house: {house_id} trained and saved. Time taken: {time.time() - start_time}---'
    )
    
    return model_dump
    
def forecast_house_data(house_id, slice_gap):
    try:
        if utils.check_skip_forecast(house_id, slice_gap, "XGBoost"):
            logger.info(f"--- No new house data, skip forecast for house {house_id} using XGBoost Model.")
            return
        
        model = utils.get_model(house_id, "XGBoost", slice_gap)
        model_dump = None
        if model:
            model_dump = model.model
        else:
            model_dump = build_model(house_id, slice_gap)
            
        reg = pickle.loads(model_dump)
        
        recent_data = utils.get_recent_house_data(house_id=house_id, slice_gap=slice_gap, lags=3)
        recent_data.index = recent_data['start_time']
        recent_data.drop(['slice_index', 'start_time', 'year', 'month', 'day'], axis=1, inplace=True)
        
        
        df_to_predict = recent_data
        next_index = df_to_predict.index[-1] + pd.Timedelta(minutes=2*slice_gap)
        df_to_predict = pd.concat([df_to_predict, pd.DataFrame(index=[next_index])])
        df_to_predict = create_features(df_to_predict)
        
        start_time = time.time()
        
        forecast = reg.predict(df_to_predict[FEATURES].iloc[-1].to_frame().T)
        
        avg_forecast = max(forecast[0], 0)
        year, month, day, slice_index = utils.datetime_to_slice_index(next_index, slice_gap=slice_gap)
        
        with get_session() as db:
            house_pred = HousePredXGBoost(
                            house_id=house_id,
                            reg_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            slice_gap=slice_gap,
                            year=year,
                            month=month,
                            day=day,
                            slice_index=slice_index,
                            avg=avg_forecast
                        )
            db.merge(house_pred)
            # db.commit()
            logger.info(
                f'---House data forecasted and saved to DB for house: {house_id}. Time taken: {time.time() - start_time}---'
            )
    
    except Exception as e:
        logger.error(f"Error forecasting for house {house_id} using XGBoost: {e}")

def retrain_model(house_id, slice_gap):
    try:
        last_data = utils.get_recent_house_data(house_id=house_id, slice_gap=slice_gap, lags=1)
        if last_data.empty:
            logger.info(f"Can not get last data of house {house_id}, skipping retrain model")
            return
        
        last_data_datetime = last_data.iloc[-1]['start_time']
    
        model = utils.get_model(house_id=house_id, model_name="XGBoost", slice_gap=slice_gap)
        if not model:
            logger.info(f"XGBoost model for house {house_id} is not exist, skip retrain model")
            return
        model_updated_at = model.updated_at
        
        if last_data_datetime - model_updated_at >= timedelta(hours=1):
            build_model(house_id=house_id, slice_gap=slice_gap)
        else:
            logger.info(f"XGBoost model for house {house_id} is up to date, skipping retrain.")
        
        
    except Exception as e:
        logger.error(f"Error while retraining XGBoost model for house {house_id}: {e}")

def initialize_engine():
    engine.dispose(close=False)


def run_forecast():
    logger.info('--- Forecast Houses Data Process started ---')

    with ProcessPoolExecutor(max_workers=4, initializer=initialize_engine) as executor:
        futures = [executor.submit(forecast_house_data, house_id, const.SLICE_GAP) for house_id in const.HOUSE_IDS_TO_FORECAST]
        wait(futures)

    logger.info('--- Forecast House Data Process ended ---')

def retrain_models():
    logger.info('--- Retrain Models Process started ---')

    with ProcessPoolExecutor(max_workers=4, initializer=initialize_engine) as executor:
        futures = [executor.submit(retrain_model, house_id, const.SLICE_GAP) for house_id in const.HOUSE_IDS_TO_FORECAST]
        wait(futures)

    logger.info('--- Retrain Models Process ended ---')
    

def main():
    run_forecast()
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_forecast, 'interval', minutes=const.FORECAST_INTERVAL, coalesce=True, max_instances=1)
    scheduler.add_job(retrain_models, 'interval', minutes=const.RETRAIN_INTERVAL, coalesce=True, max_instances=1)

    scheduler.start()
    logger.info('<=========== Scheduler Started ===========>')

    while True:
        time.sleep(2)

    
if __name__ == '__main__':
    main()

import os
import logging
import pandas as pd
import pickle
import pmdarima
from dotenv import load_dotenv

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

dotenv_path = os.path.join(base_dir, '.env')
load_dotenv(dotenv_path)
from db.db import get_session, engine
from db.model import HousePredArima
import utils
import constants as const
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, wait
import time


file_handler = logging.FileHandler("arima.log")
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)

logger = logging.getLogger(__name__)


def build_model(house_id, slice_gap):
    train_data = utils.get_house_train_data(house_id=house_id, slice_gap=slice_gap)
    updated_at = train_data.iloc[-1]['start_time']
    # Preprocess
    train_data.index = train_data['start_time']
    train_data.drop(['slice_index', 'start_time', 'year', 'month', 'day'], axis=1, inplace=True)
    
    start_time = time.time()
    
    auto_arima = pmdarima.auto_arima(train_data, stepwise=True, seasonal=False)
   
    model_dump = pickle.dumps(auto_arima, protocol=pickle.HIGHEST_PROTOCOL)
    model = utils.save_model(house_id=house_id, slice_gap=slice_gap, model_name="ARIMA", model_dump=model_dump, updated_at=updated_at)
    
    logger.info(
        f'---ARIMA model for house: {house_id} trained and saved. Time taken: {time.time() - start_time}---'
    )
    
    return model

def update_observed_data_to_model(house_id, slice_gap, model):
    try:
        model_updated_at = model.updated_at
        df_new_data = utils.get_house_train_data(house_id, slice_gap)
        df_new_data = df_new_data[df_new_data['start_time'] > model_updated_at]
        if df_new_data.empty:
            logger.debug(f"Observed data of ARIMA Model of house {house_id} is up to date.")
            return model
    
        updated_at = df_new_data.iloc[-1]['start_time']
        df_new_data.index = df_new_data['start_time']
        df_new_data.drop(['slice_index', 'start_time', 'year', 'month', 'day'], axis=1, inplace=True)
        
        arima_model = pickle.loads(model.model)
        
        for i in range(df_new_data.shape[0]):
            # print(df_new_data.iloc[i])
            arima_model.update(df_new_data.iloc[i])
            # logger.debug(f"Update value {df_new_data.iloc[i]} to ARIMA Model of house {house_id}")
        
        model_dump = pickle.dumps(arima_model, protocol=pickle.HIGHEST_PROTOCOL)
        model = utils.save_model(house_id, slice_gap, "ARIMA", model_dump, updated_at)
        
        return model
    except Exception as e:
        logger.error(f"Error while updating observed data to ARIMA Model of house {house_id}: {e}")

def forecast_house_data(house_id, slice_gap):
    try:
        if utils.check_skip_forecast(house_id, slice_gap, "ARIMA"):
            logger.info(f"--- No new house data, skip forecast for house {house_id} using ARIMA Model.")
            return
        
        model = utils.get_model(house_id, "ARIMA", slice_gap)
        model_dump = None
        if model:
            model_dump = update_observed_data_to_model(house_id=house_id, slice_gap=slice_gap, model=model)
        else:
            model_dump = build_model(house_id, slice_gap)
            
        arima = pickle.loads(model_dump.model)
        next_index = model_dump.updated_at + timedelta(minutes=2*slice_gap)
        
        start_time = time.time()
        
        forecast = arima.predict(n_periods=2)
        avg_forecast_0 = max(forecast[0], 0)
        avg_forecast = max(forecast[1], 0)
        # print(f"ARIMA FORECAST {avg_forecast}")
        
        year, month, day, slice_index = utils.datetime_to_slice_index(next_index, slice_gap=slice_gap)
        
        with get_session() as db:
            house_pred = HousePredArima(
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
            db.commit()
            logger.info(
                f'---House data forecasted and saved to DB for house: {house_id}. Time taken: {time.time() - start_time}---'
            )
    
    except Exception as e:
        logger.error(f"Error forecasting for house {house_id} using ARIMA: {e}")

def retrain_model(house_id, slice_gap):
    try:
        last_data = utils.get_recent_house_data(house_id=house_id, slice_gap=slice_gap, lags=1)
        if last_data.empty:
            logger.info(f"Can not get last data of house {house_id}, skipping retrain model")
            return
        
        last_data_datetime = last_data.iloc[-1]['start_time']
    
        model = utils.get_model(house_id=house_id, model_name="ARIMA", slice_gap=slice_gap)
        if not model:
            logger.info(f"ARIMA model for house {house_id} is not exist, skip retrain model")
            return
        model_updated_at = model.updated_at
        
        if last_data_datetime - model_updated_at >= timedelta(hours=1):
            build_model(house_id=house_id, slice_gap=slice_gap)
        else:
            logger.info(f"ARIMA model for house {house_id} is up to date, skipping retrain.")
        
        
    except Exception as e:
        logger.error(f"Error while retraining ARIMA model for house {house_id}: {e}")

def initialize_engine():
    engine.dispose(close=False)


def run_forecast():
    logger.info('--- Forecast Houses Data Process started ---')

    with ProcessPoolExecutor(max_workers=4, initializer=initialize_engine) as executor:
        futures = [executor.submit(forecast_house_data, house_id, 5) for house_id in const.HOUSE_IDS_TO_FORECAST]
        wait(futures)

    logger.info('--- Forecast House Data Process ended ---')

def retrain_models():
    logger.info('--- Retrain Models Process started ---')

    with ProcessPoolExecutor(max_workers=4, initializer=initialize_engine) as executor:
        futures = [executor.submit(retrain_model, house_id, 5) for house_id in const.HOUSE_IDS_TO_FORECAST]
        wait(futures)

    logger.info('--- Retrain Models Process ended ---')
    

def main():
    run_forecast()
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_forecast, 'interval', minutes=1, coalesce=True, max_instances=1)
    # scheduler.add_job(retrain_models, 'interval', minutes=30, coalesce=True, max_instances=1)

    scheduler.start()
    logger.info('<=========== Scheduler Started ===========>')

    while True:
        time.sleep(2)

    
if __name__ == '__main__':
    main()

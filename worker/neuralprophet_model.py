import logging
import numpy as np
import pandas as pd
import pickle

from neuralprophet import NeuralProphet

from db.db import get_session, engine
from db.model import HousePredNeuralProphet
import utils
import constants as const
# from dotenv import load_dotenv
# load_dotenv()
from sqlalchemy.orm import Session
import json
import multiprocessing
# from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait
import time



logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
    handlers=[
        logging.FileHandler("neuralprophet.log"),  # Log to a file
        logging.StreamHandler()          # Log to the console
    ]
)

logger = logging.getLogger(__name__)

## max_allowed_packet=104857600 to [mysqld] in /etc/my.cnf
##
def build_model(house_id, slice_gap):
    train_data = utils.get_house_train_data(house_id=house_id, slice_gap=slice_gap)
    
    # Preprocess
    train_data.drop(['slice_index', 'year', 'month', 'day'], axis=1, inplace=True)
    
    df_train = train_data.copy()
    df_train = df_train.rename(columns={'start_time':'ds',
                                  'avg':'y'})
    
    start_time = time.time()
    n_lags = const.NEURAL_PROPHET_N_LAGS
    m = NeuralProphet(n_lags=n_lags)
    metrics = m.fit(df=df_train, freq=const.SLICE_GAP_TO_MIN[5])
    model_dump = pickle.dumps(m)
    utils.save_model(house_id=house_id, slice_gap=slice_gap, model_name="NeuralProphet", model_dump=model_dump)
    
    logger.info(
        f'---NeuralProphet model for house: {house_id} trained and saved. Time taken: {time.time() - start_time}---'
    )
    
    return model_dump
    
def forecast_house_data(house_id, slice_gap):
    try:
        model_dump = utils.get_model(house_id, "NeuralProphet", slice_gap)
        if not model_dump:
            model_dump = build_model(house_id, slice_gap)
            
        m = pickle.loads(model_dump)
        
        recent_data = utils.get_recent_house_data(house_id=house_id, slice_gap=slice_gap, lags=3)
        
        recent_data.drop(['slice_index', 'year', 'month', 'day'], axis=1, inplace=True)
        
        df_train = recent_data.copy()
        df_train = df_train.rename(columns={'start_time':'ds',
                                    'avg':'y'})
        
        future = m.make_future_dataframe(df_train)
        forecast = m.predict(future)
        avg_forecast = max(forecast.iloc[-1]['yhat1'], 0)
        print(f'{avg_forecast}')
        return avg_forecast
        
        print(f"{next_index} - {avg_forecast}")
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
            db.commit()
            logger.info(
                f'---House data forecasted and saved to DB for house: {house_id}. Time taken: {time.time() - start_time}---'
            )
    
    except Exception as e:
        logger.error(f"Error forecasting for house {house_id} using NeuralProphet: {e}")

def initialize_engine():
    engine.dispose(close=False)


def run_forecast():
    # logger.info('[x] Forecasting process started')

    with ProcessPoolExecutor(max_workers=4, initializer=initialize_engine) as executor:
        futures = [executor.submit(forecast_house_data, house_id, 5) for house_id in const.HOUSE_IDS_TO_FORECAST]
        wait(futures)

    # logger.info('Forecasting completed')


def main():
    
    forecast_house_data(20, 5)
    return
    run_forecast()
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_forecast, 'interval', minutes=1, coalesce=True, max_instances=1)
    scheduler.add_job(retrain_all_models, 'interval', minutes=15, coalesce=True, max_instances=1)

    scheduler.start()
    logger.info('<=========== Scheduler Started ===========>')

    while True:
        time.sleep(2)

    
    
if __name__ == '__main__':
    main()
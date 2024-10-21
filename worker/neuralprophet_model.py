import os
import logging
import pandas as pd
import pickle
from neuralprophet import NeuralProphet

from dotenv import load_dotenv

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

dotenv_path = os.path.join(base_dir, '.env')
print(dotenv_path)
load_dotenv(dotenv_path)

from db.db import get_session, engine
from db.model import HousePredNeuralProphet
import utils
import constants as const
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, wait
import time

file_handler = logging.FileHandler("neuralprophet.log")
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)

logger = logging.getLogger(__name__)

## max_allowed_packet=104857600 to [mysqld] in /etc/my.cnf
##
def build_model(house_id, slice_gap):
    train_data = utils.get_house_train_data(house_id=house_id, slice_gap=slice_gap)
    updated_at = train_data.iloc[-1]['start_time']
    # Preprocess
    train_data.drop(['slice_index', 'year', 'month', 'day'], axis=1, inplace=True)
    
    df_train = train_data.copy()
    df_train = df_train.rename(columns={'start_time':'ds',
                                  'avg':'y'})
    
    start_time = time.time()
    n_lags = const.NEURAL_PROPHET_N_LAGS
    m = NeuralProphet(n_lags=n_lags)
    metrics = m.fit(df=df_train, freq=const.SLICE_GAP_TO_MIN[5])
    with open(f'neuralprophet_{house_id}.pkl', 'wb') as handle:
        pickle.dump(m, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # model_dump = pickle.dumps(m)
    utils.save_model(house_id=house_id, slice_gap=slice_gap, model_name="NeuralProphet", model_dump=0, updated_at=updated_at)
    
    logger.info(
        f'---NeuralProphet model for house: {house_id} trained and saved. Time taken: {time.time() - start_time}---'
    )
    
    return m
    
def forecast_house_data(house_id, slice_gap):
    try:
        if utils.check_skip_forecast(house_id, slice_gap, "NeuralProphet"):
            logger.info(f"--- No new house data, skip forecast for house {house_id} using NeuralProphet Model.")
            return
        
        model = utils.get_model(house_id, "NeuralProphet", slice_gap)
        m = None
        if model:
            model_save_path = f"neuralprophet_{house_id}.pkl"
            if os.path.exists(model_save_path):
                with open(model_save_path, 'rb') as handle:
                    m = pickle.load(handle)
                    
            else:
                logger.warning(f"NeuralProphet model save file for house {house_id} not found.")
                m = build_model(house_id, slice_gap)
                
        else:
            m = build_model(house_id, slice_gap)
        
        recent_data = utils.get_recent_house_data(house_id=house_id, slice_gap=slice_gap, lags=3)
        
        recent_data.drop(['slice_index', 'year', 'month', 'day'], axis=1, inplace=True)
        
        df_train = recent_data.copy()
        df_train = df_train.rename(columns={'start_time':'ds',
                                    'avg':'y'})
        
        future = m.make_future_dataframe(df_train)
        next_index = future.iloc[-1]['ds']
        
        start_time = time.time()
        forecast = m.predict(future)
        avg_forecast = max(forecast.iloc[-1]['yhat1'], 0)
        
        year, month, day, slice_index = utils.datetime_to_slice_index(next_index, slice_gap=slice_gap)
        with get_session() as db:
            house_pred = HousePredNeuralProphet(
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

def retrain_model(house_id, slice_gap):
    try:
        last_data = utils.get_recent_house_data(house_id=house_id, slice_gap=slice_gap, lags=1)
        if last_data.empty:
            logger.info(f"Can not get last data of house {house_id}, skipping retrain model")
            return
        
        last_data_datetime = last_data.iloc[-1]['start_time']
    
        model = utils.get_model(house_id=house_id, model_name="NeuralProphet", slice_gap=slice_gap)
        if not model:
            logger.info(f"NeuralProphet model for house {house_id} is not exist, skip retrain model")
            return
        model_updated_at = model.updated_at
        
        if last_data_datetime - model_updated_at >= timedelta(hours=1):
            build_model(house_id=house_id, slice_gap=slice_gap)
        else:
            logger.info(f"NeuralProphet model for house {house_id} is up to date, skipping retrain.")
        
        
    except Exception as e:
        logger.error(f"Error while retraining NeuralProphet model for house {house_id}: {e}")


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
    scheduler.add_job(retrain_models, 'interval', minutes=30, coalesce=True, max_instances=1)

    scheduler.start()
    logger.info('<=========== Scheduler Started ===========>')

    while True:
        time.sleep(2)

    
    
if __name__ == '__main__':
    main()
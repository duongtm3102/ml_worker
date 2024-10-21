import logging
import numpy as np
import pandas as pd
import datetime
from db.db import get_session
from db.model import PredictModel, HouseData, HousePredArima, HousePredXGBoost, HousePredNeuralProphet
import constants as const

logger = logging.getLogger(__name__)


def get_model(house_id, model_name, slice_gap):
    with get_session() as db:
        model = db.query(
                    PredictModel
                ).filter(
                    PredictModel.house_id == house_id,
                    PredictModel.model_name == model_name, 
                    PredictModel.slice_gap==slice_gap
                ).first()

        if not model:
            # LOG
            return None
        
        return model
        

def save_model(house_id, slice_gap, model_name, model_dump, updated_at):
    with get_session() as db:
        new_model = PredictModel(house_id=house_id, slice_gap=slice_gap, model_name=model_name, model=model_dump, updated_at=updated_at)
        model = db.merge(new_model)
        db.commit()
        
        logger.info(f"Model saved - House_id: {house_id} - Model: {model_name}")
        
        return model
    
    
def get_house_train_data(house_id, slice_gap):
    with get_session() as db:
        house_data = db.query(
                        HouseData.avg, HouseData.year, HouseData.month, HouseData.day, HouseData.slice_index
                    ).filter(
                        HouseData.house_id == house_id, HouseData.slice_gap == slice_gap
                    ).order_by(
                        HouseData.year, HouseData.month, HouseData.day, HouseData.slice_index
                    ).all()
    
    if not house_data:
        # LOG
        return None
    
    house_data_df = pd.DataFrame(house_data, columns=["avg", "year", "month", "day", "slice_index"])
    
    house_data_df['start_time'] = house_data_df.apply(
        lambda row: slice_index_to_datetime(row['year'], row['month'], row['day'], row['slice_index'], slice_gap),
        axis=1
    )
    
    house_data_df.sort_values(by='start_time', inplace=True)
    house_data_df.reset_index(drop=True, inplace=True)
    
    house_data_df['start_time'] = pd.to_datetime(house_data_df['start_time'])
    min_date = house_data_df['start_time'].min()
    max_date = house_data_df['start_time'].max()

    complete_time_range = pd.date_range(start=min_date, end=max_date, freq=const.SLICE_GAP_TO_MIN[slice_gap])

    time_df = pd.DataFrame(complete_time_range, columns=['start_time'])

    filled_house_data_df = pd.merge(time_df, house_data_df, on='start_time', how='left')

    # Fill missing avg values with 0
    filled_house_data_df.fillna({'avg': 0}, inplace=True)

    return filled_house_data_df
    
    
def get_recent_house_data(house_id, slice_gap, lags):
    with get_session() as db:
        # Fetch required fields: avg, slice_index, year, month, day
        house_data = db.query(
            HouseData.avg, HouseData.slice_index, HouseData.year, HouseData.month, HouseData.day
        ).filter(
            HouseData.house_id == house_id, 
            HouseData.slice_gap == slice_gap
        ).order_by(
            HouseData.year.desc(), HouseData.month.desc(), HouseData.day.desc(), HouseData.slice_index.desc()
        ).limit(lags).all()
    if not house_data:
        logger.error(f"Can not get recent data of house {house_id}.")
        return None
    
    # Convert query result to a DataFrame
    house_data_df = pd.DataFrame(house_data, columns=["avg", "slice_index", "year", "month", "day"])

    # Calculate reg_date using slice_index_to_datetime for each row
    house_data_df['start_time'] = house_data_df.apply(
        lambda row: slice_index_to_datetime(row['year'], row['month'], row['day'], row['slice_index'], slice_gap),
        axis=1
    )
    house_data_df.sort_values(by='start_time', inplace=True)
    house_data_df.reset_index(drop=True, inplace=True)
    
    house_data_df['start_time'] = pd.to_datetime(house_data_df['start_time'])
    min_date = house_data_df['start_time'].min()
    max_date = house_data_df['start_time'].max()

    complete_time_range = pd.date_range(start=min_date, end=max_date, freq=const.SLICE_GAP_TO_MIN[slice_gap])

    time_df = pd.DataFrame(complete_time_range, columns=['start_time'])

    filled_house_data_df = pd.merge(time_df, house_data_df, on='start_time', how='left')

    # Fill missing avg values with 0
    filled_house_data_df.fillna({'avg': 0}, inplace=True)
    
    return filled_house_data_df


def slice_index_to_datetime(year, month, day, slice_index, slice_gap):
    # Each slice represents 'slice_gap' minutes, so we calculate total minutes from the slice_index
    total_minutes = slice_index * slice_gap
    
    # Calculate the hours and minutes from total_minutes
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    # Create a datetime object using year, month, day, hours, and minutes
    return datetime.datetime(int(year), int(month), int(day), int(hours), int(minutes))

def datetime_to_slice_index(dt, slice_gap):
    # Extract year, month, and day from the datetime object
    year = dt.year
    month = dt.month
    day = dt.day
    
    # Calculate the number of minutes since midnight
    minutes_since_midnight = dt.hour * 60 + dt.minute
    
    # Calculate the slice_index (each slice is 5 minutes)
    slice_index = minutes_since_midnight // slice_gap
    
    return year, month, day, slice_index

def check_skip_forecast(house_id, slice_gap, model_name):
    if model_name == "ARIMA":
        houseForecast = HousePredArima
    elif model_name == "XGBoost":
        houseForecast = HousePredXGBoost
    elif model_name == "NeuralProphet":
        houseForecast = HousePredNeuralProphet
    else:
        logger.error("Error while checking skip forecast: Invalid Model Name")
        return False
    with get_session() as db:
        # Fetch required fields: avg, slice_index, year, month, day
        house_data = db.query(
            HouseData.slice_index, HouseData.year, HouseData.month, HouseData.day
        ).filter(
            HouseData.house_id == house_id, 
            HouseData.slice_gap == slice_gap
        ).order_by(
            HouseData.year.desc(), HouseData.month.desc(), HouseData.day.desc(), HouseData.slice_index.desc()
        ).limit(1).first()
        
        house_forecast = db.query(
            houseForecast.slice_index, houseForecast.year, houseForecast.month, houseForecast.day
        ).filter(
            houseForecast.house_id == house_id, 
            houseForecast.slice_gap == slice_gap
        ).order_by(
            houseForecast.year.desc(), houseForecast.month.desc(), houseForecast.day.desc(), houseForecast.slice_index.desc()
        ).limit(1).first()

    if not house_data or not house_forecast:
        return False
    
    last_data_datetime = slice_index_to_datetime(house_data.year, house_data.month, house_data.day, house_data.slice_index,
                                                 slice_gap)
    last_forecast_datetime = slice_index_to_datetime(house_forecast.year, house_forecast.month, house_forecast.day, house_forecast.slice_index,
                                                 slice_gap)
    
    return last_forecast_datetime - last_data_datetime == datetime.timedelta(minutes=slice_gap)

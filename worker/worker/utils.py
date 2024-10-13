import datetime
import logging
import numpy as np
import pandas as pd
from worker.db.db import get_session
from worker.db.model import PredictModel, HouseData


logger = logging.getLogger(__name__)


def get_model(house_id, model_name, slice_gap):
    with get_session() as db:
        model = db.query(PredictModel.model).filter(
                                                PredictModel.house_id == house_id, 
                                                PredictModel.model_name == model_name, 
                                                PredictModel.slice_gap==slice_gap
                                                ).first()
        
        if not model:
            # LOG
            return None
        
        return model.model
        

def save_model(house_id, slice_gap, model_name, model_dump):
    with get_session() as db:
        new_model = PredictModel(house_id=house_id, slice_gap=slice_gap, model_name=model_name, model=model_dump, updated_at=datetime.datetime.now(datetime.timezone.utc))
        model = db.merge(new_model)
        db.commit()
        
        logger.info(f"Model saved - House_id: {house_id} - Model: {model_name}")
        
        return model
    
    
def get_house_train_data(house_id, slice_gap):
    with get_session() as db:
        house_data = db.query(HouseData.avg, HouseData.slice_index, HouseData.reg_date).filter(
                        HouseData.house_id == house_id, HouseData.slice_gap == slice_gap).order_by(
                        HouseData.reg_date).all()
    
    if not house_data:
        # LOG
        return None
    
    house_data_df = pd.DataFrame(house_data, columns=["avg", "slice_index", "reg_date"])
    # house_data_df.sort_values(by='reg_date', inplace=True)
    house_data_df.reset_index(drop=True, inplace=True)
    
    house_data_df['reg_date'] = pd.to_datetime(house_data_df['reg_date'])
    min_date = house_data_df['reg_date'].min()
    max_date = house_data_df['reg_date'].max()

    complete_time_range = pd.date_range(start=min_date, end=max_date, freq='5min')

    time_df = pd.DataFrame(complete_time_range, columns=['reg_date'])

    filled_house_data_df = pd.merge(time_df, house_data_df, on='reg_date', how='left')

    # Fill missing avg values with 0
    filled_house_data_df.fillna({'avg': 0}, inplace=True)

    # TODO: Slice index
    
    return filled_house_data_df
    first_slice = min(house.slice_index for house in house_data)
    last_slice = max(house.slice_index for house in house_data) - 1
    
    # LOG logger.info(f'house_id: {house_id}, max_slice: {max_slice}, min_slice: {min_slice}, count: {len(house_data)}')
    
    slice_to_avg = {house.slice_index: house.avg for house in house_data}
    
    data = [slice_to_avg.get(s, 0) for s in range(first_slice, last_slice + 1)]
    
    if not data:
        #LOG
        
        return None
        
    return np.array(data)
    
    
def get_recent_house_data(house_id, slice_gap):
    with get_session() as db:
        house_data = db.query(HouseData.avg, HouseData.slice_index, HouseData.reg_date).filter(
            HouseData.house_id == house_id, 
            HouseData.slice_gap == slice_gap
        ).order_by(HouseData.reg_date.desc()).limit(3).all()

    house_data_df = pd.DataFrame(house_data, columns=["avg", "slice_index", "reg_date"])
    house_data_df['reg_date'] = pd.to_datetime(house_data_df['reg_date'])
    return house_data_df

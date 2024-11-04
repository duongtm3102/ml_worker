import sys
import datetime
from db.db import get_session
from db.model import HouseData, HousePredArima, HousePredXGBoost, HousePredNeuralProphet
from worker import utils, constants

def check_delayed_forecast(house_id, slice_gap, model_name):
    if model_name == "ARIMA":
        houseForecast = HousePredArima
    elif model_name == "XGBoost":
        houseForecast = HousePredXGBoost
    elif model_name == "NeuralProphet":
        houseForecast = HousePredNeuralProphet
    else:
        print("Invalid Model Name.")
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
    
    last_data_datetime = utils.slice_index_to_datetime(house_data.year, house_data.month, house_data.day, house_data.slice_index,
                                                 slice_gap)
    last_forecast_datetime = utils.slice_index_to_datetime(house_forecast.year, house_forecast.month, house_forecast.day, house_forecast.slice_index,
                                                 slice_gap)
    
    return last_data_datetime - last_forecast_datetime > datetime.timedelta(minutes=slice_gap)




def main():
    try:
        model_name = "ARIMA"
        slice_gap = 5
        if len(sys.argv) > 2:
            model_name = sys.argv[1]
            slice_gap = int(sys.argv[2])
        print(f"Model: {model_name}, slice_gap: {slice_gap}")
        
        for house_id in constants.HOUSE_IDS_TO_FORECAST:
            if check_delayed_forecast(house_id, slice_gap, model_name):
                print(f"Forecast Worker for house {house_id} using model {model_name} may be error.")
                sys.exit(1)
        print("OK")
        sys.exit(0)
    except Exception as e:
        print(f"Exception while healthchecking: {e}")
        sys.exit(1)
    
if __name__ == '__main__':
    main()
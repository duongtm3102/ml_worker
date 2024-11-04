import os

SLICE_GAP_TO_MIN = {
    5: "5min"
}

SLICE_GAP_TO_RETRAIN_TIME = {
    5: "3600" # 5 min predict -> retrain model after 1 hour
}

if not os.getenv('HOUSE_IDS_TO_FORECAST'):
    HOUSE_IDS_TO_FORECAST = [0, 5, 20]
else:
    HOUSE_IDS_TO_FORECAST = os.getenv('HOUSE_IDS_TO_FORECAST').split(',')
    
NEURAL_PROPHET_N_LAGS = 2

if not os.getenv('FORECAST_INTERVAL'):
    FORECAST_INTERVAL = 1
else:
    FORECAST_INTERVAL = int(os.getenv('FORECAST_INTERVAL'))
    
if not os.getenv('RETRAIN_INTERVAL'):
    RETRAIN_INTERVAL = 30
else:
    RETRAIN_INTERVAL = int(os.getenv('RETRAIN_INTERVAL'))

if not os.getenv('SLICE_GAP'):
    SLICE_GAP = 5
else:
    SLICE_GAP = int(os.getenv('SLICE_GAP'))

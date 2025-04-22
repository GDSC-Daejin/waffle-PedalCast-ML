"""
설정 파일: 경로, 파라미터 등의 설정값 정의
"""
import os
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# 파일 경로 설정
BIKE_DATA_PATH = os.path.join(DATA_DIR, '서울특별시_공공자전거_월별_이용정보.csv')
WEATHER_DATA_PATH = os.path.join(DATA_DIR, '서울시_날씨_데이터.csv')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'bike_prediction_model.pkl')

# 모델 파라미터
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# 날씨 API 설정 (기상청 API 사용 예시)
WEATHER_API_KEY = "YOUR_API_KEY"
WEATHER_API_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"

# 서울시 좌표 (기상청 API용)
SEOUL_NX = 60
SEOUL_NY = 127

# 모델 하이퍼파라미터
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_SEED
    },
    'lightgbm': {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'regression',
        'random_state': RANDOM_SEED
    },
    'prophet': {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'seasonality_mode': 'multiplicative'
    }
}

# 날짜 관련 설정
DATE_FORMAT = '%Y-%m-%d'

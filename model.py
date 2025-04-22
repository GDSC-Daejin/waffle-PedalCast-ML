"""
모델 정의 모듈: 다양한 예측 모델 구현
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import Dict, Any, List, Tuple, Optional, Union

import xgboost as xgb
import lightgbm as lgb
# from prophet import Prophet  # <-- 전역 import 제거
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import MODEL_PARAMS, MODEL_SAVE_PATH, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BikeUsagePredictionModel:
    def __init__(self, model_type: str = 'xgboost'):
        """
        Args:
            model_type: 사용할 모델 유형 ('xgboost', 'lightgbm', 'prophet', 'randomforest', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.model_params = MODEL_PARAMS.get(model_type, {})
        self._prophet_imported = False  # prophet import 상태 플래그
        
    def _initialize_model(self):
        """모델 초기화"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**self.model_params)
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**self.model_params)
        elif self.model_type == 'prophet':
            try:
                from prophet import Prophet
                self.model = Prophet(**self.model_params)
                self._prophet_imported = True
            except ImportError:
                raise ImportError(
                    "prophet 패키지가 설치되어 있지 않습니다. "
                    "Python 3.13 환경에서는 prophet 설치가 어렵습니다. "
                    "Python 3.10 이하 환경에서 'pip install prophet' 또는 'pip install --pre prophet' 명령어로 설치해 주세요."
                )
        elif self.model_type == 'randomforest':
            self.model = RandomForestRegressor(
                n_estimators=1000, 
                max_depth=12,
                random_state=RANDOM_SEED
            )
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {self.model_type}")
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Any:
        """모델 학습"""
        logger.info(f"{self.model_type} 모델 학습 중...")
        
        if self.model is None:
            self._initialize_model()
        
        if self.model_type == 'prophet':
            if not self._prophet_imported:
                try:
                    from prophet import Prophet
                    self._prophet_imported = True
                except ImportError:
                    raise ImportError(
                        "prophet 패키지가 설치되어 있지 않습니다. "
                        "Python 3.13 환경에서는 prophet 설치가 어렵습니다. "
                        "Python 3.10 이하 환경에서 'pip install prophet' 또는 'pip install --pre prophet' 명령어로 설치해 주세요."
                    )
            # Prophet 모델은 특별한 데이터 형식 필요
            prophet_df = pd.DataFrame({
                'ds': X['rental_date'],
                'y': y
            })
            
            # 추가 회귀 변수 설정 (날씨 데이터 등)
            for col in X.columns:
                if col != 'rental_date' and col in ['avg_temp', 'precipitation', 'humidity', 'wind_speed']:
                    prophet_df[col] = X[col]
                    self.model.add_regressor(col)
            
            self.model.fit(prophet_df)
        else:
            # 일반적인 scikit-learn API를 따르는 모델
            self.model.fit(X, y)
            
            # 특성 중요도 저장 (지원하는 모델만)
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        logger.info(f"{self.model_type} 모델 학습 완료")
        return self.model
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], future_periods: int = 0) -> np.ndarray:
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        
        logger.info(f"{self.model_type} 모델로 예측 수행 중...")
        
        if self.model_type == 'prophet':
            if not self._prophet_imported:
                try:
                    from prophet import Prophet
                    self._prophet_imported = True
                except ImportError:
                    raise ImportError(
                        "prophet 패키지가 설치되어 있지 않습니다. "
                        "Python 3.13 환경에서는 prophet 설치가 어렵습니다. "
                        "Python 3.10 이하 환경에서 'pip install prophet' 또는 'pip install --pre prophet' 명령어로 설치해 주세요."
                    )
            # Prophet 모델은 미래 데이터프레임 생성 필요
            if future_periods > 0:
                future_df = self.model.make_future_dataframe(periods=future_periods)
                
                # 추가 회귀 변수가 있는 경우 처리
                for regressor in self.model.extra_regressors:
                    regressor_name = regressor['name']
                    if regressor_name in X.columns:
                        # 기존 데이터의 값 복사
                        future_df[regressor_name] = X[regressor_name].tolist() + [None] * future_periods
                        
                        # 미래 값은 과거 평균으로 대체 (더 정교한 방법으로 대체 가능)
                        future_mean = X[regressor_name].mean()
                        future_df.loc[future_df[regressor_name].isnull(), regressor_name] = future_mean
                
                forecast = self.model.predict(future_df)
                predictions = forecast['yhat'].values
            else:
                prophet_df = pd.DataFrame({
                    'ds': X['rental_date']
                })
                
                # 추가 회귀 변수 설정
                for regressor in self.model.extra_regressors:
                    regressor_name = regressor['name']
                    if regressor_name in X.columns:
                        prophet_df[regressor_name] = X[regressor_name]
                
                forecast = self.model.predict(prophet_df)
                predictions = forecast['yhat'].values
        else:
            # 일반적인 scikit-learn API를 따르는 모델
            predictions = self.model.predict(X)
        
        logger.info(f"{self.model_type} 모델 예측 완료")
        return predictions
    
    def save_model(self, path: str = MODEL_SAVE_PATH):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다. 먼저 train() 메서드를 호출하세요.")
        
        logger.info(f"모델 저장 중: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"모델 저장 완료: {path}")
    
    def load_model(self, path: str = MODEL_SAVE_PATH):
        """저장된 모델 로드"""
        logger.info(f"모델 로드 중: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"모델 로드 완료: {path}")
        return self.model
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """특성 중요도 반환"""
        return self.feature_importance


class EnsembleModel:
    """여러 모델의 앙상블"""
    def __init__(self, models: List[BikeUsagePredictionModel], weights: Optional[List[float]] = None):
        """
        Args:
            models: 앙상블에 사용할 모델 리스트
            weights: 각 모델의 가중치 (None인 경우 동일 가중치 적용)
        """
        self.models = models
        
        if weights is None:
            self.weights = [1/len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("모델 수와 가중치 수가 일치해야 합니다.")
            # 가중치 정규화
            total = sum(weights)
            self.weights = [w/total for w in weights]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """앙상블 예측 수행"""
        logger.info("앙상블 모델로 예측 수행 중...")
        
        predictions = []
        for model, weight in zip(self.models, self.weights):
            model_pred = model.predict(X)
            predictions.append(model_pred * weight)
        
        # 가중 평균 계산
        ensemble_pred = np.sum(predictions, axis=0)
        
        logger.info("앙상블 모델 예측 완료")
        return ensemble_pred

"""
예측 수행 모듈: 미래 데이터 예측 및 시각화
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, Any, List, Tuple, Optional, Union

from config import RESULT_DIR, DATE_FORMAT
from model import BikeUsagePredictionModel, EnsembleModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BikeUsagePredictor:
    def __init__(self, model: Union[BikeUsagePredictionModel, EnsembleModel], 
                feature_engineer, weather_fetcher=None):
        """
        Args:
            model: 학습된 예측 모델
            feature_engineer: 특성 공학 객체
            weather_fetcher: 날씨 데이터 수집 객체 (선택적)
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.weather_fetcher = weather_fetcher
        os.makedirs(RESULT_DIR, exist_ok=True)
    
    def generate_future_dates(self, start_date: datetime, periods: int = 365) -> pd.DataFrame:
        """미래 날짜 생성"""
        logger.info(f"{periods}일 미래 날짜 생성 중...")
        future_dates = [start_date + timedelta(days=i) for i in range(periods)]
        future_df = pd.DataFrame({
            'rental_date': future_dates
        })
        if not np.issubdtype(future_df['rental_date'].dtype, np.datetime64):
            future_df['rental_date'] = pd.to_datetime(future_df['rental_date'])
        logger.info(f"미래 날짜 생성 완료: {len(future_dates)}일")
        return future_df
    
    def generate_future_weather(self, future_dates: pd.DataFrame) -> pd.DataFrame:
        """미래 날씨 데이터 생성 (예측 또는 과거 평균)"""
        logger.info("미래 날씨 데이터 생성 중...")
        future_df = future_dates.copy()
        if 'rental_date' in future_df.columns and not np.issubdtype(future_df['rental_date'].dtype, np.datetime64):
            future_df['rental_date'] = pd.to_datetime(future_df['rental_date'])
        
        if self.weather_fetcher is not None:
            # 날씨 API를 통해 예보 데이터 가져오기 (가능한 범위까지)
            try:
                weather_forecast = self.weather_fetcher.get_weather_forecast(future_dates['rental_date'].min())
                future_df = pd.merge(
                    future_df,
                    weather_forecast,
                    on='rental_date',
                    how='left'
                )
            except Exception as e:
                logger.error(f"날씨 예보 데이터 수집 실패: {str(e)}")
        
        # 결측치는 과거 데이터의 월별/요일별 평균으로 대체
        if 'avg_temp' not in future_df.columns:
            # 기본 날씨 변수 추가
            future_df['avg_temp'] = None
            future_df['max_temp'] = None
            future_df['min_temp'] = None
            future_df['precipitation'] = None
            future_df['humidity'] = None
            future_df['wind_speed'] = None
            future_df['weather_condition'] = None
        
        # 과거 데이터의 월별/요일별 평균 계산 (예시)
        # 실제 구현 시에는 과거 데이터를 이용해 계산
        monthly_avg_temp = {
            1: 0, 2: 2, 3: 8, 4: 14, 5: 19, 6: 23,
            7: 26, 8: 27, 9: 22, 10: 16, 11: 9, 12: 2
        }
        
        monthly_precipitation_prob = {
            1: 0.1, 2: 0.15, 3: 0.2, 4: 0.25, 5: 0.3, 6: 0.4,
            7: 0.5, 8: 0.45, 9: 0.3, 10: 0.2, 11: 0.15, 12: 0.1
        }
        
        # 날짜 기반 특성 추가
        future_df['year'] = future_df['rental_date'].dt.year
        future_df['month'] = future_df['rental_date'].dt.month
        future_df['day'] = future_df['rental_date'].dt.day
        future_df['dayofweek'] = future_df['rental_date'].dt.dayofweek
        
        # 결측치 채우기
        for idx, row in future_df.iterrows():
            month = row['month']
            
            # 온도 데이터 (월별 평균 + 랜덤 변동)
            if pd.isnull(row['avg_temp']):
                base_temp = monthly_avg_temp[month]
                random_var = np.random.normal(0, 2)  # 표준편차 2도의 랜덤 변동
                future_df.at[idx, 'avg_temp'] = base_temp + random_var
                future_df.at[idx, 'max_temp'] = base_temp + random_var + 5
                future_df.at[idx, 'min_temp'] = base_temp + random_var - 5
            
            # 강수량 (월별 확률 기반)
            if pd.isnull(row['precipitation']):
                if np.random.random() < monthly_precipitation_prob[month]:
                    future_df.at[idx, 'precipitation'] = np.random.exponential(5)  # 평균 5mm
                else:
                    future_df.at[idx, 'precipitation'] = 0
            
            # 습도 (계절 기반)
            if pd.isnull(row['humidity']):
                if month in [6, 7, 8]:  # 여름
                    future_df.at[idx, 'humidity'] = np.random.normal(70, 10)
                elif month in [12, 1, 2]:  # 겨울
                    future_df.at[idx, 'humidity'] = np.random.normal(50, 15)
                else:  # 봄, 가을
                    future_df.at[idx, 'humidity'] = np.random.normal(60, 12)
            
            # 풍속
            if pd.isnull(row['wind_speed']):
                future_df.at[idx, 'wind_speed'] = np.random.normal(2, 1)
            
            # 날씨 상태
            if pd.isnull(row['weather_condition']):
                if future_df.at[idx, 'precipitation'] > 0:
                    future_df.at[idx, 'weather_condition'] = '비'
                elif future_df.at[idx, 'avg_temp'] < 0:
                    future_df.at[idx, 'weather_condition'] = '눈'
                elif future_df.at[idx, 'humidity'] > 80:
                    future_df.at[idx, 'weather_condition'] = '흐림'
                else:
                    future_df.at[idx, 'weather_condition'] = '맑음'
        
        logger.info("미래 날씨 데이터 생성 완료")
        return future_df
    
    def prepare_future_features(self, future_df: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
        """미래 데이터의 특성 준비"""
        logger.info("미래 데이터 특성 준비 중...")
        
        # 시간 관련 특성 생성
        future_df = self.feature_engineer.create_time_features(future_df)
        
        # 날씨 관련 특성 생성
        future_df = self.feature_engineer.create_weather_features(future_df)
        
        # 순환 특성 생성
        future_df = self.feature_engineer.create_cyclical_features(future_df)
        
        # 시차 특성 생성을 위한 과거 데이터 필요
        # 과거 데이터의 타겟 변수를 미래 데이터에 추가

        # rental_date 컬럼이 없으면 인덱스에서 복구 시도
        if 'rental_date' not in historical_data.columns:
            if historical_data.index.name == 'rental_date':
                historical_data = historical_data.reset_index()
            elif 'rental_date' in historical_data.index.names:
                historical_data = historical_data.reset_index()
            else:
                raise KeyError("'rental_date' 컬럼이 historical_data에 존재하지 않습니다. 전처리 과정을 확인하세요.")

        combined_df = pd.concat([
            historical_data[['rental_date', 'usage_count']],
            future_df[['rental_date']]
        ]).sort_values('rental_date').reset_index(drop=True)
        
        # 시차 특성 생성
        combined_df = self.feature_engineer.create_lag_features(combined_df)
        
        # 미래 데이터만 필터링
        future_with_features = pd.merge(
            future_df,
            combined_df.drop('usage_count', axis=1, errors='ignore'),
            on='rental_date',
            how='left'
        )
        
        logger.info("미래 데이터 특성 준비 완료")
        return future_with_features
    
    def predict_future(self, start_date: datetime, periods: int = 365, 
                      historical_data: pd.DataFrame = None) -> pd.DataFrame:
        """미래 이용량 예측"""
        logger.info(f"{periods}일 미래 이용량 예측 시작...")
        
        # 미래 날짜 생성
        future_dates = self.generate_future_dates(start_date, periods)
        
        # 미래 날씨 데이터 생성
        future_with_weather = self.generate_future_weather(future_dates)
        
        # 미래 특성 준비
        future_with_features = self.prepare_future_features(future_with_weather, historical_data)
        
        # 예측 수행
        if isinstance(self.model, BikeUsagePredictionModel) and self.model.model_type == 'prophet':
            # Prophet 모델은 특별한 처리 필요
            prophet_future = pd.DataFrame({
                'ds': future_with_features['rental_date']
            })
            
            # 추가 회귀 변수 설정
            for regressor in self.model.model.extra_regressors:
                regressor_name = regressor['name']
                if regressor_name in future_with_features.columns:
                    prophet_future[regressor_name] = future_with_features[regressor_name]
            
            forecast = self.model.model.predict(prophet_future)
            predictions = forecast['yhat'].values
        else:
            # 예측에 필요한 특성만 선택
            X_cols = [col for col in future_with_features.columns if col != 'rental_date' and col != 'usage_count']
            predictions = self.model.predict(future_with_features[X_cols])
        
        # 예측 결과를 데이터프레임에 추가
        future_with_features['predicted_usage'] = predictions
        
        # 음수 예측값 보정
        future_with_features['predicted_usage'] = future_with_features['predicted_usage'].clip(lower=0)
        
        logger.info("미래 이용량 예측 완료")
        return future_with_features
    
    def plot_future_predictions(self, predictions_df: pd.DataFrame, 
                               historical_data: Optional[pd.DataFrame] = None,
                               title: str = 'Future Bike Usage Predictions',
                               save_path: Optional[str] = None) -> None:
        """미래 예측 결과 시각화"""
        plt.figure(figsize=(15, 7))
        
        # 과거 데이터 표시 (있는 경우)
        if historical_data is not None:
            historical = historical_data.sort_values('rental_date')
            plt.plot(historical['rental_date'], historical['usage_count'], 
                    label='Historical Usage', color='blue', alpha=0.7)
        
        # 미래 예측 표시
        predictions = predictions_df.sort_values('rental_date')
        plt.plot(predictions['rental_date'], predictions['predicted_usage'], 
                label='Predicted Usage', color='red', linestyle='--', alpha=0.7)
        
        # 월별 구분선 추가
        all_dates = pd.date_range(
            start=predictions['rental_date'].min() - pd.Timedelta(days=30) if historical_data is None else historical['rental_date'].min(),
            end=predictions['rental_date'].max(),
            freq='MS'  # 월 시작일
        )
        for date in all_dates:
            plt.axvline(x=date, color='gray', linestyle=':', alpha=0.3)
        
        # 그래프 꾸미기
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Usage Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # x축 날짜 포맷 설정
        plt.gcf().autofmt_xdate()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"그래프 저장 완료: {save_path}")
        
        plt.show()
    
    def export_predictions(self, predictions_df: pd.DataFrame, 
                          export_path: str = os.path.join(RESULT_DIR, 'future_predictions.csv')) -> None:
        """예측 결과 내보내기"""
        # 필요한 열만 선택
        export_df = predictions_df[['rental_date', 'predicted_usage', 'avg_temp', 'precipitation']]
        
        # 날짜 형식 변환
        export_df['rental_date'] = export_df['rental_date'].dt.strftime(DATE_FORMAT)
        
        # CSV 파일로 저장
        export_df.to_csv(export_path, index=False)
        logger.info(f"예측 결과 내보내기 완료: {export_path}")

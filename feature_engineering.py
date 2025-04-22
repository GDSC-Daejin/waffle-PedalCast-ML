"""
특성 공학 모듈: 새로운 특성 생성, 변환 등
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import logging
from typing import Tuple, List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime%s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        try:
            self.kr_holidays = holidays.KR()
        except AttributeError:
            self.kr_holidays = holidays.country_holidays('KR')
        
    def create_time_features(self, df: pd.DataFrame, date_col: str = 'rental_date') -> pd.DataFrame:
        """시간 관련 특성 생성"""
        logger.info("시간 관련 특성 생성 중...")
        df = df.copy()
        if (date_col in df.columns and not np.issubdtype(df[date_col].dtype, np.datetime64)):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 기본 시간 특성
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek  # 0: 월요일, 6: 일요일
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # 계절 특성
        df['season'] = df['month'].apply(lambda x: 1 if x in [3, 4, 5] else  # 봄
                                         2 if x in [6, 7, 8] else  # 여름
                                         3 if x in [9, 10, 11] else  # 가을
                                         4)  # 겨울
        
        # 공휴일 특성 robust하게 생성
        def is_holiday_safe(x):
            if pd.isnull(x):
                return 0
            try:
                return 1 if x in self.kr_holidays else 0
            except Exception:
                return 0
        df['is_holiday'] = df[date_col].apply(is_holiday_safe)
        
        # 분기 특성
        df['quarter'] = df['month'].apply(lambda x: (x - 1) // 3 + 1)
        
        # 월초, 월중, 월말 특성
        df['month_part'] = df['day'].apply(lambda x: 1 if x <= 10 else 2 if x <= 20 else 3)
        
        logger.info("시간 관련 특성 생성 완료")
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """날씨 관련 특성 생성"""
        logger.info("날씨 관련 특성 생성 중...")
        df = df.copy()

        # 날씨 관련 컬럼이 없으면 생성 (NaN으로)
        weather_cols = ['precipitation', 'avg_temp', 'humidity', 'wind_speed', 'weather_condition']
        for col in weather_cols:
            if (col not in df.columns):
                df[col] = np.nan

        # 만약 날씨 데이터가 모두 결측치라면(즉, 날씨 데이터가 없는 경우) 특성 생성 건너뜁
        if df[weather_cols].isnull().all().all():
            logger.info("날씨 데이터가 없어 날씨 특성 생성을 건너뜁니다.")
            return df

        # 강수 여부
        df['is_rainy'] = (df['precipitation'] > 0).astype('Int64')

        # 기온 구간화
        df['temp_category'] = pd.cut(
            df['avg_temp'],
            bins=[-float('inf'), 0, 10, 20, 30, float('inf')],
            labels=[0, 1, 2, 3, 4]
        ).astype('Int64')

        # 습도 구간화
        df['humidity_category'] = pd.cut(
            df['humidity'],
            bins=[0, 30, 60, 100],
            labels=[0, 1, 2]
        ).astype('Int64')

        # 풍속 구간화
        df['wind_category'] = pd.cut(
            df['wind_speed'],
            bins=[0, 2, 5, float('inf')],
            labels=[0, 1, 2]
        ).astype('Int64')

        # 날씨 상태 원-핫 인코딩 (필요시)
        if 'weather_condition' in df.columns:
            weather_dummies = pd.get_dummies(df['weather_condition'], prefix='weather')
            df = pd.concat([df, weather_dummies], axis=1)

        # 극단적 날씨 조건 특성
        df['extreme_weather'] = ((df['avg_temp'] < -5) | (df['avg_temp'] > 33) | 
                                (df['precipitation'] > 20) | 
                                (df['wind_speed'] > 7)).astype('Int64')

        logger.info("날씨 관련 특성 생성 완료")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'usage_count', lags: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
        """시차(lag) 특성 생성"""
        logger.info("시차(lag) 특성 생성 중...")
        df = df.copy()
        
        # 데이터를 날짜순으로 정렬
        df = df.sort_values('rental_date')
        
         # 각 시차에 대한 특성 생성
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # 이동 평균 특성
        df[f'{target_col}_ma_7'] = df[target_col].rolling(window=7).mean()
        df[f'{target_col}_ma_14'] = df[target_col].rolling(window=14).mean()
        df[f'{target_col}_ma_30'] = df[target_col].rolling(window=30).mean()
        
        # 이동 표준편차 특성
        df[f'{target_col}_std_7'] = df[target_col].rolling(window=7).std()
        df[f'{target_col}_std_30'] = df[target_col].rolling(window=30).std()
        
        logger.info("시차(lag) 특성 생성 완료")
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """순환 특성 생성 (월, 요일 등을 사인, 코사인 변환)"""
        logger.info("순환 특성 생성 중...")
        df = df.copy()
        
        # 월 순환 특성
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 요일 순환 특성
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # 일 순환 특성
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        logger.info("순환 특성 생성 완료")
        return df
    
    def feature_selection(self, df: pd.DataFrame, target_col: str = 'usage_count') -> pd.DataFrame:
        """특성 선택 (중요도가 낮은 특성 제거)"""
        logger.info("특성 선택 중...")
        df = df.copy()
        
        # 결측치가 많은 특성 제거
        missing_threshold = 0.3
        missing_cols = [col for col in df.columns if df[col].isnull().mean() > missing_threshold]
        df = df.drop(columns=missing_cols, errors='ignore')
        
        # 상관관계가 높은 특성 중 하나만 선택 (선택적)
        # 여기서는 생략하고 모델의 특성 중요도를 통해 나중에 처리
        
        logger.info("특성 선택 완료")
        return df
    
    def process_features(self, df: pd.DataFrame, target_col: str = 'usage_count') -> pd.DataFrame:
        """모든 특성 공학 과정을 순차적으로 적용"""
        # rental_date가 인덱스에 있으면 복구 (최우선)
        if 'rental_date' not in df.columns:
            if df.index.name == 'rental_date':
                df = df.reset_index()
            elif 'rental_date' in df.index.names:
                df = df.reset_index()

        # rental_date 컬럼이 없으면 에러 발생 (이 시점에서 반드시 존재해야 함)
        if 'rental_date' not in df.columns:
            raise KeyError("'rental_date' 컬럼이 존재하지 않습니다. create_time_features()를 확인하세요.")

        df = self.create_time_features(df)
        df = self.create_weather_features(df)
        df = self.create_lag_features(df, target_col)
        df = self.create_cyclical_features(df)
        df = self.feature_selection(df, target_col)

        # rental_date가 결측치인 행은 미리 제거
        if 'rental_date' in df.columns:
            df = df[df['rental_date'].notnull()]

        # rental_date 컬럼은 결측치 제거 대상에서 제외
        non_date_cols = [col for col in df.columns if col != 'rental_date']
        df = df.dropna(subset=non_date_cols)

        # 최종적으로 rental_date 컬럼이 없으면 에러
        if 'rental_date' not in df.columns:
            raise KeyError("'rental_date' 컬럼이 전처리 후에도 존재하지 않습니다.")

        return df

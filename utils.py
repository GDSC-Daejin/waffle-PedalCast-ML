"""
유틸리티 함수 모듈: 보조 기능 구현
"""
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
import os

from config import WEATHER_API_KEY, WEATHER_API_URL, SEOUL_NX, SEOUL_NY, DATE_FORMAT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherDataFetcher:
    """날씨 데이터 수집 클래스"""
    def __init__(self, api_key: str = WEATHER_API_KEY, api_url: str = WEATHER_API_URL):
        self.api_key = api_key
        self.api_url = api_url
    
    def get_weather_forecast(self, start_date: datetime, days: int = 7) -> pd.DataFrame:
        """
        기상청 API를 통해 날씨 예보 데이터 수집
        
        Args:
            start_date: 시작 날짜
            days: 예보 일수 (최대 7일)
            
        Returns:
            날씨 예보 데이터프레임
        """
        logger.info(f"{start_date}부터 {days}일간의 날씨 예보 데이터 수집 중...")
        
        # 실제 API 호출 구현 (예시)
        # 여기서는 간단한 예시 데이터 반환
        
        forecast_dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # 예시 데이터 생성
        forecast_data = pd.DataFrame({
            'rental_date': forecast_dates,
            'avg_temp': np.random.normal(15, 5, days),
            'max_temp': np.random.normal(20, 5, days),
            'min_temp': np.random.normal(10, 5, days),
            'precipitation': np.random.exponential(1, days),
            'humidity': np.random.normal(60, 10, days),
            'wind_speed': np.random.normal(2, 1, days),
            'weather_condition': np.random.choice(['맑음', '흐림', '비', '눈'], days)
        })
        
        logger.info(f"날씨 예보 데이터 수집 완료: {len(forecast_data)}일")
        return forecast_data
    
    def get_historical_weather(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        과거 날씨 데이터 수집
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            과거 날씨 데이터프레임
        """
        logger.info(f"{start_date}부터 {end_date}까지의 과거 날씨 데이터 수집 중...")
        
        # 실제 API 호출 구현 (예시)
        # 여기서는 간단한 예시 데이터 반환
        
        days = (end_date - start_date).days + 1
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # 예시 데이터 생성
        historical_data = pd.DataFrame({
            'rental_date': dates,
            'avg_temp': np.random.normal(15, 10, days),
            'max_temp': np.random.normal(20, 10, days),
            'min_temp': np.random.normal(10, 10, days),
            'precipitation': np.random.exponential(1, days),
            'humidity': np.random.normal(60, 15, days),
            'wind_speed': np.random.normal(2, 1, days),
            'weather_condition': np.random.choice(['맑음', '흐림', '비', '눈'], days)
        })
        
        logger.info(f"과거 날씨 데이터 수집 완료: {len(historical_data)}일")
        return historical_data


def create_date_range(start_date: str, end_date: str, format: str = DATE_FORMAT) -> List[datetime]:
    """날짜 범위 생성"""
    start = datetime.strptime(start_date, format)
    end = datetime.strptime(end_date, format)
    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(days)]


def save_model_results(model_name: str, metrics: Dict[str, float], 
                      result_dir: str, timestamp: Optional[str] = None) -> None:
    """모델 평가 결과 저장"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    result_path = os.path.join(result_dir, f'{model_name}_results_{timestamp}.json')
    
    with open(result_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"모델 결과 저장 완료: {result_path}")


def load_model_results(result_path: str) -> Dict[str, float]:
    """저장된 모델 평가 결과 로드"""
    with open(result_path, 'r') as f:
        metrics = json.load(f)
    
    logger.info(f"모델 결과 로드 완료: {result_path}")
    return metrics

"""
데이터 전처리 모듈: 데이터 로드, 정제, 결측치 처리 등
"""
import pandas as pd
import numpy as np
import os
import glob
import logging
from typing import Tuple, Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_dir='./data'):
        """
        데이터 전처리 클래스 초기화
        
        Args:
            data_dir: 데이터 파일이 저장된 디렉토리 경로
        """
        self.data_dir = data_dir
        self.bike_data = None
        self.weather_data = None
        self.merged_data = None
    
    def find_bike_data_files(self) -> List[str]:
        """
        데이터 디렉토리에서 따릉이 이용정보 CSV 파일 목록 찾기
        
        Returns:
            파일 경로 리스트
        """
        # 다양한 파일명 패턴 검색
        patterns = [
            os.path.join(self.data_dir, '서울특별시 공공자전거 이용정보(월별)_*.csv'),
            os.path.join(self.data_dir, '서울특별시_공공자전거_이용정보_월별_*.csv'),
            os.path.join(self.data_dir, 'seoulteugbyeolsi-gonggongjajeongeo-iyongjeongbo-weolbyeol-*.csv')
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        
        logger.info(f"발견된 파일 수: {len(files)}")
        return files
    
    def load_bike_data(self) -> pd.DataFrame:
        """
        따릉이 이용 데이터 로드 및 통합
        
        Returns:
            통합된 데이터프레임
        """
        logger.info("따릉이 이용 데이터 로드 중...")
        
        files = self.find_bike_data_files()
        if not files:
            logger.error(f"'{self.data_dir}' 디렉토리에서 데이터 파일을 찾을 수 없습니다.")
            raise FileNotFoundError(f"'{self.data_dir}' 디렉토리에서 데이터 파일을 찾을 수 없습니다.")
        
        # 데이터프레임 리스트
        dfs = []
        
        # 예상 컬럼명
        expected_columns = [
            '대여일자', '대여소번호', '대여소', '대여구분코드', 
            '성별', '연령대코드', '이용건수', '운동량', 
            '탄소량', '이동거리(M)', '이동시간(분)'
        ]
        
        for file in files:
            try:
                # 인코딩 시도 (utf-8 또는 cp949)
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding='cp949')
                
                # 컬럼명에서 따옴표 제거
                df.columns = [col.strip("'").strip() for col in df.columns]
                
                # 예상 컬럼명과 일치하는지 확인하고 필요시 조정
                if not all(col in df.columns for col in expected_columns):
                    logger.warning(f"파일 '{file}'의 컬럼이 예상과 다릅니다. 컬럼명을 조정합니다.")
                    # 컬럼 수가 맞으면 컬럼명 재지정
                    if len(df.columns) == len(expected_columns):
                        df.columns = expected_columns
                
                # 데이터 값에서 따옴표 제거
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.strip("'").str.strip()
                
                # 대여일자 처리
                try:
                    # YYYYMMDD 또는 YYYYMM 형식 처리
                    if df['대여일자'].str.len().max() <= 8:
                        df['대여일자'] = pd.to_datetime(df['대여일자'], format='%Y%m%d', errors='coerce')
                    else:
                        df['대여일자'] = pd.to_datetime(df['대여일자'], errors='coerce')
                except Exception as e:
                    logger.warning(f"대여일자 변환 오류: {str(e)}. 기본 변환 시도.")
                    df['대여일자'] = pd.to_datetime(df['대여일자'], errors='coerce')

                # Ensure '대여일자' is datetime
                if not pd.api.types.is_datetime64_any_dtype(df['대여일자']):
                    df['대여일자'] = pd.to_datetime(df['대여일자'], errors='coerce')
                
                # 숫자형 컬럼 변환
                numeric_cols = ['이용건수', '운동량', '탄소량', '이동거리(M)', '이동시간(분)']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                dfs.append(df)
                logger.info(f"파일 '{file}' 로드 완료: {df.shape[0]} 행, {df.shape[1]} 열")
            
            except Exception as e:
                logger.error(f"파일 '{file}' 로드 실패: {str(e)}")
        
        if not dfs:
            logger.error("로드된 데이터가 없습니다.")
            raise ValueError("로드된 데이터가 없습니다.")
        
        # 데이터프레임 통합
        self.bike_data = pd.concat(dfs, ignore_index=True)
        logger.info(f"데이터 로드 완료: 총 {self.bike_data.shape[0]} 행, {self.bike_data.shape[1]} 열")
        
        return self.bike_data
    
    def clean_bike_data(self) -> pd.DataFrame:
        """
        따릉이 데이터 정제
        
        Returns:
            정제된 데이터프레임
        """
        if self.bike_data is None:
            self.load_bike_data()
            
        logger.info("따릉이 데이터 정제 중...")
        df = self.bike_data.copy()
        
        # 열 이름 영문화 (필요시)
        column_mapping = {
            '대여일자': 'rental_date',
            '대여소번호': 'station_id',
            '대여소': 'station_name',
            '대여구분코드': 'rental_type',
            '성별': 'gender',
            '연령대코드': 'age_group',
            '이용건수': 'usage_count',
            '운동량': 'exercise_amount',
            '탄소량': 'carbon_reduction',
            '이동거리(M)': 'distance',
            '이동시간(분)': 'duration'
        }
        
        # 컬럼명 변경 전에 존재하는 컬럼만 매핑
        actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=actual_mapping)
        
        # rental_date 컬럼이 없으면 생성 시도 (예: '대여일자' 등에서)
        if 'rental_date' not in df.columns:
            for cand in ['대여일자', 'date', '일시']:
                if cand in df.columns:
                    df['rental_date'] = pd.to_datetime(df[cand], errors='coerce')
                    break

        # rental_date 컬럼이 여전히 없으면 에러
        if 'rental_date' not in df.columns:
            raise KeyError("'rental_date' 컬럼을 찾을 수 없습니다. 원본 데이터에 날짜 컬럼이 포함되어 있는지 확인하세요.")

        # 결측치 처리
        df = df.fillna({
            'gender': '미정',
            'age_group': '미정',
            'usage_count': 0,
            'exercise_amount': 0,
            'carbon_reduction': 0,
            'distance': 0,
            'duration': 0
        })
        
        # 이상치 처리 (예: 이용건수가 음수인 경우)
        df = df[df['usage_count'] >= 0]
        
        # 날짜 관련 특성 추가
        if pd.api.types.is_datetime64_any_dtype(df['rental_date']):
            df['year'] = df['rental_date'].dt.year
            df['month'] = df['rental_date'].dt.month
            df['day'] = df['rental_date'].dt.day
            df['dayofweek'] = df['rental_date'].dt.dayofweek
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        logger.info("따릉이 데이터 정제 완료")
        self.bike_data = df
        return df
    
    def load_weather_data(self, weather_file_path: Optional[str] = None) -> pd.DataFrame:
        """
        날씨 데이터 로드 (파일이 있는 경우)
        
        Args:
            weather_file_path: 날씨 데이터 파일 경로 (없으면 디렉토리에서 검색)
            
        Returns:
            날씨 데이터프레임
        """
        logger.info("날씨 데이터 로드 중...")
        
        # 날씨 파일 경로가 지정되지 않은 경우 디렉토리에서 검색
        if weather_file_path is None:
            weather_patterns = [
                os.path.join(self.data_dir, '*날씨*.csv'),
                os.path.join(self.data_dir, '*weather*.csv')
            ]
            
            weather_files = []
            for pattern in weather_patterns:
                weather_files.extend(glob.glob(pattern))
            
            if weather_files:
                weather_file_path = weather_files[0]
                logger.info(f"날씨 데이터 파일 발견: {weather_file_path}")
            else:
                logger.warning("날씨 데이터 파일을 찾을 수 없습니다.")
                return None
        
        try:
            # 인코딩 시도
            try:
                weather_df = pd.read_csv(weather_file_path, encoding='utf-8')
            except UnicodeDecodeError:
                weather_df = pd.read_csv(weather_file_path, encoding='cp949')
            
            # 날짜 컬럼 찾기 및 변환
            date_columns = [col for col in weather_df.columns if '일시' in col or 'date' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                weather_df[date_col] = pd.to_datetime(weather_df[date_col], errors='coerce')
                
                # 날짜 관련 특성 추가
                weather_df['year'] = weather_df[date_col].dt.year
                weather_df['month'] = weather_df[date_col].dt.month
                weather_df['day'] = weather_df[date_col].dt.day
                weather_df['hour'] = weather_df[date_col].dt.hour
            
            logger.info(f"날씨 데이터 로드 완료: {weather_df.shape[0]} 행, {weather_df.shape[1]} 열")
            self.weather_data = weather_df
            return weather_df
            
        except Exception as e:
            logger.error(f"날씨 데이터 로드 실패: {str(e)}")
            return None
    
    def merge_data(self, weather_file_path: Optional[str] = None) -> pd.DataFrame:
        """
        따릉이 데이터와 날씨 데이터 병합 (날씨 데이터가 있는 경우)
        
        Args:
            weather_file_path: 날씨 데이터 파일 경로
            
        Returns:
            병합된 데이터프레임
        """
        if self.bike_data is None:
            self.clean_bike_data()
        
        # 날씨 데이터 로드 시도
        weather_df = self.load_weather_data(weather_file_path)
        
        # 날씨 데이터가 없으면 따릉이 데이터만 반환
        if weather_df is None:
            logger.warning("날씨 데이터가 없어 따릉이 데이터만 반환합니다.")
            self.merged_data = self.bike_data
            return self.bike_data
        
        logger.info("데이터 병합 중...")
        
        # 날짜 컬럼 찾기
        bike_date_col = 'rental_date'
        weather_date_cols = [col for col in weather_df.columns if '일시' in col or 'date' in col.lower()]
        
        if not weather_date_cols:
            logger.warning("날씨 데이터에서 날짜 컬럼을 찾을 수 없습니다. 병합을 건너뜁니다.")
            self.merged_data = self.bike_data
            return self.bike_data
        
        weather_date_col = weather_date_cols[0]
        
        # 병합 키 설정 (년, 월, 일, 시간)
        if 'hour' in weather_df.columns and 'hour' in self.bike_data.columns:
            # 시간별 데이터인 경우
            merge_keys = ['year', 'month', 'day', 'hour']
        else:
            # 일별 또는 월별 데이터인 경우
            merge_keys = ['year', 'month']
            if 'day' in weather_df.columns and 'day' in self.bike_data.columns:
                merge_keys.append('day')
        
        # 병합 수행
        try:
            merged_df = pd.merge(
                self.bike_data,
                weather_df.drop(columns=[weather_date_col], errors='ignore'),
                on=merge_keys,
                how='left'
            )
            
            logger.info(f"데이터 병합 완료: {merged_df.shape[0]} 행, {merged_df.shape[1]} 열")
            self.merged_data = merged_df
            return merged_df
            
        except Exception as e:
            logger.error(f"데이터 병합 실패: {str(e)}")
            self.merged_data = self.bike_data
            return self.bike_data
    
    def get_preprocessed_data(self, weather_file_path: Optional[str] = None) -> pd.DataFrame:
        """
        전처리된 데이터 반환
        
        Args:
            weather_file_path: 날씨 데이터 파일 경로 (선택적)
            
        Returns:
            전처리된 데이터프레임
        """
        if self.merged_data is None:
            self.merge_data(weather_file_path)
        return self.merged_data

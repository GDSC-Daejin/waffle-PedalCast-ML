"""
메인 실행 파일: 전체 프로세스 실행
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging
import argparse
from typing import Dict, Any, List, Tuple, Optional

# 모듈 임포트
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model import BikeUsagePredictionModel, EnsembleModel
from evaluation import ModelEvaluator
from weight_calculation import WeightCalculator
from prediction import BikeUsagePredictor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levellevelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bike_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='서울시 공공자전거 이용량 예측 시스템')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='데이터 파일이 저장된 디렉토리 경로')
    
    parser.add_argument('--weather_file', type=str, default=None,
                        help='날씨 데이터 파일 경로 (없으면 자동 검색)')
    
    parser.add_argument('--mode', type=str, default='train_predict',
                        choices=['train', 'predict', 'train_predict', 'evaluate'],
                        help='실행 모드 (train: 학습만, predict: 예측만, train_predict: 학습 후 예측, evaluate: 평가만)')
    
    parser.add_argument('--model_type', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'prophet', 'randomforest', 'linear', 'ensemble'],
                        help='사용할 모델 유형')
    
    parser.add_argument('--start_date', type=str, default=None,
                        help='예측 시작 날짜 (YYYY-MM-DD 형식, 기본값: 마지막 학습 데이터 다음 날)')
    
    parser.add_argument('--periods', type=int, default=365,
                        help='예측 기간 (일 단위, 기본값: 365일)')
    
    parser.add_argument('--model_path', type=str, default='./models/bike_prediction_model.pkl',
                        help='모델 저장/로드 경로')
    
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='결과 저장 디렉토리')
    
    parser.add_argument('--save_model', action='store_true',
                        help='학습된 모델 저장 여부')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    args = parse_args()
    
    # 결과 디렉토리 생성
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    logger.info("서울시 공공자전거 이용량 예측 시스템 시작")
    logger.info(f"실행 모드: {args.mode}, 모델 유형: {args.model_type}")
    
    # 데이터 전처리
    preprocessor = DataPreprocessor(data_dir=args.data_dir)
    processed_data = preprocessor.get_preprocessed_data(args.weather_file)

    # Check if 'rental_date' exists and is datetime
    if 'rental_date' not in processed_data.columns:
        logger.error("'rental_date' 컬럼이 processed_data에 없습니다. DataPreprocessor 단계를 확인하세요.")
        raise KeyError("'rental_date' 컬럼이 processed_data에 없습니다.")
    if not pd.api.types.is_datetime64_any_dtype(processed_data['rental_date']):
        logger.error("'rental_date' 컬럼이 datetime 형식이 아닙니다. DataPreprocessor 단계를 확인하세요.")
        raise TypeError("'rental_date' 컬럼이 datetime 형식이 아닙니다.")
    
    # 특성 공학
    feature_engineer = FeatureEngineer()
    processed_data = feature_engineer.process_features(processed_data)

    # rental_date 컬럼 확인
    if 'rental_date' not in processed_data.columns:
        logger.error("'rental_date' 컬럼이 전처리 후에도 존재하지 않습니다. 데이터 전처리 과정을 확인하세요.")
        raise KeyError("'rental_date' 컬럼이 전처리 후에도 존재하지 않습니다.")
    
    # 학습/테스트 데이터 분할
    test_size = 0.2
    train_size = int(len(processed_data) * (1 - test_size))
    train_data = processed_data.iloc[:train_size]
    test_data = processed_data.iloc[train_size:]
    
    # 특성과 타겟 분리
    target_col = 'usage_count'
    date_col = 'rental_date'
    
    # 사용할 특성 선택 (날짜 컬럼과 타겟 컬럼 제외)
    feature_cols = [col for col in processed_data.columns 
                   if col != date_col and col != target_col]

    # object(문자열) 타입 컬럼 자동 제거 (XGBoost 등 호환성)
    object_cols = processed_data[feature_cols].select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        logger.warning(f"모델 입력에서 object 타입 컬럼 자동 제외: {object_cols}")
        feature_cols = [col for col in feature_cols if col not in object_cols]

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    dates_train = train_data[date_col] if date_col in train_data.columns else None

    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    dates_test = test_data[date_col] if date_col in test_data.columns else None
    
    # 모델 학습 또는 로드
    if args.mode in ['train', 'train_predict']:
        if args.model_type == 'ensemble':
            # 개별 모델 학습
            models = {}
            for model_type in ['xgboost', 'lightgbm', 'randomforest']:
                model = BikeUsagePredictionModel(model_type=model_type)
                model.train(X_train, y_train)
                models[model_type] = model
            
            # 앙상블 모델 생성
            weight_calculator = WeightCalculator()
            predictions = [model.predict(X_test) for model in models.values()]
            weights = weight_calculator.calculate_optimal_weights(predictions, y_test.values)
            
            model = EnsembleModel(list(models.values()), weights)
        else:
            # 단일 모델 학습
            model = BikeUsagePredictionModel(model_type=args.model_type)
            model.train(X_train, y_train)
        
        # 모델 저장 (옵션에 따라)
        if args.save_model and hasattr(model, 'save_model'):
            model.save_model(args.model_path)
            logger.info(f"모델 저장 완료: {args.model_path}")
    elif args.mode in ['predict', 'evaluate']:
        # 저장된 모델 로드
        model = BikeUsagePredictionModel(model_type=args.model_type)
        try:
            model.load_model(args.model_path)
            logger.info(f"모델 로드 완료: {args.model_path}")
        except FileNotFoundError:
            logger.error(f"모델 파일을 찾을 수 없습니다: {args.model_path}")
            return
    
    # 모델 평가
    if args.mode in ['train', 'train_predict', 'evaluate']:
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, X_test, y_test, dates_test, model_name=args.model_type)
        
        # 결과 저장
        result_file = os.path.join(args.result_dir, f'{args.model_type}_metrics.txt')
        with open(result_file, 'w') as f:
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value}\n")
        logger.info(f"평가 결과 저장 완료: {result_file}")
        
        # 특성 중요도 시각화 (가능한 경우)
        if hasattr(model, 'get_feature_importance') and model.get_feature_importance() is not None:
            evaluator.plot_feature_importance(
                model.get_feature_importance(),
                title=f'{args.model_type} - 특성 중요도',
                save_path=os.path.join(args.result_dir, f'{args.model_type}_feature_importance.png')
            )
        
        # 시계열 분해 시각화
        if date_col in processed_data.columns and len(processed_data) > 30:
            try:
                evaluator.plot_seasonal_decomposition(
                    processed_data, date_col, target_col,
                    save_path=os.path.join(args.result_dir, 'seasonal_decomposition.png')
                )
            except Exception as e:
                logger.warning(f"시계열 분해 시각화 실패: {str(e)}")
    
    # 미래 예측
    if args.mode in ['predict', 'train_predict']:
        # 예측 시작 날짜 설정
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        else:
            # 마지막 학습 데이터 다음 날
            if date_col in processed_data.columns:
                start_date = processed_data[date_col].max() + timedelta(days=1)
            else:
                start_date = datetime.now()
        
        # 예측기 초기화
        predictor = BikeUsagePredictor(model, feature_engineer)
        
        # 미래 이용량 예측
        future_predictions = predictor.predict_future(
            start_date=start_date,
            periods=args.periods,
            historical_data=processed_data
        )
        
        # 예측 결과 시각화
        predictor.plot_future_predictions(
            future_predictions,
            historical_data=processed_data,
            title=f'서울시 공공자전거 이용량 예측 ({args.model_type} 모델)',
            save_path=os.path.join(args.result_dir, f'{args.model_type}_future_predictions.png')
        )
        
        # 예측 결과 내보내기
        export_path = os.path.join(args.result_dir, f'{args.model_type}_future_predictions.csv')
        predictor.export_predictions(future_predictions, export_path)
        logger.info(f"예측 결과 저장 완료: {export_path}")
    
    logger.info("서울시 공공자전거 이용량 예측 시스템 종료")


if __name__ == "__main__":
    main()

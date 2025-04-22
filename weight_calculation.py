"""
가중치 계산 모듈: 앙상블 모델의 최적 가중치 계산
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeightCalculator:
    def __init__(self):
        pass
    
    def calculate_optimal_weights(self, predictions_list: List[np.ndarray], y_true: np.ndarray) -> np.ndarray:
        """
        최적의 가중치 계산 (최소 제곱 오차 최소화)
        
        Args:
            predictions_list: 각 모델의 예측값 리스트
            y_true: 실제값
            
        Returns:
            최적 가중치 배열
        """
        logger.info("최적 가중치 계산 중...")
        
        n_models = len(predictions_list)
        
        # 초기 가중치 (균등 분배)
        initial_weights = np.ones(n_models) / n_models
        
        # 제약 조건: 가중치 합 = 1, 모든 가중치 >= 0
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        # 목적 함수: 가중 평균 예측의 MSE
        def objective(weights):
            weighted_pred = np.zeros_like(y_true, dtype=float)
            for i, pred in enumerate(predictions_list):
                weighted_pred += weights[i] * pred
            return np.mean((weighted_pred - y_true) ** 2)
        
        # 최적화 수행
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result['x']
        logger.info(f"최적 가중치 계산 완료: {optimal_weights}")
        
        return optimal_weights
    
    def calculate_weights_by_performance(self, metrics_list: List[Dict[str, float]], 
                                        metric_name: str = 'RMSE') -> np.ndarray:
        """
        성능 지표 기반 가중치 계산
        
        Args:
            metrics_list: 각 모델의 성능 지표 딕셔너리 리스트
            metric_name: 사용할 성능 지표 이름 (낮을수록 좋은 지표여야 함)
            
        Returns:
            성능 기반 가중치 배열
        """
        logger.info(f"{metric_name} 기반 가중치 계산 중...")
        
        # 지표값 추출
        metric_values = np.array([metrics[metric_name] for metrics in metrics_list])
        
        # 지표가 낮을수록 좋은 경우 (MAE, RMSE 등) 역수 취함
        inverse_metrics = 1 / metric_values
        
        # 가중치 정규화
        weights = inverse_metrics / np.sum(inverse_metrics)
        
        logger.info(f"{metric_name} 기반 가중치 계산 완료: {weights}")
        return weights
    
    def calculate_weights_by_cv(self, models: List[Any], X: pd.DataFrame, y: pd.Series, 
                              cv_folds: int = 5) -> np.ndarray:
        """
        교차 검증 기반 가중치 계산
        
        Args:
            models: 학습된 모델 리스트
            X: 특성 데이터
            y: 타겟 데이터
            cv_folds: 교차 검증 폴드 수
            
        Returns:
            교차 검증 기반 가중치 배열
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        logger.info("교차 검증 기반 가중치 계산 중...")
        
        # 시계열 교차 검증 분할
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # 각 모델의 평균 RMSE 저장
        model_rmse = []
        
        for model in models:
            fold_rmse = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 모델 재학습 (또는 이미 학습된 모델 사용)
                if hasattr(model, 'train'):
                    model.train(X_train, y_train)
                    y_pred = model.predict(X_val)
                else:
                    y_pred = model.predict(X_val)
                
                # RMSE 계산
                rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                fold_rmse.append(rmse)
            
            # 평균 RMSE 저장
            model_rmse.append(np.mean(fold_rmse))
        
        # RMSE 역수로 가중치 계산
        inverse_rmse = 1 / np.array(model_rmse)
        weights = inverse_rmse / np.sum(inverse_rmse)
        
        logger.info(f"교차 검증 기반 가중치 계산 완료: {weights}")
        return weights

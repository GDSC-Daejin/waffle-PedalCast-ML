"""
모델 평가 모듈: 모델 성능 평가 및 시각화
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List, Tuple, Optional
import logging
import os

from config import RESULT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        os.makedirs(RESULT_DIR, exist_ok=True)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """성능 지표 계산"""
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # y_true가 0이 아닌 경우만
        }
        
        logger.info(f"모델 성능 지표: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}, MAPE={metrics['MAPE']:.2f}%")
        return metrics
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, dates: pd.Series, 
                         title: str = 'Actual vs Predicted', save_path: Optional[str] = None) -> None:
        """실제값과 예측값 시각화"""
        # None 또는 길이 0 체크
        if dates is None or y_true is None or y_pred is None:
            logger.warning("plot_predictions: 입력 데이터가 None입니다. 그래프를 그리지 않습니다.")
            return
        if len(dates) == 0 or len(y_true) == 0 or len(y_pred) == 0:
            logger.warning("plot_predictions: 입력 데이터 길이가 0입니다. 그래프를 그리지 않습니다.")
            return

        # 결측치가 있는 행 제거 (동일 인덱스 기준)
        plot_df = pd.DataFrame({'dates': dates, 'y_true': y_true, 'y_pred': y_pred})
        plot_df = plot_df.dropna()
        if plot_df.empty:
            logger.warning("plot_predictions: 결측치 제거 후 데이터가 없습니다. 그래프를 그리지 않습니다.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df['dates'], plot_df['y_true'], label='Actual', marker='o', alpha=0.7)
        plt.plot(plot_df['dates'], plot_df['y_pred'], label='Predicted', marker='x', alpha=0.7)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Usage Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"그래프 저장 완료: {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       title: str = 'Residual Plot', save_path: Optional[str] = None) -> None:
        """잔차 시각화"""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 6))
        
        # 잔차 산점도
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residuals vs Predicted')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        
        # 잔차 히스토그램
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.title('Residuals Distribution')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"그래프 저장 완료: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                               title: str = 'Feature Importance', 
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
        """특성 중요도 시각화"""
        if feature_importance is None:
            logger.warning("특성 중요도 데이터가 없습니다.")
            return
        
        # 상위 N개 특성만 선택
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"그래프 저장 완료: {save_path}")
        
        plt.show()
    
    def plot_seasonal_decomposition(self, df: pd.DataFrame, date_col: str, target_col: str,
                                   save_path: Optional[str] = None) -> None:
        """시계열 분해 시각화"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # 인덱스를 날짜로 설정
        ts_data = df.set_index(date_col)[target_col]
        
        # 시계열 분해
        try:
            result = seasonal_decompose(ts_data, model='multiplicative', period=30)  # 월별 주기
        except:
            result = seasonal_decompose(ts_data, model='additive', period=30)  # 대안으로 가법 모델
        
        # 결과 시각화
        plt.figure(figsize=(14, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(result.observed)
        plt.title('Observed')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 2)
        plt.plot(result.trend)
        plt.title('Trend')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 3)
        plt.plot(result.seasonal)
        plt.title('Seasonality')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 4)
        plt.plot(result.resid)
        plt.title('Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"그래프 저장 완료: {save_path}")
        
        plt.show()
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      dates_test: pd.Series, model_name: str = 'Model') -> Dict[str, float]:
        """모델 평가 및 시각화 수행"""
        # 예측 수행
        y_pred = model.predict(X_test)
        
        # 성능 지표 계산
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # 결과 시각화
        self.plot_predictions(
            y_test, y_pred, dates_test, 
            title=f'{model_name} - Actual vs Predicted',
            save_path=os.path.join(RESULT_DIR, f'{model_name}_predictions.png')
        )
        
        self.plot_residuals(
            y_test, y_pred,
            title=f'{model_name} - Residual Analysis',
            save_path=os.path.join(RESULT_DIR, f'{model_name}_residuals.png')
        )
        
        # 특성 중요도 시각화 (가능한 경우)
        if hasattr(model, 'get_feature_importance') and model.get_feature_importance() is not None:
            self.plot_feature_importance(
                model.get_feature_importance(),
                title=f'{model_name} - Feature Importance',
                save_path=os.path.join(RESULT_DIR, f'{model_name}_feature_importance.png')
            )
        
        return metrics
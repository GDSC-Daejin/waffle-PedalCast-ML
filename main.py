import os
from src.data_loader import load_all_data
from src.preprocessor import preprocess
from src.feature_engineering import add_time_features, add_lag_rolling_features, encode_categorical
from src.modeler import train_all_stations
from src.predictor import make_future_table, fill_future_features, predict_for_station, get_redistribution_suggestion

# 데이터 로드 및 전처리
data_dir = './data'
df = load_all_data(data_dir)
df = preprocess(df)
df = add_time_features(df)
df = add_lag_rolling_features(df)
df = encode_categorical(df)
df = df.dropna()  # lag/rolling 등 후 결측치
print("After preprocessing:", df.shape)  # 추가

# 피처 정의
feature_cols = [c for c in df.columns if c not in ['대여일자','대여소명','이용건수']]

# 대여소별 모델 학습
station_models = train_all_stations(df, feature_cols)
print("Number of station models:", len(station_models))  # 추가

# 예측, 권장
from tqdm import tqdm
all_preds = {}
future_days = 365
for station, model in tqdm(station_models.items(), desc="Processing stations"):
    station_df = df[df['대여소번호'] == station]
    last_date = station_df['대여일자'].max()
    future_df = make_future_table(station, last_date, future_days)
    future_df = fill_future_features(future_df, station_df)
    # future_df = encode_categorical(future_df)  # 이미 컬럼 형태 맞춤
    future_df = future_df.reindex(columns=feature_cols, fill_value=0)
    future_df = predict_for_station(model, future_df, feature_cols)
    all_preds[station] = future_df

print("Number of prediction results:", len(all_preds))  # 추가

# 재배치 권장 결과 저장
suggestion_df = get_redistribution_suggestion(all_preds, threshold=0.05)
suggestion_df.to_csv('./redistribution_suggestion.csv', index=False)

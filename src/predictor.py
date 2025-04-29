import pandas as pd

def make_future_table(station_id, last_date, periods=365):
    future = pd.DataFrame({'대여일자': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')})
    future['대여소번호'] = station_id
    # 아래 값은 미래이기 때문에 평균값 등으로 대체
    return future

def fill_future_features(future_df, ref_df):
    # 시간 파생
    future_df = future_df.copy()
    future_df['월'] = future_df['대여일자'].dt.month
    future_df['요일'] = future_df['대여일자'].dt.dayofweek
    future_df['주말'] = (future_df['요일'] >= 5).astype(int)
    future_df['분기'] = future_df['대여일자'].dt.quarter
    # 범주형은 대여소별 최빈값
    for col in ['대여구분_정기', '성별_남성', '성별_여성', '성별_미상', '연령대_10대', '연령대_20대', '연령대_30대', '연령대_40대', '연령대_50대', '연령대_60대이상', '연령대_미상']:
        if col in ref_df.columns:
            mode = ref_df[col].mode()[0]
            future_df[col] = mode
    # lag/rolling 등은 마지막값으로
    for col in ['lag_1','lag_2','lag_3','rolling_mean_3','rolling_mean_6']:
        if col in ref_df.columns:
            last_val = ref_df.iloc[-1][col]
            future_df[col] = last_val
    return future_df

def predict_for_station(model, future_df, features):
    X = future_df[features]
    future_df['예측_이용건수'] = model.predict(X)
    return future_df

def get_redistribution_suggestion(all_preds, threshold=0.2):
    results = []
    for station, pred_df in all_preds.items():
        mean_usage = pred_df['예측_이용건수'].mean()
        pred_df['평균대비증가율'] = (pred_df['예측_이용건수'] - mean_usage) / mean_usage
        pred_df['자전거추가필요'] = pred_df['평균대비증가율'] > threshold
        results.append(pred_df[pred_df['자전거추가필요']])
    if results:
        return pd.concat(results)
    else:
        return pd.DataFrame()

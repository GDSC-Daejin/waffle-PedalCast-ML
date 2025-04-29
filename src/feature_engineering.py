import pandas as pd

def add_time_features(df):
    df['월'] = df['대여일자'].dt.month
    df['요일'] = df['대여일자'].dt.dayofweek
    df['주말'] = (df['요일'] >= 5).astype(int)
    df['분기'] = df['대여일자'].dt.quarter
    return df

def add_lag_rolling_features(df):
    df = df.sort_values(['대여소번호', '대여일자'])
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df.groupby('대여소번호')['이용건수'].shift(lag)
    df['rolling_mean_3'] = df.groupby('대여소번호')['이용건수'].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    df['rolling_mean_6'] = df.groupby('대여소번호')['이용건수'].transform(
        lambda x: x.rolling(6, min_periods=1).mean())
    return df

def encode_categorical(df):
    categorical_columns = ['대여구분', '성별', '연령대']
    existing_columns = [col for col in categorical_columns if col in df.columns]
    return pd.get_dummies(df, columns=existing_columns, drop_first=True)

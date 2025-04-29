from lightgbm import LGBMRegressor

def train_station_model(train_df, features, target='이용건수'):
    X = train_df[features]
    y = train_df[target]
    model = LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_all_stations(df, features):
    station_models = {}
    for station in df['대여소번호'].unique():
        station_df = df[df['대여소번호']==station].copy()
        if len(station_df) < 20:  # 데이터 너무 적은 경우 제외
            continue
        model = train_station_model(station_df, features)
        station_models[station] = model
    return station_models

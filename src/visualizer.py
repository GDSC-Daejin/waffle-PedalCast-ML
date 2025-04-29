import matplotlib.pyplot as plt

def plot_station_forecast(pred_df, station_id):
    plt.figure(figsize=(14,6))
    plt.plot(pred_df['대여일자'], pred_df['예측_이용건수'])
    plt.title(f'Station {station_id} Predicted Usage')
    plt.xlabel("Date")
    plt.ylabel("Prediction")
    plt.tight_layout()
    plt.show()

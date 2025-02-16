import pandas as pd
from prophet import Prophet

def main():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
    df.head()
    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=365)
    future.tail()

    forecast = m.predict(future)
    forecast.head()

if __name__ == "__main__":
    main()
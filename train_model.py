import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import joblib

# Polygon API key and symbols
POLYGON_API_KEY = 'ZTppfs8VAEPg6EPEhB0_8xtbzC0mjT0m'
SYMBOL_1M = 'O:FOREX:AUDCAD'
SYMBOL_5M = 'O:FOREX:AUDCAD'

def fetch_candles(symbol, timespan='1min', limit=1000):
    url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/minute/now?adjusted=true&limit={limit}&apiKey={POLYGON_API_KEY}'
    response = requests.get(url)
    data = response.json()
    if 'results' not in data:
        raise Exception("Error fetching data")
    df = pd.DataFrame(data['results'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    return df[['t', 'o', 'h', 'l', 'c', 'v']]

def add_rsi(df, window=14):
    rsi = RSIIndicator(close=df['c'], window=window)
    df['rsi'] = rsi.rsi()
    df.fillna(method='bfill', inplace=True)
    return df

def prepare_features(df_1m, df_5m):
    df_1m = add_rsi(df_1m)
    df_5m = add_rsi(df_5m)
    df_1m.set_index('t', inplace=True)
    df_5m.set_index('t', inplace=True)
    df_5m_1m = df_5m.resample('1min').ffill().reindex(df_1m.index)
    features = pd.concat([
        df_1m[['o','h','l','c','v','rsi']].add_suffix('_1m'),
        df_5m_1m[['o','h','l','c','v','rsi']].add_suffix('_5m')
    ], axis=1).fillna(method='ffill').fillna(method='bfill')
    return features

def create_labels(df):
    df['target'] = (df['c_1m'].shift(-1) > df['c_1m']).astype(int)
    df.dropna(inplace=True)
    return df

def build_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    return model

def main():
    print("Fetching candle data...")
    df_1m = fetch_candles(SYMBOL_1M, '1min', 1000)
    df_5m = fetch_candles(SYMBOL_5M, '5min', 1000)

    print("Preparing features...")
    features = prepare_features(df_1m, df_5m)
    features = create_labels(features)

    X = features.drop(columns=['target'])
    y = features['target']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    time_steps = 50
    n_features = X.shape[1]

    X_lstm, y_lstm = [], []
    for i in range(len(X_scaled) - time_steps):
        X_lstm.append(X_scaled[i:i+time_steps])
        y_lstm.append(y.iloc[i+time_steps])

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)

    X_train, X_val, y_train, y_val = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

    model = build_model((time_steps, n_features))
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    print("Training model...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64, callbacks=[checkpoint])

    print("Saving scaler...")
    joblib.dump(scaler, 'scaler.pkl')

    print("Training complete and model saved as best_model.h5")

if __name__ == "__main__":
    main()

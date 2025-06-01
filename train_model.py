import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from joblib import dump
import requests

# === CONFIG ===
API_KEY = "ZTppfs8VAEPg6EPEhB0_8xtbzC0mjT0m"
symbol = "C:AUDCAD"
limit = 500

def fetch_data():
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/2024-05-01/2024-05-15?adjusted=true&sort=asc&limit={limit}&apiKey={API_KEY}"
    res = requests.get(url).json()
    candles = res['results']
    df = pd.DataFrame(candles)
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
    return df

def add_indicators(df):
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['ema'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df.dropna(inplace=True)
    return df

def train_model(df):
    X = df[['close', 'rsi', 'ema']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.2f}")
    dump(model, 'model.joblib')

df = fetch_data()
df = add_indicators(df)
train_model(df)

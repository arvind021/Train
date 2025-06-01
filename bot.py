import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, CallbackContext
import logging

# --- CONFIG ---
POLYGON_API_KEY = 'ZTppfs8VAEPg6EPEhB0_8xtbzC0mjT0m'
TELEGRAM_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN_HERE'  # Replace with your Telegram Bot Token
SYMBOL_1M = 'O:FOREX:AUDCAD'
SYMBOL_5M = 'O:FOREX:AUDCAD'

TIME_STEPS = 50

# --- SETUP LOGGING ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Load the model and scaler
model = load_model('best_model.h5')
scaler = joblib.load('scaler.pkl')

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
    from ta.momentum import RSIIndicator
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

def predict_candle_direction():
    # Fetch latest candle data
    df_1m = fetch_candles(SYMBOL_1M, '1min', limit=TIME_STEPS + 10)
    df_5m = fetch_candles(SYMBOL_5M, '5min', limit=TIME_STEPS//5 + 10)

    features = prepare_features(df_1m, df_5m)
    # We don't need labels here
    X = features.dropna()
    
    # Take last TIME_STEPS rows for prediction input
    if len(X) < TIME_STEPS:
        return "Not enough data to predict."

    X_input = X.iloc[-TIME_STEPS:]
    X_scaled = scaler.transform(X_input)
    X_scaled = np.expand_dims(X_scaled, axis=0)  # (1, TIME_STEPS, n_features)

    pred = model.predict(X_scaled)[0][0]
    if pred > 0.5:
        return "Prediction: Next 1-min candle likely UP 📈"
    else:
        return "Prediction: Next 1-min candle likely DOWN 📉"

# Telegram command handler
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello! Send /predict to get next 1-min candle prediction for AUDCAD.")

def predict(update: Update, context: CallbackContext):
    update.message.reply_text("Fetching data and predicting, please wait...")
    try:
        result = predict_candle_direction()
    except Exception as e:
        result = f"Error during prediction: {str(e)}"
    update.message.reply_text(result)

def main():
    updater = Updater(TELEGRAM_TOKEN)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", predict))

    print("Bot started. Waiting for commands...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import pandas as pd
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from joblib import load
import datetime

API_KEY = "ZTppfs8VAEPg6EPEhB0_8xtbzC0mjT0m"
symbol = "C:AUDCAD"
model = load("model.joblib")

def get_latest_data():
    now = datetime.datetime.utcnow()
    start = (now - datetime.timedelta(minutes=60)).strftime('%Y-%m-%d')
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start}/{start}?adjusted=true&sort=desc&limit=50&apiKey={API_KEY}"
    data = requests.get(url).json()
    df = pd.DataFrame(data['results'])
    df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['ema'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df.dropna(inplace=True)
    latest = df.iloc[-1]
    return [[latest['close'], latest['rsi'], latest['ema']]]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome! Send /predict to get 1-min candle direction prediction.")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    X = get_latest_data()
    pred = model.predict(X)[0]
    msg = "📈 UP (BUY)" if pred == 1 else "📉 DOWN (SELL)"
    await update.message.reply_text(f"Prediction: {msg}")

app = ApplicationBuilder().token("8030718150:AAFp5QuwaC-103ruvB5TsBMGY5MwMvkq-5g").build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("predict", predict))

app.run_polling()

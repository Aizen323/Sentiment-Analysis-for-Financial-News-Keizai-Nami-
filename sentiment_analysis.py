import requests
import pandas as pd
import yfinance as yf
import os
import nltk
import pandas_market_calendars as mcal
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from prophet import Prophet
from langdetect import detect, DetectorFactory

# Download necessary NLTK data
nltk.download("vader_lexicon")

# Initialize sentiment analyzers
sia = SentimentIntensityAnalyzer()
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Load API key
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "9a979c087752412e8908ba50f804682a")

# Ensure consistent language detection
DetectorFactory.seed = 0  

def is_english(text):
    """Check if the given text is in English."""
    try:
        return detect(text) == "en"
    except:
        return False  # If detection fails, assume it's not English

def fetch_financial_news(ticker="GOOGL", api_key=NEWS_API_KEY):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])

        filtered_articles = [
            {
                "Date": pd.to_datetime(article.get("publishedAt", "")).date(),
                "Title": article.get("title", "No Data"),
                "Description": article.get("description", "No Data"),
            }
            for article in articles
            if article.get("publishedAt")
            and is_english(article.get("title", ""))
            and is_english(article.get("description", ""))
        ]

        return pd.DataFrame(filtered_articles)
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return pd.DataFrame(columns=["Date", "Title", "Description"])

def analyze_sentiment(text):
    if not text or not isinstance(text, str):
        return {"vader": "Neutral", "finbert": "Neutral"}
    
    text = text.encode("utf-8", "ignore").decode("utf-8")
    vader_score = sia.polarity_scores(text)["compound"]
    vader_sentiment = "Positive" if vader_score > 0.2 else "Negative" if vader_score < -0.2 else "Neutral"
    
    finbert_result = finbert(text)
    finbert_label = finbert_result[0]["label"] if finbert_result[0]["score"] > 0.6 else "Neutral"
    
    return {"vader": vader_sentiment, "finbert": finbert_label}

def analyze_historical_sentiment(ticker="GOOGL"):
    news_df = fetch_financial_news(ticker)
    if news_df.empty:
        return pd.DataFrame(columns=["Date", "Title", "Description", "VADER Sentiment", "FinBERT Sentiment"])
    
    news_df[["VADER Sentiment", "FinBERT Sentiment"]] = news_df["Title"].apply(
        lambda x: pd.Series(analyze_sentiment(x)) if pd.notna(x) else pd.Series({"vader": "Neutral", "finbert": "Neutral"})
    )
    return news_df

def fetch_stock_history(ticker="GOOGL", period="2y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    
    hist.reset_index(inplace=True)
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
    return hist[["Date", "Open", "High", "Low", "Close", "Volume"]]

def predict_stock_prices(ticker="GOOGL", future_days=30):
    stock_df = yf.download(ticker, period="2y")
    if stock_df.empty:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    
    df = stock_df.reset_index()[["Date", "Close"]].dropna()
    df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])
    
    # Filter trading days
    nyse = mcal.get_calendar("NYSE")
    trading_days = nyse.valid_days(start_date=df["ds"].min(), end_date=df["ds"].max())
    df = df[df["ds"].isin(pd.to_datetime(trading_days))]
    
    # Train Prophet model
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(df)
    
    # Generate future dates only on business days
    future_dates = pd.bdate_range(start=df["ds"].max(), periods=future_days, freq="B")
    future = pd.DataFrame({"ds": future_dates})
    
    # Predict and smooth results
    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].rolling(window=3, min_periods=1).mean()
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def merge_stock_sentiment_data(ticker="GOOGL"):
    stock_df = fetch_stock_history(ticker, period="2y")
    sentiment_df = analyze_historical_sentiment(ticker)
    if stock_df.empty:
        return stock_df
    
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])
    
    merged_df = stock_df.merge(sentiment_df, on="Date", how="left").fillna("No Data")
    return merged_df

if __name__ == "__main__":
    ticker = "GOOGL"
    print("\n🔹 Fetching financial news for", ticker)
    print(fetch_financial_news(ticker).head())
    print("\n🔹 Analyzing sentiment...")
    print(analyze_historical_sentiment(ticker).head())
    print("\n🔹 Fetching stock history for", ticker)
    print(fetch_stock_history(ticker).head())
    print("\n🔹 Predicting future stock prices...")
    print(predict_stock_prices(ticker).head())
    print("\n🔹 Merging stock and sentiment data...")
    print(merge_stock_sentiment_data(ticker).head())
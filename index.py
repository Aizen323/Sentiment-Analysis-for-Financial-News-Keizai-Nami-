from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pandas_market_calendars as mcal
import plotly.express as px
from sentiment_analysis import (
    fetch_financial_news,
    analyze_sentiment,
    fetch_stock_history,
    analyze_historical_sentiment
)
from prophet import Prophet

app = Flask(__name__)
CORS(app)
CORS(app, origins=["http://localhost:3000"])
# Route to fetch stock history
def get_stock_data(ticker):
    return fetch_stock_history(ticker)

# Route to fetch sentiment analysis
def get_sentiment_data(ticker):
    return analyze_historical_sentiment(ticker)

@app.route('/api/trends', methods=['GET'])
def stock_trends():
    ticker = request.args.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({'error': 'Stock ticker is required'}), 400
    
    stock_data = get_stock_data(ticker)
    sentiment_data = get_sentiment_data(ticker)
    
    if stock_data is None or stock_data.empty:
        return jsonify({'error': 'No stock price data found'}), 404
    
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date']).dt.date if sentiment_data is not None else None
    
    merged_data = pd.merge(stock_data, sentiment_data, on='Date', how='left') if sentiment_data is not None else stock_data
    merged_data.fillna({'VADER Sentiment': 'Neutral', 'FinBERT Sentiment': 'Neutral'}, inplace=True)
    
    return jsonify(merged_data.to_dict(orient='records'))

@app.route('/api/predict', methods=['GET'])
def future_prediction():
    ticker = request.args.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({'error': 'Stock ticker is required'}), 400
    
    stock_data = get_stock_data(ticker)
    if stock_data is None or stock_data.empty:
        return jsonify({'error': 'No stock price data found'}), 404
    
    stock_data['ds'] = pd.to_datetime(stock_data['Date'])
    stock_data['y'] = stock_data['Close']
    
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(stock_data[['ds', 'y']])
    
    future = model.make_future_dataframe(periods=30, freq='B')
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].rolling(window=3, min_periods=1).mean()
    
    today = pd.to_datetime('today').date()
    future_predictions = forecast[forecast['ds'].dt.date >= today][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    return jsonify(future_predictions.rename(columns={'ds': 'Date', 'yhat': 'Predicted Price', 'yhat_lower': 'Lower Estimate', 'yhat_upper': 'Upper Estimate'}).to_dict(orient='records'))

@app.route('/api/sentiment', methods=['GET'])
def sentiment_analysis():
    ticker = request.args.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({'error': 'Stock ticker is required'}), 400
    
    news_data = fetch_financial_news(ticker)
    if news_data is None or news_data.empty:
        return jsonify({'error': 'No news found'}), 404
    
    news_data['Sentiment Analysis'] = news_data['Title'].apply(analyze_sentiment)
    news_data['VADER Sentiment'] = news_data['Sentiment Analysis'].apply(lambda x: x.get('vader', 'Neutral'))
    news_data['FinBERT Sentiment'] = news_data['Sentiment Analysis'].apply(lambda x: x.get('finbert', 'Neutral'))
    
    return jsonify(news_data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

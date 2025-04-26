import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import yfinance as yf
from sentiment_analysis import (
    fetch_financial_news,
    analyze_sentiment,
    fetch_stock_history,
    analyze_historical_sentiment
)

st.set_page_config(page_title="Stock Sentiment Analysis", layout="wide", page_icon="📊")

# 🔥 Apply Dark Theme
def load_css():
    with open("style.css", "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Apply CSS
load_css()

# 📊 App Title
st.title("📊 Keizai Nami - Economic Waves")

# ✅ Sidebar - User Input
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., GOOGL)", value="GOOGL", key="ticker_input").strip().upper()

# Fetch Stock Data
if ticker:
    stock = yf.Ticker(ticker)
    stock_info = stock.info

    # 1️⃣ Stock Overview Summary
    st.sidebar.markdown("""
    <style>
        .metric-title { font-size: 18px; font-weight: bold; color: white; margin-bottom: 2px; }
        .metric-value { font-size: 22px; font-weight: bold; color: #00ff99; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

    st.sidebar.markdown("<p class='metric-title'>Price</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p class='metric-value'>${stock_info.get('currentPrice', 'N/A')}</p>", unsafe_allow_html=True)

    st.sidebar.markdown("<p class='metric-title'>Market Cap</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p class='metric-value'>{stock_info.get('marketCap', 'N/A'):,}</p>", unsafe_allow_html=True)

    st.sidebar.markdown("<p class='metric-title'>Volume</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p class='metric-value'>{stock_info.get('volume', 'N/A'):,}</p>", unsafe_allow_html=True)

    # 5️⃣ Quick Links to Financial Sites
    google_finance_url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
    st.sidebar.markdown(f"[View {ticker} on Google Finance]({google_finance_url})")

# ✅ Cached Function for API Calls
@st.cache_data(ttl=600)
def get_stock_data(ticker):
    stock_data = fetch_stock_history(ticker)
    if stock_data is not None and not stock_data.empty:
        stock_data["Date"] = pd.to_datetime(stock_data["Date"])
        stock_data = stock_data[stock_data["Date"] <= pd.to_datetime("today")]
        stock_data["Date"] = stock_data["Date"].dt.date
    return stock_data

@st.cache_data(ttl=600)
def get_sentiment_data(ticker):
    return analyze_historical_sentiment(ticker)

# 📊 Tabs for Better Organization
tab1, tab2, tab3 = st.tabs(["📊 Trends", "🔮 Predictions", "📰 Sentiment"])

# 🌈 Historical Sentiment and Stock Trends Section
with tab1:
    show_trends = st.checkbox("Show Trends", key="show_trends", value=False)

    if show_trends:
        if not ticker:
            st.error("❌ Please enter a valid stock ticker.")
        else:
            st.write(f"💽 Fetching stock and sentiment data for: **{ticker}**")

            # ✅ GIF Loader in Sidebar while Fetching Data
            loading_gif = """
                <div style="text-align: center;">
                    <img src="https://media.giphy.com/media/w4zX4PfkQmZ4xeG7ny/giphy.gif?cid=ecf05e472yxvfpayf2i69q4d3x95xjhdmrre1zivf9ty38vp&ep=v1_gifs_related&rid=giphy.gif&ct=g" width="100">
                    <p style="color: white; font-weight: bold; font-size: 16px;"></p>
                </div>
            """
            st.sidebar.markdown(loading_gif, unsafe_allow_html=True)

            with st.spinner("Fetching stock data..."):
                stock_data = get_stock_data(ticker)
                sentiment_data = get_sentiment_data(ticker)

            # ✅ Loading GIF  
            time.sleep(2)  # Simulating loading time
            st.sidebar.success("✅ Sentiment data loaded!")

            # ✅ Display Stock Price Data
            if stock_data is None or stock_data.empty:
                st.error("⚠️ No stock price data found! Try another stock ticker.")
            else:
                stock_data["Date"] = pd.to_datetime(stock_data["Date"])
                stock_data = stock_data.sort_values(by="Date", ascending=False)
                stock_data["Date"] = stock_data["Date"].dt.date
                st.subheader("📊 Stock Price Data (Recent First)")
                st.dataframe(stock_data.head())

                # ✅ Full Stock Price Trend Graph
                fig_stock = go.Figure()
                fig_stock.add_trace(go.Scatter(
                    x=stock_data["Date"], y=stock_data["Close"], mode='lines+markers',
                    name=f"{ticker} Stock Price", line=dict(width=2, color='#00C8FF'),
                    marker=dict(size=6, symbol='circle', color='#FF4500')
                ))
                fig_stock.update_layout(
                    title="📈 Stock Price Trend",
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Stock Price",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_stock, use_container_width=True)

                # ✅ **7-Day Stock Graph with Multiple Views**
                st.subheader("📅 Recent 7-Day Stock Price Analysis")

                stock_7d = stock_data.head(7)  # Take recent 7 days
                graph_type = st.selectbox("Select Chart Type", ["Candlestick", "Bar", "Area", "Scatter"], key="chart_type")

                fig_7d = go.Figure()
                
                if graph_type == "Candlestick":
                    fig_7d.add_trace(go.Candlestick(
                        x=stock_7d["Date"], open=stock_7d["Open"], high=stock_7d["High"],
                        low=stock_7d["Low"], close=stock_7d["Close"], name="Candlestick"
                    ))
                elif graph_type == "Bar":
                    fig_7d.add_trace(go.Bar(
                        x=stock_7d["Date"], y=stock_7d["Close"], name="Bar Chart", marker_color="#FF4500"
                    ))
                elif graph_type == "Area":
                    fig_7d.add_trace(go.Scatter(
                        x=stock_7d["Date"], y=stock_7d["Close"], fill='tozeroy', mode='lines',
                        name="Area Chart", line=dict(color="#FFD700", width=2)
                    ))
                elif graph_type == "Scatter":
                    fig_7d.add_trace(go.Scatter(
                        x=stock_7d["Date"], y=stock_7d["Close"], mode='markers+lines',
                        name="Scatter Plot", marker=dict(size=8, color="#00C8FF")
                    ))

                fig_7d.update_layout(
                    title="📊 Stock Price - Last 7 Days",
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117"
                )
                st.plotly_chart(fig_7d, use_container_width=True)

            # ✅ Display Sentiment Analysis Results
            if sentiment_data is None or sentiment_data.empty:
                st.warning("⚠️ No sentiment data found!")
            else:
                sentiment_data["Date"] = pd.to_datetime(sentiment_data["Date"]).dt.date
                st.subheader("📰 Sentiment Analysis Results")
                st.dataframe(sentiment_data.head())

                # ✅ Sentiment Trend Graph
                sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
                sentiment_data["VADER Sentiment Score"] = sentiment_data["VADER Sentiment"].map(sentiment_map).fillna(0)
                sentiment_data["Smoothed Sentiment"] = sentiment_data["VADER Sentiment Score"].rolling(window=3, min_periods=1).mean()
                
                fig_sentiment = go.Figure()
                fig_sentiment.add_trace(go.Scatter(
                    x=sentiment_data["Date"], y=sentiment_data["Smoothed Sentiment"], mode='lines+markers',
                    name="Sentiment Trend", line=dict(width=2, color='#FFD700'),
                    marker=dict(size=6, symbol='diamond', color='#FF4500')
                ))
                fig_sentiment.update_layout(
                    title="📉 Sentiment Trend Over Time",
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Sentiment Score",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117"
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)


# 📈 Future Prediction Section
with tab2:
    st.subheader("🔮 Future Stock Price Predictions")

    # User can select prediction period (7 to 30 days)
    forecast_days = st.slider("Select prediction period (days)", 7, 30, 14)

    if st.button("Generate Prediction"):
        if not ticker:
            st.error("❌ Please enter a valid stock ticker.")
        else:
            st.markdown(f"📡 **Predicting future prices for:** `{ticker}`", unsafe_allow_html=True)

            with st.spinner("Fetching stock data..."):
                stock_data = get_stock_data(ticker)

            if stock_data is None or stock_data.empty:
                st.error("⚠️ **No stock price data found!** Try another stock ticker.")
            elif len(stock_data) < 10:
                st.error("⚠️ **Not enough data to make a reliable prediction!**")
            else:
                stock_data["ds"] = pd.to_datetime(stock_data["Date"])
                stock_data["y"] = stock_data["Close"]

                # 📊 Train Prophet Model
                model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                model.fit(stock_data[["ds", "y"]])

                # Generate future dates
                future = model.make_future_dataframe(periods=forecast_days, freq="B")
                forecast = model.predict(future)
                forecast["yhat"] = forecast["yhat"].rolling(window=3, min_periods=1).mean()

                # ✅ Accuracy Calculation (MAPE-based)
                merged = stock_data.merge(forecast, on="ds", how="inner")
                accuracy = None
                if len(merged) > 5:
                    mape = mean_absolute_percentage_error(merged["y"], merged["yhat"]) * 100
                    accuracy = max(0, 100 - mape)

                # 🔹 Show only predictions from today onwards
                today = pd.to_datetime("today").date()
                future_predictions = forecast[forecast["ds"].dt.date >= today][["ds", "yhat", "yhat_lower", "yhat_upper"]]

                # 📊 Future Prediction Graph
                fig_future = go.Figure()

                # 🔵 Predicted trend line
                fig_future.add_trace(
                    go.Scatter(
                        x=future_predictions["ds"],
                        y=future_predictions["yhat"],
                        mode="lines+markers",
                        name="Predicted Price",
                        line=dict(color="cyan", width=2)
                    )
                )

                # 🔷 Confidence intervals
                fig_future.add_trace(
                    go.Scatter(
                        x=future_predictions["ds"],
                        y=future_predictions["yhat_upper"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                fig_future.add_trace(
                    go.Scatter(
                        x=future_predictions["ds"],
                        y=future_predictions["yhat_lower"],
                        mode="lines",
                        fill="tonexty",
                        fillcolor="rgba(0, 191, 255, 0.2)",  # Light blue shade
                        line=dict(width=0),
                        name="Confidence Interval"
                    )
                )

                # 🟠 Actual stock price trend
                fig_future.add_trace(
                    go.Scatter(
                        x=stock_data["ds"],
                        y=stock_data["y"],
                        mode="lines",
                        name="Actual Prices",
                        line=dict(color="orange", width=2, dash="dot")
                    )
                )

                # 🔧 Update layout with better alignment and readability
                fig_future.update_layout(
                    title=dict(
                        text=f"📈 Predicted Stock Prices for {ticker}",
                        font=dict(size=18, color="white"),
                        x=0,  # Align title to the left
                        xanchor="left"
                    ),
                    xaxis_title="📅 Date",
                    yaxis_title="💲 Stock Price (USD)",
                    template="plotly_dark",
                    paper_bgcolor="#121212",
                    plot_bgcolor="#1E1E1E",
                    font=dict(color="white"),
                    xaxis=dict(range=[today, future_predictions["ds"].max()]),
                    legend=dict(
                        font=dict(color="white")  # ✅ Ensure legend is readable
                    )
                )

                st.plotly_chart(fig_future, use_container_width=True)

                # 📋 Future Prediction Table
                st.subheader("📋 **Future Stock Price Predictions**")
                st.dataframe(future_predictions.rename(columns={
                    "ds": "📅 Date",
                    "yhat": "💲 Predicted Price",
                    "yhat_lower": "🔻 Lower Estimate",
                    "yhat_upper": "🔺 Upper Estimate"
                }).head(15))  # Show first 15 predictions

                # ✅ Display Prediction Accuracy in UI
                if accuracy is not None:
                    accuracy_color = "🟢" if accuracy >= 80 else "🟡" if accuracy >= 60 else "🔴"
                    styled_text = f"<span style='font-size:24px; font-weight:bold; color:white;'>{accuracy:.2f}%</span>"
                    st.markdown(
                    f"""
                    <div style="display: flex; flex-direction: column; align-items: left;">
                        <span style="font-size:18px; font-weight:bold; color:white;">📊 Prediction Accuracy</span>
                        <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
                            <span style="font-size: 30px;">{accuracy_color}</span>
                            {styled_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
                else:
                    st.warning("⚠️ **Not enough historical data to calculate accuracy.**")
                    
# 📈 Real-Time Sentiment Analysis Section
with tab3:
    if st.button("Analyze Sentiment"):
        if not ticker:
            st.error("❌ Please enter a valid stock ticker.")
        else:
            st.write(f"📡 Fetching news for: **{ticker}**")
            with st.spinner("Fetching news articles..."):
                news_data = fetch_financial_news(ticker)

            if news_data is None or news_data.empty:
                st.error("⚠️ No news found! Try another stock ticker.")
            else:
                with st.spinner("Analyzing sentiment..."):
                    news_data["Sentiment Analysis"] = news_data["Title"].apply(analyze_sentiment)
                    news_data["VADER Sentiment"] = news_data["Sentiment Analysis"].apply(lambda x: x.get("vader", "Neutral"))

                # 📰 Display Financial News Data
                st.subheader("📰 Financial News Data")
                news_data = news_data.drop(columns=["Sentiment Analysis"], errors="ignore")
                st.dataframe(news_data, hide_index=True)  # ✅ Display Table

                # 📊 Sentiment Distribution Pie Chart
                sentiment_counts = news_data["VADER Sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]

                # Define colors based on sentiment labels
                sentiment_colors = {
                    "Positive": "green",
                    "Neutral": "blue",
                    "Negative": "red"
                }

                # Assign colors dynamically based on sentiment labels
                colors = [sentiment_colors[label] for label in sentiment_counts["Sentiment"]]

                fig_pie = go.Figure(data=[go.Pie(
                    labels=sentiment_counts["Sentiment"],
                    values=sentiment_counts["Count"],
                    hole=0.3,  # Donut-style
                    marker=dict(colors=colors),
                    textinfo="percent+label"
                )])

                fig_pie.update_layout(
                    title=dict(
                        text="📊 Sentiment Distribution",
                        font=dict(size=18, color="white")
                    ),
                    template="plotly_dark",
                    showlegend=True,
                    paper_bgcolor="#0e1117",
                    font=dict(color="white"),
                    legend=dict(
                        font=dict(color="white", size=12),  # ✅ Fix: Improved readability
                        bgcolor="rgba(0, 0, 0, 0)"  # ✅ Transparent background
                    )
                )

                st.plotly_chart(fig_pie, use_container_width=True)



            
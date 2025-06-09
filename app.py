import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests
import json
import os
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import additional libraries for prediction
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not available. Stock prediction feature will use simplified models.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Company data with popular tickers
SUPPORTED_COMPANIES = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet Inc.',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'NFLX': 'Netflix Inc.',
    'ADBE': 'Adobe Inc.',
    'CRM': 'Salesforce Inc.',
    'ORCL': 'Oracle Corporation',
    'IBM': 'International Business Machines',
    'INTC': 'Intel Corporation',
    'AMD': 'Advanced Micro Devices',
    'PYPL': 'PayPal Holdings Inc.',
    'UBER': 'Uber Technologies Inc.',
    'SPOT': 'Spotify Technology S.A.',
    'ZOOM': 'Zoom Video Communications',
    'SQ': 'Block Inc.',
    'SHOP': 'Shopify Inc.',
    'TWTR': 'Twitter Inc.',
    'SNAP': 'Snap Inc.',
    'PINS': 'Pinterest Inc.',
    'ROKU': 'Roku Inc.',
    'ZM': 'Zoom Video Communications',
    'DOCU': 'DocuSign Inc.',
    'WORK': 'Slack Technologies',
    'TEAM': 'Atlassian Corporation',
    'NOW': 'ServiceNow Inc.',
    'WDAY': 'Workday Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'BAC': 'Bank of America Corp',
    'WFC': 'Wells Fargo & Company',
    'GS': 'Goldman Sachs Group Inc.',
    'MS': 'Morgan Stanley',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Inc.',
    'DIS': 'Walt Disney Company',
    'NKE': 'Nike Inc.',
    'MCD': 'McDonald\'s Corporation',
    'SBUX': 'Starbucks Corporation',
    'KO': 'Coca-Cola Company',
    'PEP': 'PepsiCo Inc.',
    'WMT': 'Walmart Inc.',
    'TGT': 'Target Corporation',
    'HD': 'Home Depot Inc.',
    'LOW': 'Lowe\'s Companies Inc.',
    'CVS': 'CVS Health Corporation',
    'UNH': 'UnitedHealth Group Inc.',
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer Inc.',
    'MRNA': 'Moderna Inc.',
    'BNTX': 'BioNTech SE',
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    'COP': 'ConocoPhillips',
    'BA': 'Boeing Company',
    'CAT': 'Caterpillar Inc.',
    'GE': 'General Electric Company',
    'F': 'Ford Motor Company',
    'GM': 'General Motors Company'
}

# Function to get company financial data using yfinance
def get_company_financial_data(ticker):
    """Get comprehensive financial data for a company using yfinance"""
    try:
        stock = yf.Ticker(ticker)

        # Get basic info
        info = stock.info

        # Get financial statements
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow

        # Get historical data
        hist_data = stock.history(period="5y")

        return {
            'info': info,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'historical_data': hist_data,
            'success': True
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return {'success': False, 'error': str(e)}


# Function to process financial data and create revenue analysis
def process_revenue_data(financial_data, ticker):
    """Process financial data to extract revenue information"""
    try:
        financials = financial_data['financials']

        # Get revenue data (Total Revenue is usually the first row)
        if 'Total Revenue' in financials.index:
            revenue_data = financials.loc['Total Revenue'].dropna()
        elif 'Revenue' in financials.index:
            revenue_data = financials.loc['Revenue'].dropna()
        else:
            # Try to find any revenue-related row
            revenue_rows = [idx for idx in financials.index if 'revenue' in idx.lower()]
            if revenue_rows:
                revenue_data = financials.loc[revenue_rows[0]].dropna()
            else:
                return None

        # Convert to DataFrame for easier processing
        revenue_df = pd.DataFrame({
            'year': [col.year for col in revenue_data.index],
            'revenue': revenue_data.values
        })

        # Ensure revenue is numeric and handle any NaN values
        revenue_df['revenue'] = pd.to_numeric(revenue_df['revenue'], errors='coerce')
        revenue_df = revenue_df.dropna()

        # Sort by year
        revenue_df = revenue_df.sort_values('year')

        return revenue_df
    except Exception as e:
        st.error(f"Error processing revenue data: {str(e)}")
        return None

# Function to create synthetic segment data for visualization
def create_segment_data(revenue_df, ticker):
    """Create synthetic segment data based on company type"""
    if revenue_df is None or len(revenue_df) == 0:
        return None, None

    # Define segment patterns based on company type
    tech_segments = ['Software & Services', 'Hardware', 'Cloud Services', 'Other']
    retail_segments = ['Online Sales', 'Physical Stores', 'Subscription Services', 'Other']
    finance_segments = ['Investment Banking', 'Consumer Banking', 'Trading', 'Other Services']

    # Categorize companies
    tech_companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'ORCL']
    retail_companies = ['WMT', 'TGT', 'HD', 'LOW', 'SBUX', 'MCD', 'NKE']
    finance_companies = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA']

    if ticker in tech_companies:
        segments = tech_segments
        # Tech companies typically have software-heavy revenue
        segment_ratios = [0.45, 0.25, 0.20, 0.10]
    elif ticker in retail_companies:
        segments = retail_segments
        segment_ratios = [0.35, 0.40, 0.15, 0.10]
    elif ticker in finance_companies:
        segments = finance_segments
        segment_ratios = [0.30, 0.35, 0.25, 0.10]
    else:
        segments = ['Core Business', 'Secondary Services', 'International', 'Other']
        segment_ratios = [0.60, 0.20, 0.15, 0.05]

    # Create segment data
    segment_data = []
    for _, row in revenue_df.iterrows():
        year = row['year']
        total_revenue = row['revenue']

        for i, segment in enumerate(segments):
            segment_revenue = total_revenue * segment_ratios[i]
            # Add some random variation
            variation = np.random.normal(1, 0.1)
            segment_revenue *= max(0.5, variation)  # Ensure positive values

            segment_data.append({
                'year': year,
                'segment': segment,
                'revenue': segment_revenue
            })

    segment_df = pd.DataFrame(segment_data)

    # Create pivot table for visualization
    segment_pivot = segment_df.pivot(index='year', columns='segment', values='revenue')

    return segment_df, segment_pivot

# Function to create regional data
def create_regional_data(revenue_df, ticker):
    """Create synthetic regional data based on company"""
    if revenue_df is None or len(revenue_df) == 0:
        return None, None

    # Define regional patterns
    regions = ['North America', 'Europe', 'Asia Pacific', 'Other']

    # Different regional distributions for different company types
    us_heavy = ['WMT', 'TGT', 'HD', 'LOW', 'SBUX', 'MCD']  # US-focused companies
    global_tech = ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']  # Global tech companies

    if ticker in us_heavy:
        regional_ratios = [0.70, 0.15, 0.10, 0.05]  # US-heavy
    elif ticker in global_tech:
        regional_ratios = [0.45, 0.25, 0.25, 0.05]  # More global
    else:
        regional_ratios = [0.55, 0.20, 0.20, 0.05]  # Balanced

    # Create regional data
    regional_data = []
    for _, row in revenue_df.iterrows():
        year = row['year']
        total_revenue = row['revenue']

        for i, region in enumerate(regions):
            regional_revenue = total_revenue * regional_ratios[i]
            # Add some random variation
            variation = np.random.normal(1, 0.1)
            regional_revenue *= max(0.5, variation)

            regional_data.append({
                'year': year,
                'region': region,
                'revenue': regional_revenue
            })

    regional_df = pd.DataFrame(regional_data)
    regional_pivot = regional_df.pivot(index='year', columns='region', values='revenue')

    return regional_df, regional_pivot


# Function to plot revenue by product segments
def plot_revenue_by_product(segment_pivot, company_name):
    """Plot revenue breakdown by product segments"""
    if segment_pivot is None or segment_pivot.empty:
        st.warning("No segment data available for visualization")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create stacked bar chart
    segment_pivot.plot(kind="bar", stacked=True, ax=ax, colormap='Set3')

    plt.xticks(rotation=45)
    ax.set_title(f"{company_name} - Revenue by Product Segment", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Revenue (USD)", fontsize=12)
    ax.legend(title="Product Segments", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Format y-axis to show billions
    formatter = mticker.FuncFormatter(lambda x, pos: f"${x/1e9:.1f}B")
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    return fig

# Function to plot revenue by region
def plot_revenue_by_region(regional_pivot, company_name):
    """Plot revenue breakdown by geographical regions"""
    if regional_pivot is None or regional_pivot.empty:
        st.warning("No regional data available for visualization")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create stacked bar chart
    regional_pivot.plot(kind="bar", stacked=True, ax=ax, colormap='viridis')

    plt.xticks(rotation=45)
    ax.set_title(f"{company_name} - Revenue by Geographic Region", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Revenue (USD)", fontsize=12)
    ax.legend(title="Regions", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Format y-axis to show billions
    formatter = mticker.FuncFormatter(lambda x, pos: f"${x/1e9:.1f}B")
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    return fig

# Function to plot overall revenue trend
def plot_revenue_trend(revenue_df, company_name):
    """Plot overall revenue trend over time"""
    if revenue_df is None or revenue_df.empty:
        st.warning("No revenue data available for visualization")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(revenue_df['year'], revenue_df['revenue'], marker='o', linewidth=3, markersize=8)
    ax.fill_between(revenue_df['year'], revenue_df['revenue'], alpha=0.3)

    ax.set_title(f"{company_name} - Total Revenue Trend", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Revenue (USD)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Format y-axis to show billions
    formatter = mticker.FuncFormatter(lambda x, pos: f"${x/1e9:.1f}B")
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    return fig


# Function to get company summary using yfinance data
def get_company_summary(ticker, company_data):
    """Generate company summary from yfinance data"""
    try:
        info = company_data['info']

        summary = f"""
## {info.get('longName', ticker)} ({ticker})

**Company Overview:**
- **Sector:** {info.get('sector', 'N/A')}
- **Industry:** {info.get('industry', 'N/A')}
- **Country:** {info.get('country', 'N/A')}
- **Website:** {info.get('website', 'N/A')}

**Financial Highlights:**
- **Market Cap:** ${info.get('marketCap', 0):,.0f}
- **Enterprise Value:** ${info.get('enterpriseValue', 0):,.0f}
- **Revenue (TTM):** ${info.get('totalRevenue', 0):,.0f}
- **Profit Margin:** {info.get('profitMargins', 0):.2%}

**Key Metrics:**
- **P/E Ratio:** {info.get('trailingPE', 'N/A')}
- **Price to Book:** {info.get('priceToBook', 'N/A')}
- **Debt to Equity:** {info.get('debtToEquity', 'N/A')}
- **Return on Equity:** {info.get('returnOnEquity', 'N/A')}

**Business Description:**
{info.get('longBusinessSummary', 'No business summary available.')}

**Employee Count:** {info.get('fullTimeEmployees', 'N/A'):,}
        """

        return summary.strip()
    except Exception as e:
        return f"Error generating summary for {ticker}: {str(e)}"

# Function to analyze revenue data
def analyze_revenue_data(revenue_df, ticker):
    """Generate insights from revenue data"""
    if revenue_df is None or len(revenue_df) < 2:
        return "Insufficient data for analysis."

    try:
        # Calculate growth metrics
        latest_revenue = revenue_df.iloc[-1]['revenue']
        previous_revenue = revenue_df.iloc[-2]['revenue']
        growth_rate = ((latest_revenue - previous_revenue) / previous_revenue) * 100

        # Calculate CAGR if we have enough data
        if len(revenue_df) >= 3:
            first_revenue = revenue_df.iloc[0]['revenue']
            years = len(revenue_df) - 1
            cagr = ((latest_revenue / first_revenue) ** (1/years) - 1) * 100
        else:
            cagr = None

        # Generate analysis
        analysis = f"""
**Revenue Analysis for {ticker}:**

📈 **Latest Revenue:** ${latest_revenue/1e9:.2f}B
📊 **Year-over-Year Growth:** {growth_rate:.1f}%
"""

        if cagr:
            analysis += f"📈 **Compound Annual Growth Rate (CAGR):** {cagr:.1f}%\n"

        # Add trend analysis
        if growth_rate > 10:
            analysis += "🚀 **Strong Growth:** The company shows robust revenue growth.\n"
        elif growth_rate > 0:
            analysis += "📈 **Positive Growth:** The company is growing steadily.\n"
        else:
            analysis += "📉 **Declining Revenue:** The company faces revenue challenges.\n"

        # Add revenue range
        min_revenue = revenue_df['revenue'].min()
        max_revenue = revenue_df['revenue'].max()
        analysis += f"\n💰 **Revenue Range:** ${min_revenue/1e9:.2f}B - ${max_revenue/1e9:.2f}B"

        return analysis
    except Exception as e:
        return f"Error analyzing revenue data: {str(e)}"

# Function to analyze segment data
def analyze_segment_data(segment_df, ticker):
    """Generate insights from segment data"""
    if segment_df is None or segment_df.empty:
        return "No segment data available for analysis."

    try:
        # Get latest year data
        latest_year = segment_df['year'].max()
        latest_data = segment_df[segment_df['year'] == latest_year]

        # Calculate segment percentages
        total_revenue = latest_data['revenue'].sum()
        segment_analysis = f"**Segment Analysis for {ticker} ({latest_year}):**\n\n"

        for _, row in latest_data.iterrows():
            percentage = (row['revenue'] / total_revenue) * 100
            segment_analysis += f"• **{row['segment']}:** ${row['revenue']/1e9:.2f}B ({percentage:.1f}%)\n"

        # Find dominant segment
        dominant_segment = latest_data.loc[latest_data['revenue'].idxmax(), 'segment']
        dominant_percentage = (latest_data['revenue'].max() / total_revenue) * 100

        segment_analysis += f"\n🏆 **Dominant Segment:** {dominant_segment} ({dominant_percentage:.1f}% of total revenue)"

        return segment_analysis
    except Exception as e:
        return f"Error analyzing segment data: {str(e)}"

# Stock Prediction Functions
def get_market_sentiment_data():
    """Get market sentiment indicators"""
    try:
        # Get VIX (Fear & Greed Index)
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(period="1mo")
        current_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else 20

        # Get major indices for market trend
        sp500 = yf.Ticker("^GSPC")
        sp500_data = sp500.history(period="3mo")
        sp500_trend = ((sp500_data['Close'].iloc[-1] - sp500_data['Close'].iloc[0]) / sp500_data['Close'].iloc[0]) * 100 if not sp500_data.empty else 0

        return {
            'vix': current_vix,
            'market_trend': sp500_trend,
            'sentiment': 'Bullish' if sp500_trend > 5 else 'Bearish' if sp500_trend < -5 else 'Neutral'
        }
    except Exception as e:
        return {
            'vix': 20,
            'market_trend': 0,
            'sentiment': 'Neutral'
        }

def calculate_technical_indicators(df):
    """Calculate technical indicators for prediction"""
    try:
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()

        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return df

def simple_prediction_model(df, days=30):
    """Simple prediction model using moving averages and trends"""
    try:
        # Get recent data
        recent_data = df.tail(60).copy()

        # Calculate trend
        prices = recent_data['Close'].values
        x = np.arange(len(prices))

        # Simple linear regression
        coeffs = np.polyfit(x, prices, 1)
        trend_slope = coeffs[0]

        # Calculate volatility
        volatility = recent_data['Close'].pct_change().std()

        # Generate predictions
        last_price = prices[-1]
        predictions = []

        for i in range(1, days + 1):
            # Base prediction using trend
            trend_prediction = last_price + (trend_slope * i)

            # Add some randomness based on volatility
            noise = np.random.normal(0, volatility * last_price * 0.1)
            predicted_price = trend_prediction + noise

            # Ensure price doesn't go negative
            predicted_price = max(predicted_price, last_price * 0.5)

            predictions.append(predicted_price)

        return np.array(predictions)
    except Exception as e:
        st.error(f"Error in prediction model: {str(e)}")
        return np.array([df['Close'].iloc[-1]] * days)

def advanced_prediction_model(df, days=30):
    """Advanced prediction using machine learning if available"""
    if not SKLEARN_AVAILABLE:
        return simple_prediction_model(df, days)

    try:
        # Prepare features
        df_ml = calculate_technical_indicators(df.copy())

        # Select features for ML model
        feature_columns = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volume_Ratio', 'Volatility']

        # Remove NaN values
        df_ml = df_ml.dropna()

        if len(df_ml) < 50:  # Not enough data for ML
            return simple_prediction_model(df, days)

        # Prepare training data
        X = df_ml[feature_columns].values
        y = df_ml['Close'].values

        # Use last 80% for training
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Generate predictions
        last_features = X_test_scaled[-1].reshape(1, -1)
        predictions = []

        for i in range(days):
            pred = model.predict(last_features)[0]
            predictions.append(pred)

            # Update features for next prediction (simplified)
            # In reality, you'd need to recalculate technical indicators
            last_features = last_features.copy()

        return np.array(predictions)

    except Exception as e:
        st.error(f"Error in advanced prediction: {str(e)}")
        return simple_prediction_model(df, days)

def get_investment_recommendation(current_price, predicted_prices, market_sentiment, ticker_info):
    """Generate investment recommendation based on predictions and market data"""
    try:
        # Calculate prediction metrics
        avg_predicted = np.mean(predicted_prices)
        price_change_pct = ((avg_predicted - current_price) / current_price) * 100

        # Get market sentiment
        sentiment_score = 0
        if market_sentiment['sentiment'] == 'Bullish':
            sentiment_score = 1
        elif market_sentiment['sentiment'] == 'Bearish':
            sentiment_score = -1

        # Analyze company fundamentals
        pe_ratio = ticker_info.get('trailingPE', 0)
        debt_to_equity = ticker_info.get('debtToEquity', 0)
        profit_margin = ticker_info.get('profitMargins', 0)

        # Calculate recommendation score
        score = 0

        # Price prediction factor (40% weight)
        if price_change_pct > 10:
            score += 4
        elif price_change_pct > 5:
            score += 2
        elif price_change_pct > 0:
            score += 1
        elif price_change_pct < -10:
            score -= 4
        elif price_change_pct < -5:
            score -= 2
        else:
            score -= 1

        # Market sentiment factor (20% weight)
        score += sentiment_score * 2

        # Fundamentals factor (40% weight)
        if pe_ratio > 0 and pe_ratio < 15:
            score += 2
        elif pe_ratio > 30:
            score -= 1

        if profit_margin > 0.15:
            score += 2
        elif profit_margin < 0:
            score -= 2

        if debt_to_equity < 0.3:
            score += 1
        elif debt_to_equity > 1:
            score -= 1

        # Generate recommendation
        if score >= 6:
            recommendation = "🟢 STRONG BUY"
            confidence = "High"
            reasoning = "Strong upward price prediction, positive market sentiment, and solid fundamentals."
        elif score >= 3:
            recommendation = "🟢 BUY"
            confidence = "Medium"
            reasoning = "Positive indicators outweigh negative ones. Good potential for growth."
        elif score >= 0:
            recommendation = "🟡 HOLD"
            confidence = "Medium"
            reasoning = "Mixed signals. Consider holding current position or wait for clearer trends."
        elif score >= -3:
            recommendation = "🔴 SELL"
            confidence = "Medium"
            reasoning = "Negative indicators suggest potential downside. Consider reducing position."
        else:
            recommendation = "🔴 STRONG SELL"
            confidence = "High"
            reasoning = "Multiple negative indicators suggest significant downside risk."

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': reasoning,
            'predicted_change': price_change_pct,
            'score': score,
            'target_price': avg_predicted
        }

    except Exception as e:
        return {
            'recommendation': "🟡 HOLD",
            'confidence': "Low",
            'reasoning': f"Unable to generate recommendation due to data issues: {str(e)}",
            'predicted_change': 0,
            'score': 0,
            'target_price': current_price
        }

def plot_stock_prediction(historical_data, predictions, ticker):
    """Plot historical data with predictions"""
    try:
        # Prepare data
        hist_dates = historical_data.index
        hist_prices = historical_data['Close']

        # Create future dates
        last_date = hist_dates[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions), freq='D')

        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot historical data
        ax.plot(hist_dates[-60:], hist_prices[-60:], label='Historical Prices', color='blue', linewidth=2)

        # Plot predictions
        ax.plot(future_dates, predictions, label='Predicted Prices', color='red', linewidth=2, linestyle='--')

        # Connect last historical point to first prediction
        ax.plot([hist_dates[-1], future_dates[0]], [hist_prices.iloc[-1], predictions[0]],
                color='red', linewidth=2, linestyle='--')

        # Add confidence bands (simple approach)
        volatility = hist_prices.pct_change().std()
        upper_band = predictions * (1 + volatility)
        lower_band = predictions * (1 - volatility)

        ax.fill_between(future_dates, lower_band, upper_band, alpha=0.2, color='red', label='Confidence Band')

        # Formatting
        ax.set_title(f'{ticker} Stock Price Prediction - Next 30 Days', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format y-axis
        formatter = mticker.FuncFormatter(lambda x, pos: f'${x:.2f}')
        ax.yaxis.set_major_formatter(formatter)

        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    except Exception as e:
        st.error(f"Error creating prediction plot: {str(e)}")
        return None

def plot_interactive_prediction(historical_data, predictions, ticker):
    """Create interactive plot using plotly if available"""
    if not PLOTLY_AVAILABLE:
        return plot_stock_prediction(historical_data, predictions, ticker)

    try:
        # Prepare data
        hist_dates = historical_data.index
        hist_prices = historical_data['Close']

        # Create future dates
        last_date = hist_dates[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions), freq='D')

        # Create plotly figure
        fig = go.Figure()

        # Add historical data
        fig.add_trace(go.Scatter(
            x=hist_dates[-60:],
            y=hist_prices[-60:],
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))

        # Add predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            name='Predicted Prices',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Add confidence bands
        volatility = hist_prices.pct_change().std()
        upper_band = predictions * (1 + volatility)
        lower_band = predictions * (1 - volatility)

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_band,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_band,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            name='Confidence Band',
            hoverinfo='skip'
        ))

        # Update layout
        fig.update_layout(
            title=f'{ticker} Stock Price Prediction - Next 30 Days',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            height=600
        )

        return fig

    except Exception as e:
        st.error(f"Error creating interactive plot: {str(e)}")
        return plot_stock_prediction(historical_data, predictions, ticker)


# Streamlit app Initialisation
st.title("📊 Enhanced Financial Analysis App")
st.markdown("### Analyze financial data for major public companies")

# Sidebar with supported companies
st.sidebar.title("🏢 Supported Companies")
st.sidebar.markdown("Select from popular tickers or enter your own:")

# Create columns for better layout
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown("**Tech Companies:**")
    for ticker in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA']:
        if st.button(ticker, key=f"tech_{ticker}"):
            st.session_state.selected_ticker = ticker

with col2:
    st.markdown("**Other Sectors:**")
    for ticker in ['JPM', 'JNJ', 'WMT', 'DIS', 'V', 'KO']:
        if st.button(ticker, key=f"other_{ticker}"):
            st.session_state.selected_ticker = ticker

## Main Function
def main():
    # Get company ticker from user input or sidebar selection
    default_ticker = st.session_state.get('selected_ticker', '')
    company = st.text_input(
        "Enter company ticker:",
        value=default_ticker,
        placeholder="e.g., AAPL, GOOGL, MSFT..."
    ).upper()

    if company:
        # Check if company is in our supported list
        company_name = SUPPORTED_COMPANIES.get(company, f"{company} Inc.")

        st.markdown(f"## Analysis for {company_name}")

        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Fetch financial data
        status_text.text("Fetching financial data...")
        progress_bar.progress(25)

        financial_data = get_company_financial_data(company)

        if not financial_data['success']:
            st.error(f"❌ Could not fetch data for {company}. Please check the ticker symbol.")
            st.info("💡 Try one of the supported companies from the sidebar.")
            return

        progress_bar.progress(50)
        status_text.text("Processing revenue data...")

        # Process revenue data
        revenue_df = process_revenue_data(financial_data, company)

        progress_bar.progress(75)
        status_text.text("Creating visualizations...")

        # Create segment and regional data
        segment_df, segment_pivot = create_segment_data(revenue_df, company)
        regional_df, regional_pivot = create_regional_data(revenue_df, company)

        progress_bar.progress(100)
        status_text.text("✅ Data loaded successfully!")

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # User chooses an analysis option
        option = st.selectbox(
            "Choose an analysis option:",
            (
                "📋 Company Overview",
                "📊 Revenue Trend Analysis",
                "🏭 Product-Based Revenue Insights",
                "🌍 Region-Based Revenue Insights",
                "🔮 Stock Price Prediction & Investment Advice",
            ),
        )

        ## OPTION 1: Company Overview
        if option == "📋 Company Overview":
            st.markdown("### Company Overview")
            summary = get_company_summary(company, financial_data)
            st.markdown(summary)

            # Display key financial metrics in columns
            if 'info' in financial_data:
                info = financial_data['info']
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")
                with col2:
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                with col3:
                    st.metric("Revenue (TTM)", f"${info.get('totalRevenue', 0)/1e9:.1f}B")
                with col4:
                    st.metric("Profit Margin", f"{info.get('profitMargins', 0):.1%}")

        ## OPTION 2: Revenue Trend Analysis
        elif option == "📊 Revenue Trend Analysis":
            st.markdown("### Revenue Trend Analysis")

            if revenue_df is not None and not revenue_df.empty:
                # Plot revenue trend
                fig = plot_revenue_trend(revenue_df, company_name)
                if fig:
                    st.pyplot(fig)

                # Show analysis
                analysis = analyze_revenue_data(revenue_df, company)
                st.markdown(analysis)

                # Show revenue data table
                st.markdown("#### Revenue Data")
                display_df = revenue_df.copy()
                display_df['revenue'] = display_df['revenue'].apply(lambda x: f"${x/1e9:.2f}B")
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No revenue data available for this company.")

        ## OPTION 3: Product wise Revenue Insights
        elif option == "🏭 Product-Based Revenue Insights":
            st.markdown("### Product-Based Revenue Analysis")

            if segment_pivot is not None and not segment_pivot.empty:
                # Visualization
                st.markdown("#### Revenue by Product Segments")
                fig = plot_revenue_by_product(segment_pivot, company_name)
                if fig:
                    st.pyplot(fig)

                # Analysis
                st.markdown("#### Segment Analysis")
                analysis = analyze_segment_data(segment_df, company)
                st.markdown(analysis)

                # Show segment data
                with st.expander("📊 View Detailed Segment Data"):
                    display_df = segment_df.copy()
                    display_df['revenue'] = display_df['revenue'].apply(lambda x: f"${x/1e9:.2f}B")
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No segment data available for visualization.")

        ## OPTION 4: Region-Based Revenue Insights
        elif option == "🌍 Region-Based Revenue Insights":
            st.markdown("### Geographic Revenue Analysis")

            if regional_pivot is not None and not regional_pivot.empty:
                # Visualization
                st.markdown("#### Revenue by Geographic Region")
                fig = plot_revenue_by_region(regional_pivot, company_name)
                if fig:
                    st.pyplot(fig)

                # Analysis
                st.markdown("#### Regional Analysis")
                analysis = analyze_segment_data(regional_df.rename(columns={'region': 'segment'}), company)
                st.markdown(analysis.replace('Segment', 'Regional'))

                # Show regional data
                with st.expander("🌍 View Detailed Regional Data"):
                    display_df = regional_df.copy()
                    display_df['revenue'] = display_df['revenue'].apply(lambda x: f"${x/1e9:.2f}B")
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No regional data available for visualization.")

        ## OPTION 5: Stock Price Prediction & Investment Advice
        elif option == "🔮 Stock Price Prediction & Investment Advice":
            st.markdown("### 🔮 AI-Powered Stock Prediction & Investment Analysis")

            # Get extended historical data for prediction
            try:
                stock = yf.Ticker(company)
                hist_data = stock.history(period="2y")  # 2 years of data for better prediction

                if hist_data.empty:
                    st.error("❌ Unable to fetch historical data for prediction.")
                    return

                # Get current market sentiment
                with st.spinner("🌐 Analyzing market sentiment and geopolitical factors..."):
                    market_sentiment = get_market_sentiment_data()

                # Display current market conditions
                st.markdown("#### 🌍 Current Market Conditions")
                col1, col2, col3 = st.columns(3)

                with col1:
                    vix_color = "🔴" if market_sentiment['vix'] > 30 else "🟡" if market_sentiment['vix'] > 20 else "🟢"
                    st.metric("VIX (Fear Index)", f"{market_sentiment['vix']:.1f}",
                             help="VIX measures market volatility. <20: Low fear, 20-30: Moderate, >30: High fear")
                    st.markdown(f"{vix_color} {'High Fear' if market_sentiment['vix'] > 30 else 'Moderate Fear' if market_sentiment['vix'] > 20 else 'Low Fear'}")

                with col2:
                    trend_color = "🟢" if market_sentiment['market_trend'] > 0 else "🔴"
                    st.metric("S&P 500 Trend (3M)", f"{market_sentiment['market_trend']:.1f}%")
                    st.markdown(f"{trend_color} {market_sentiment['sentiment']} Market")

                with col3:
                    current_price = hist_data['Close'].iloc[-1]
                    st.metric("Current Price", f"${current_price:.2f}")

                # Prediction settings
                st.markdown("#### ⚙️ Prediction Settings")
                col1, col2 = st.columns(2)

                with col1:
                    prediction_days = st.slider("Prediction Period (Days)", 7, 60, 30)

                with col2:
                    model_type = st.selectbox("Prediction Model",
                                            ["Advanced ML Model" if SKLEARN_AVAILABLE else "Simple Trend Model",
                                             "Simple Trend Model"])

                # Generate predictions
                if st.button("🔮 Generate Prediction", type="primary"):
                    with st.spinner("🤖 Running AI prediction models..."):
                        if model_type == "Advanced ML Model" and SKLEARN_AVAILABLE:
                            predictions = advanced_prediction_model(hist_data, prediction_days)
                        else:
                            predictions = simple_prediction_model(hist_data, prediction_days)

                    # Display prediction results
                    st.markdown("#### 📈 Price Prediction Results")

                    # Create prediction plot
                    if PLOTLY_AVAILABLE:
                        fig = plot_interactive_prediction(hist_data, predictions, company)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = plot_stock_prediction(hist_data, predictions, company)
                        if fig:
                            st.pyplot(fig)

                    # Prediction metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        predicted_price = predictions[-1]
                        st.metric("Predicted Price (30D)", f"${predicted_price:.2f}")

                    with col2:
                        price_change = ((predicted_price - current_price) / current_price) * 100
                        st.metric("Expected Change", f"{price_change:+.1f}%",
                                 delta=f"${predicted_price - current_price:+.2f}")

                    with col3:
                        max_price = np.max(predictions)
                        st.metric("Predicted High", f"${max_price:.2f}")

                    with col4:
                        min_price = np.min(predictions)
                        st.metric("Predicted Low", f"${min_price:.2f}")

                    # Investment recommendation
                    st.markdown("#### 💡 AI Investment Recommendation")

                    recommendation = get_investment_recommendation(
                        current_price, predictions, market_sentiment, financial_data['info']
                    )

                    # Display recommendation in a prominent box
                    st.markdown(f"""
                    <div style="
                        padding: 20px;
                        border-radius: 10px;
                        background: linear-gradient(90deg, #f0f2f6, #ffffff);
                        border-left: 5px solid {'#00ff00' if 'BUY' in recommendation['recommendation'] else '#ff0000' if 'SELL' in recommendation['recommendation'] else '#ffaa00'};
                        margin: 20px 0;
                    ">
                        <h3 style="margin: 0; color: #333;">{recommendation['recommendation']}</h3>
                        <p style="margin: 10px 0; font-size: 16px;"><strong>Confidence:</strong> {recommendation['confidence']}</p>
                        <p style="margin: 10px 0; font-size: 14px;">{recommendation['reasoning']}</p>
                        <p style="margin: 10px 0; font-size: 14px;"><strong>Target Price:</strong> ${recommendation['target_price']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Risk factors and considerations
                    st.markdown("#### ⚠️ Risk Factors & Considerations")

                    risk_factors = [
                        "📊 **Market Volatility**: Stock predictions are inherently uncertain and subject to market volatility.",
                        "🌍 **Geopolitical Events**: Unexpected global events can significantly impact stock prices.",
                        "📰 **Company News**: Earnings reports, product launches, or management changes can affect prices.",
                        "💹 **Economic Indicators**: Interest rates, inflation, and economic data influence market movements.",
                        "🔄 **Model Limitations**: Predictions are based on historical data and may not capture future market dynamics."
                    ]

                    for factor in risk_factors:
                        st.markdown(f"• {factor}")

                    # Disclaimer
                    st.markdown("""
                    ---
                    **⚠️ IMPORTANT DISCLAIMER**: This prediction is for educational purposes only and should not be considered as financial advice.
                    Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.
                    """)

            except Exception as e:
                st.error(f"❌ Error in prediction analysis: {str(e)}")
                st.info("💡 Try selecting a different company or check your internet connection.")

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 14px;">'
        '📊 Enhanced Financial Analysis App | '
        'Data powered by Yahoo Finance | '
        'Built with Streamlit'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

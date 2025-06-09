# Financial Analysis App

## Overview
The Financial Analysis App is a comprehensive web application built to analyze financial data of publicly traded companies using real-time data from Yahoo Finance. It provides advanced functionalities to fetch, analyze, and visualize financial statements, market trends, and stock predictions, enabling users to gain deep insights into company performance and make informed investment decisions.

## Features

### 1. Company Overview
Get a comprehensive summary of any publicly traded company including:
- **Financial Highlights:** Market cap, enterprise value, revenue (TTM), profit margins
- **Key Metrics:** P/E ratio, price-to-book ratio, debt-to-equity, return on equity
- **Company Information:** Sector, industry, country, website, employee count
- **Business Description:** Detailed business summary and operations overview

### 2. Revenue Trend Analysis
Analyze historical revenue performance with:
- **Interactive Revenue Charts:** Visual representation of revenue trends over time
- **Growth Metrics:** Year-over-year growth rates and compound annual growth rate (CAGR)
- **Performance Insights:** Automated analysis of revenue patterns and growth trajectory
- **Data Tables:** Detailed revenue data with formatted financial figures

### 3. Product-Based Revenue Insights
Explore revenue breakdown by business segments:
- **Segment Analysis:** Revenue distribution across different product categories
- **Visual Charts:** Stacked bar charts showing segment performance over time
- **Dominant Segments:** Identification of key revenue drivers
- **Strategic Insights:** Understanding of business diversification and focus areas

### 4. Region-Based Revenue Insights
Understand geographical revenue distribution:
- **Regional Analysis:** Revenue breakdown by geographic markets
- **Global Presence:** Assessment of international vs domestic revenue
- **Market Expansion:** Insights into regional growth opportunities
- **Risk Assessment:** Geographic diversification analysis

### 5. Stock Price Prediction & Investment Advice
Advanced AI-powered stock analysis featuring:
- **Technical Analysis:** RSI, MACD, Bollinger Bands, moving averages
- **Market Sentiment:** VIX analysis and S&P 500 trend indicators
- **Price Predictions:** 7-60 day stock price forecasts using advanced algorithms
- **Investment Recommendations:** AI-generated buy/sell/hold recommendations
- **Risk Assessment:** Comprehensive risk factors and market considerations
- **Interactive Charts:** Plotly-powered interactive visualizations with confidence bands
## How to Run the Application

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd financial_analysis_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the app:**
   Open your web browser and navigate to `http://localhost:8501`


##  Supported Companies
The application supports analysis for 80+ major publicly traded companies across various sectors:

**Technology:** AAPL, GOOGL, MSFT, AMZN, META, NVDA, NFLX, ADBE, CRM, ORCL, IBM, INTC, AMD, and more

**Finance:** JPM, BAC, WFC, GS, MS, V, MA

**Consumer:** WMT, TGT, HD, LOW, SBUX, MCD, NKE, DIS

**Healthcare:** JNJ, PFE, UNH, CVS

**Energy:** XOM, CVX, COP

**And many more!** You can also analyze any publicly traded company by entering its ticker symbol.

##  Tech Stack

### Core Technologies
- **Python 3.8+:** Primary programming language chosen for its robust ecosystem of financial and data analysis libraries
- **Streamlit:** Modern web framework for building interactive data applications with minimal code
- **Yahoo Finance API:** Real-time financial data source providing comprehensive market information

### Data Processing & Analysis
- **Pandas:** Advanced data manipulation and analysis library for handling financial datasets
- **NumPy:** Numerical computing library for mathematical operations and array processing
- **yfinance:** Python library for accessing Yahoo Finance data programmatically

### Visualization & UI
- **Matplotlib:** Comprehensive plotting library for creating static, publication-quality charts
- **Plotly:** Interactive visualization library for dynamic, web-based charts and graphs
- **Streamlit Components:** UI elements for better user experience

### Financial Analysis Features
- **Technical Indicators:** RSI, MACD, Bollinger Bands, Moving Averages
- **Market Sentiment Analysis:** VIX tracking and market trend analysis
- **Predictive Modeling:** Custom algorithms for stock price forecasting
- **Risk Assessment:** Comprehensive investment recommendation system

## Dependencies

### Core Dependencies
```
streamlit==1.34.0    # Web application framework
pandas               # Data manipulation and analysis
numpy                # Numerical computing
yfinance             # Yahoo Finance data access
matplotlib==3.8.2    # Static plotting and visualization
plotly               # Interactive charts and graphs
requests             # HTTP library for API calls
```

### Key Libraries Explained
- **yfinance:** Provides access to Yahoo Finance's comprehensive financial data including stock prices, financial statements, company information, and market indices
- **Pandas:** Essential for data manipulation, cleaning, and analysis of financial datasets with powerful DataFrame operations
- **Matplotlib:** Creates professional-quality static charts for revenue analysis, trend visualization, and financial reporting
- **Plotly:** Enables interactive visualizations with zoom, pan, and hover capabilities for user experience
- **Streamlit:** Transforms Python scripts into interactive web applications with minimal configuration
- **NumPy:** Supports mathematical operations required for financial calculations, technical indicators, and predictive modeling

## Installation Requirements
- Python 3.8 or higher
- Internet connection for real-time data fetching
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Minimum 4GB RAM recommended for optimal performance

## Key Features in Detail

### Real-Time Data Integration
- **Live Market Data:** Real-time stock prices, market indices, and financial metrics
- **Historical Analysis:** Up to 5 years of historical financial data for trend analysis
- **Market Sentiment:** VIX fear index and S&P 500 trend indicators for market context

### Advanced Analytics
- **Technical Analysis:**
  - RSI (Relative Strength Index) for momentum analysis
  - MACD (Moving Average Convergence Divergence) for trend identification
  - Bollinger Bands for volatility assessment
  - Multiple moving averages (SMA, EMA) for trend analysis

- **Predictive Modeling:**
  - Custom algorithms combining multiple technical indicators
  - Market sentiment integration for predictions
  - Confidence bands and risk assessment
  - 7-60 day forecasting capabilities

### Investment Intelligence
- **Automated Recommendations:** AI-powered buy/sell/hold suggestions
- **Risk Assessment:** Comprehensive analysis of investment risks
- **Performance Metrics:** ROI calculations, growth rates, and financial ratios
- **Market Context:** Integration of broader market conditions in analysis

## How to Use

1. **Select a Company:** Enter a ticker symbol or choose from the sidebar of popular companies
2. **Choose Analysis Type:** Select from 5 different analysis options
3. **View Results:** Interactive charts, detailed metrics, and actionable insights
4. **Export Data:** Download charts and data for further analysis
5. **Make Informed Decisions:** Use AI recommendations and risk assessments

## Data Sources & Reliability
- **Primary Source:** Yahoo Finance API for real-time and historical data
- **Market Indices:** S&P 500, VIX, and other major market indicators
- **Update Frequency:** Real-time for current prices, daily for historical data
- **Data Validation:** Built-in error handling and data quality checks

## Disclaimer
This application is designed for educational and informational purposes only. The stock predictions and investment recommendations provided should not be considered as financial advice. Always consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results, and all investments carry inherent risks.

## Contributing
Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to help improve the application.




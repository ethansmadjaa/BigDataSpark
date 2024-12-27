from datetime import datetime
import yfinance as yf
import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import last, date_add
import concurrent.futures
from functools import partial
import threading
import time


def get_ytd_days() -> int:
    """
    Calculate the number of days from the start of the current year.
    
    Returns:
        int: Number of days since January 1st of current year
    """
    today = datetime.now()
    start_of_year = datetime(today.year, 1, 1)  # January 1st of current year
    return (today - start_of_year).days


def get_stock_info(ticker: str, spark: SparkSession):
    """
    Fetch stock information from Yahoo Finance and create a cached Spark DataFrame.
    
    Args:
        ticker (str): Stock ticker symbol
        spark (SparkSession): Active Spark session
        
    Returns:
        DataFrame: Cached Spark DataFrame containing stock information
    """
    schema = StructType([
        StructField('ticker', StringType(), True),
        StructField("name", StringType(), True),
        StructField("sector", StringType(), True),
        StructField("industry", StringType(), True),
        StructField("market_cap", DoubleType(), True),
        StructField("current_price", DoubleType(), True),
        StructField("price_change", DoubleType(), True),
        StructField("volume", DoubleType(), True)
    ])

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price - try different possible fields
        current_price = float(
            info.get("currentPrice") or 
            info.get("regularMarketPrice") or 
            info.get("price") or 
            0
        )
        
        # Get previous close price
        previous_close = float(
            info.get("previousClose") or 
            info.get("regularMarketPreviousClose") or 
            current_price
        )
        
        # Calculate price change
        price_change = ((current_price - previous_close) / previous_close * 100) if previous_close else 0

        # Convert market cap to float
        market_cap = float(info.get("marketCap", 0))
        
        # Convert volume to float
        volume = float(info.get("volume") or info.get("regularMarketVolume") or 0)

        stock_row = Row(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=market_cap,
            current_price=current_price,
            price_change=price_change,
            volume=volume
        )

        df = spark.createDataFrame([stock_row], schema=schema)
        return df.cache()

    except Exception as e:
        st.error(f"Error fetching stock info for {ticker}: {str(e)}")
        empty_row = Row(
            ticker=ticker,
            name=None,
            sector=None,
            industry=None,
            market_cap=None,
            current_price=None,
            price_change=None,
            volume=None
        )
        df = spark.createDataFrame([empty_row], schema=schema)
        return df.cache()


def format_market_cap(market_cap: int) -> str:
    """
    Format market capitalization value into a human-readable string.
    
    Args:
        market_cap (int): Market capitalization value
        
    Returns:
        str: Formatted string (e.g., "$1.5T" for 1.5 trillion)
    """
    if market_cap is None:
        return 'N/A'

    if market_cap >= 1e12:
        return f"${market_cap / 1e12:.1f}T"
    elif market_cap >= 1e9:
        return f"${market_cap / 1e9:.1f}B"
    elif market_cap >= 1e6:
        return f"${market_cap / 1e6:.1f}M"
    else:
        return f"${market_cap:,.0f}" 


def fetch_with_timeout(func, timeout_seconds):
    """Execute function with timeout using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return None


def get_stock_history(ticker: str, spark: SparkSession, days: int = 365):
    """Fetch historical stock data with timeout handling."""
    try:
        # Function to fetch stock data
        def fetch_stock_data():
            stock = yf.Ticker(ticker)
            
            # Convert days to appropriate period format
            if days <= 5:
                period = "5d"
            elif days <= 30:
                period = "1mo"
            elif days <= 90:
                period = "3mo"
            elif days <= 180:
                period = "6mo"
            elif days <= 365:
                period = "1y"
            elif days <= 730:
                period = "2y"
            elif days <= 1825:
                period = "5y"
            else:
                period = "max"
            
            return stock.history(period=period)
            
        # Show loading message
        status_placeholder = st.empty()  # Create empty placeholder
        with status_placeholder, st.spinner(f"Fetching {get_period_name(days)} of historical data for {ticker}..."):
            # Fetch data with timeout
            hist = fetch_with_timeout(fetch_stock_data, 30)
        
        if hist is None:
            st.error(f"Request timed out while fetching data for {ticker}")
            return None
            
        if hist.empty:
            st.warning(f"No historical data found for {ticker}")
            return None
        
        # Process the data
        with st.spinner("Processing data..."):
            # Reset index to make Date a column instead of index
            hist = hist.reset_index()
            
            # Convert datetime to date string for better compatibility
            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
            
            # Convert Volume to float
            hist['Volume'] = hist['Volume'].astype(float)
            
            # Define schema explicitly for better control
            schema = StructType([
                StructField("Date", StringType(), True),
                StructField("Open", DoubleType(), True),
                StructField("High", DoubleType(), True),
                StructField("Low", DoubleType(), True),
                StructField("Close", DoubleType(), True),
                StructField("Volume", DoubleType(), True),
                StructField("Dividends", DoubleType(), True),
                StructField("Stock Splits", DoubleType(), True)
            ])
            
            # Create Spark DataFrame with explicit schema
            spark_df = spark.createDataFrame(hist, schema=schema)
            rows_count = spark_df.count()
            
            # Cache the DataFrame
            cached_df = spark_df.cache()
            
            # Show success message briefly then clear it
            with status_placeholder:
                st.success(f"Successfully loaded {rows_count} records for {ticker}")
                time.sleep(1)  # Show message for 1 second
                status_placeholder.empty()  # Clear the message
                
            return cached_df
        
    except Exception as e:
        st.error(f"Error processing {ticker}: {str(e)}")
        return None


def get_period_name(days: int) -> str:
    """Convert days to human-readable period name."""
    if days <= 5:
        return "5 days"
    elif days <= 30:
        return "1 month"
    elif days <= 90:
        return "3 months"
    elif days <= 180:
        return "6 months"
    elif days <= 365:
        return "1 year"
    elif days <= 730:
        return "2 years"
    elif days <= 1825:
        return "5 years"
    else:
        return "maximum available" 
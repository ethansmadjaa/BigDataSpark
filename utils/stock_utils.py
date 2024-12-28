import concurrent.futures
import time
from datetime import datetime

import streamlit as st
import yfinance as yf
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType


# TODO: handle yfinance timeouts better - keeps failing randomly
# TODO: add caching for frequently accessed stocks
# TODO: fix the weird bug with market cap calculation
# TODO: clean up the messy period conversion logic
# TODO: add more error handling for API failures

def get_ytd_days() -> int:
    """Get days since start of year."""
    today = datetime.now()
    jan1 = datetime(today.year, 1, 1)  # start of year
    return (today - jan1).days


def get_stock_info(tkr: str, spark: SparkSession):
    """
    Get basic stock info from Yahoo.
    Sometimes fails cuz their API is wonky.
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
        stk = yf.Ticker(tkr)
        info = stk.info

        # try diff price fields cuz yahoo api is inconsistent
        curr_price = float(
            info.get("currentPrice") or
            info.get("regularMarketPrice") or
            info.get("price") or
            0
        )

        # get prev close if available
        prev_close = float(
            info.get("previousClose") or
            info.get("regularMarketPreviousClose") or
            curr_price
        )

        # calc % change
        pct_change = ((curr_price - prev_close) / prev_close * 100) if prev_close else 0

        # convert market cap to float - sometimes comes as string
        mkt_cap = float(info.get("marketCap", 0))

        # get volume, sometimes missing
        vol = float(info.get("volume") or info.get("regularMarketVolume") or 0)

        stock_row = Row(
            ticker=tkr,
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=mkt_cap,
            current_price=curr_price,
            price_change=pct_change,
            volume=vol
        )

        df = spark.createDataFrame([stock_row], schema=schema)
        return df.cache()  # cache it cuz why not

    except Exception as e:
        st.error(f"Failed to get data for {tkr}: {str(e)}")
        # return empty df if failed
        empty_row = Row(
            ticker=tkr,
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


def format_market_cap(mkt_cap: int) -> str:
    """Make market cap readable (1.5T instead of 1500000000000)."""
    if mkt_cap is None:
        return 'N/A'

    if mkt_cap >= 1e12:
        return f"${mkt_cap / 1e12:.1f}T"
    elif mkt_cap >= 1e9:
        return f"${mkt_cap / 1e9:.1f}B"
    elif mkt_cap >= 1e6:
        return f"${mkt_cap / 1e6:.1f}M"
    else:
        return f"${mkt_cap:,.0f}"


def fetch_with_timeout(func, timeout_secs):
    """Run function with timeout cuz yahoo api is slow af."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_secs)
        except concurrent.futures.TimeoutError:
            return None


def get_stock_history(tkr: str, spark: SparkSession, days: int = 365):
    """Get historical data - might timeout if yahoo is being slow."""
    try:
        # func to get data from yahoo
        def fetch_stock_data():
            stk = yf.Ticker(tkr)

            # convert days to yahoo period format - kinda messy
            if days <= 5:
                prd = "5d"
            elif days <= 30:
                prd = "1mo"
            elif days <= 90:
                prd = "3mo"
            elif days <= 180:
                prd = "6mo"
            elif days <= 365:
                prd = "1y"
            elif days <= 730:
                prd = "2y"
            elif days <= 1825:
                prd = "5y"
            else:
                prd = "max"

            return stk.history(period=prd)

        # show loading msg
        status_msg = st.empty()
        with status_msg, st.spinner(f"Getting {get_period_name(days)} of data for {tkr}..."):
            # try to get data with timeout
            hist = fetch_with_timeout(fetch_stock_data, 30)

        if hist is None:
            st.error(f"Timeout getting data for {tkr}")
            return None

        if hist.empty:
            st.warning(f"No data found for {tkr}")
            return None

        # process the data we got
        with st.spinner("Processing..."):
            # make Date a normal column
            hist = hist.reset_index()

            # convert dates to strings
            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')

            # fix volume type
            hist['Volume'] = hist['Volume'].astype(float)

            # setup schema
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

            # create spark df
            spark_df = spark.createDataFrame(hist, schema=schema)
            n_rows = spark_df.count()

            # cache it for speed
            cached_df = spark_df.cache()

            # show quick success msg
            with status_msg:
                st.success(f"Got {n_rows} rows for {tkr}")
                time.sleep(1)  # show msg briefly
                status_msg.empty()

            return cached_df

    except Exception as e:
        st.error(f"Error with {tkr}: {str(e)}")
        return None


def get_period_name(days: int) -> str:
    """Convert days to something readable."""
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

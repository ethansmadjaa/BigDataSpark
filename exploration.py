import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, lag, mean, stddev, min, max,
    date_trunc, datediff, desc, first, format_number
)
from pyspark.sql.types import DoubleType, LongType, StructField, StringType, StructType

from utils.constants import STOCK_CATEGORIES
from utils.stock_utils import get_stock_history


# TODO: optimize performance of moving averages calc
# TODO: add more technical indicators (RSI, MACD etc)
# TODO: fix bug with correlation heatmap colors

def plot_stock_price_history(df: DataFrame, ticker: str) -> go.Figure:
    """Create a candlestick chart with volume."""
    st.markdown("""
        <div class="tooltip">ℹ️ About Candlestick Charts
            <span class="tooltiptext">
            <strong>How to Read Candlesticks:</strong><br>
            • Green/White: Closing price higher than opening<br>
            • Red/Black: Closing price lower than opening<br>
            • Wicks: Show high and low prices<br>
            • Volume: Trading activity intensity<br><br>
            Common patterns like Doji, Hammer, and Engulfing can signal potential reversals.
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # get data and sort it properly 
    data = df.orderBy('Date').collect()

    # setup subplot with 2 charts
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,  # might need to adjust this
        subplot_titles=(f'{ticker}\'s stock price', 'Volume'),
        row_heights=[0.7, 0.3]  # price takes more space than vol
    )

    # add candlestick - main price chart
    fig.add_trace(
        go.Candlestick(
            x=[row['Date'] for row in data],
            open=[row['Open'] for row in data],
            high=[row['High'] for row in data],
            low=[row['Low'] for row in data],
            close=[row['Close'] for row in data]
        ),
        row=1, col=1
    )

    # add volume bars below
    fig.add_trace(
        go.Bar(
            x=[row['Date'] for row in data],
            y=[row['Volume'] for row in data]
        ),
        row=2, col=1
    )

    # make it look nice
    fig.update_layout(
        height=600,
        template='plotly_dark',
        showlegend=False
    )

    return fig


def get_basic_stats(df: DataFrame) -> dict:
    """Get basic price/volume stats."""
    try:
        # repartition to make it faster
        df = df.repartition(col("Date"))

        # calc basic stuff we care about
        stats = df.agg(
            mean('Close').alias('avg_price'),
            min('Close').alias('min_price'),
            max('Close').alias('max_price'),
            mean('Volume').alias('avg_volume'),
            min('Volume').alias('min_volume'),
            max('Volume').alias('max_volume')
        ).collect()[0]

        # setup window for returns calc
        win_spec = Window.partitionBy(date_trunc("month", col("Date"))) \
            .orderBy("Date")

        # daily returns in %
        df_rets = df.withColumn(
            'daily_return',
            ((col('Close') - lag('Close', 1).over(win_spec)) /
             lag('Close', 1).over(win_spec)) * 100
        )

        # get return stats
        ret_stats = df_rets.agg(
            mean('daily_return').alias('avg_return'),
            stddev('daily_return').alias('volatility')
        ).collect()[0]

        return {
            'Price': {
                'Average': f"${float(stats['avg_price']):.2f}",
                'Minimum': f"${float(stats['min_price']):.2f}",
                'Maximum': f"${float(stats['max_price']):.2f}",
            },
            'Volume': {
                'Average': f"{int(float(stats['avg_volume'])):,}",
                'Minimum': f"{int(float(stats['min_volume'])):,}",
                'Maximum': f"{int(float(stats['max_volume'])):,}",
            },
            'Returns': {
                'Average Daily': f"{float(ret_stats['avg_return']):.2f}%",
                'Volatility': f"{float(ret_stats['volatility']):.2f}%"
            }
        }

    except Exception as e:
        print(f"Error in get_basic_stats: {str(e)}")
        return None


def analyze_data_quality(df: DataFrame) -> dict:
    """Analyze data quality including observations count, missing values, and data frequency."""
    try:
        # Repartition by year and month for better performance
        df = df.repartition(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        )

        # Create window spec with proper partitioning
        window_spec = Window.partitionBy(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        ).orderBy("Date")

        # Rest of the analysis with partitioned window
        date_diffs = df.select(
            "Date",
            (datediff(col("Date"),
                      lag("Date", 1).over(window_spec))
             ).alias("date_diff")
        ).filter(col("date_diff").isNotNull())

        # Get total observations
        total_rows = df.count()

        # Count missing values for each column
        missing_counts = {
            col_name: df.filter(col(col_name).isNull()).count()
            for col_name in df.columns
        }

        # Get the most common difference (mode)
        freq_mode = date_diffs.groupBy("date_diff") \
            .count() \
            .orderBy(desc("count")) \
            .first()

        # Map frequency to human-readable format
        freq_mapping = {
            1: "Daily",
            7: "Weekly",
            30: "Monthly",
            90: "Quarterly",
            365: "Yearly"
        }

        frequency = freq_mapping.get(
            freq_mode["date_diff"],
            f"Custom ({freq_mode['date_diff']} days)"
        )

        return {
            "Total Observations": total_rows,
            "Data Frequency": frequency,
            "Missing Values": missing_counts,
            "Date Range": {
                "Start": df.agg(min("Date")).first()[0],
                "End": df.agg(max("Date")).first()[0]
            }
        }

    except Exception as e:
        print(f"Error in analyze_data_quality: {str(e)}")
        return None


def calculate_correlations(df: DataFrame, spark: SparkSession) -> DataFrame:
    """Calculate correlations between numeric columns."""
    try:
        # Repartition for better performance
        df = df.repartition(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        )

        # Get numeric columns
        numeric_cols = [
            field.name for field in df.schema.fields
            if isinstance(field.dataType, (DoubleType, LongType))
        ]

        # Calculate correlations
        correlations = []
        for col1 in numeric_cols:
            for col2 in numeric_cols:
                correlation = df.stat.corr(col1, col2)
                correlations.append((col1, col2, correlation))

        # Create correlation DataFrame
        correlation_df = spark.createDataFrame(
            correlations,
            ["Column1", "Column2", "Correlation"]
        )

        return correlation_df.cache()  # Cache for multiple uses

    except Exception as e:
        print(f"Error calculating correlations: {str(e)}")
        return None


def plot_correlation_heatmap(correlation_df: DataFrame) -> go.Figure:
    """
    Create a heatmap visualization of correlations.
    
    Args:
        correlation_df (DataFrame): Correlation matrix DataFrame
        
    Returns:
        Figure: Plotly heatmap figure
    """
    # Convert to matrix format
    corr_data = correlation_df.collect()
    columns = list(set(row["Column1"] for row in corr_data))

    # Create correlation matrix
    matrix = [[0 for _ in columns] for _ in columns]
    for row in corr_data:
        i = columns.index(row["Column1"])
        j = columns.index(row["Column2"])
        matrix[i][j] = row["Correlation"]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=columns,
        y=columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title="Correlation Heatmap",
        template="plotly_dark",
        height=600
    )

    return fig


def calculate_stock_correlation(df1: DataFrame, df2: DataFrame) -> dict:
    """
    Calculate correlation between two stocks' closing prices and volumes.
    
    Args:
        df1 (DataFrame): First stock's historical data
        df2 (DataFrame): Second stock's historical data

    Returns:
        dict: Correlation metrics between the two stocks
    """
    try:
        # Join the dataframes on Date
        joined_df = df1.select(
            col("Date"),
            col("Close").alias("close1"),
            col("Volume").alias("volume1")
        ).join(
            df2.select(
                col("Date"),
                col("Close").alias("close2"),
                col("Volume").alias("volume2")
            ),
            on="Date",
            how="inner"
        )

        # Calculate correlations
        price_corr = joined_df.stat.corr("close1", "close2")
        volume_corr = joined_df.stat.corr("volume1", "volume2")

        return {
            "price_correlation": price_corr,
            "volume_correlation": volume_corr,
            "common_dates": joined_df.count()
        }

    except Exception as e:
        print(f"Error calculating stock correlation: {str(e)}")
        return None


def plot_stock_comparison(df1: DataFrame, df2: DataFrame, ticker1: str, ticker2: str) -> go.Figure:
    """
    Create a comparison plot of two stocks.
    """
    # Get data
    data1 = df1.orderBy("Date").collect()
    data2 = df2.orderBy("Date").collect()

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"Price Comparison: {ticker1} vs {ticker2}",
            "Volume Comparison"
        ),
        row_heights=[0.7, 0.3]
    )

    # Add price lines
    fig.add_trace(
        go.Scatter(
            x=[row["Date"] for row in data1],
            y=[row["Close"] for row in data1],
            name=f"{ticker1} Price",
            line=dict(color="#00B5F7")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[row["Date"] for row in data2],
            y=[row["Close"] for row in data2],
            name=f"{ticker2} Price",
            line=dict(color="#FF6B6B")
        ),
        row=1, col=1
    )

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=[row["Date"] for row in data1],
            y=[row["Volume"] for row in data1],
            name=f"{ticker1} Volume",
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=[row["Date"] for row in data2],
            y=[row["Volume"] for row in data2],
            name=f"{ticker2} Volume",
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def get_all_stocks_except(ticker: str) -> list:
    """
    Get list of all available stock tickers except the given one.
    
    Args:
        ticker (str): Ticker to exclude
        
    Returns:
        list: List of available stock tickers
    """
    all_stocks = []
    for category in STOCK_CATEGORIES.values():
        all_stocks.extend(category.keys())
    return [stock for stock in all_stocks if stock != ticker]


def calculate_price_averages(df: DataFrame) -> dict[str, DataFrame] | None:
    """
    Calculate average prices for different time periods.
    
    Args:
        df (DataFrame): Stock price DataFrame
        
    Returns:
        DataFrame: DataFrame with price averages by period
    """
    try:
        # Repartition for better performance
        df = df.repartition(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        )

        # Create period columns
        df_with_periods = df.withColumn(
            "week", date_trunc("week", col("Date"))
        ).withColumn(
            "month", date_trunc("month", col("Date"))
        ).withColumn(
            "year", date_trunc("year", col("Date"))
        )

        # Calculate averages for each period
        weekly_avg = df_with_periods.groupBy("week").agg(
            mean("Open").alias("avg_open"),
            mean("Close").alias("avg_close")
        )

        monthly_avg = df_with_periods.groupBy("month").agg(
            mean("Open").alias("avg_open"),
            mean("Close").alias("avg_close")
        )

        yearly_avg = df_with_periods.groupBy("year").agg(
            mean("Open").alias("avg_open"),
            mean("Close").alias("avg_close")
        )

        return {
            "weekly": weekly_avg.orderBy("week"),
            "monthly": monthly_avg.orderBy("month"),
            "yearly": yearly_avg.orderBy("year")
        }

    except Exception as e:
        print(f"Error calculating price averages: {str(e)}")
        return None


def calculate_price_changes(df: DataFrame) -> DataFrame:
    """Calculate day-to-day and month-to-month price changes."""
    try:
        # Repartition for better performance
        df = df.repartition(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        )

        # Create windows with proper partitioning
        day_window = Window.partitionBy(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        ).orderBy("Date")

        month_window = Window.partitionBy(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        ).orderBy("Date")

        return df.withColumn(
            "daily_change",
            col("Close") - lag("Close", 1).over(day_window)
        ).withColumn(
            "daily_change_pct",
            ((col("Close") - lag("Close", 1).over(day_window)) /
             lag("Close", 1).over(day_window) * 100)
        ).withColumn(
            "monthly_change",
            col("Close") - first("Close").over(month_window)
        ).withColumn(
            "monthly_change_pct",
            ((col("Close") - first("Close").over(month_window)) /
             first("Close").over(month_window) * 100)
        )

    except Exception as e:
        print(f"Error calculating price changes: {str(e)}")
        return None


def calculate_returns(df: DataFrame) -> DataFrame:
    """Calculate various return metrics."""
    try:
        # Repartition for better performance
        df = df.repartition(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        )

        # Create windows with proper partitioning
        day_window = Window.partitionBy(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        ).orderBy("Date")

        week_window = Window.partitionBy(
            date_trunc("year", col("Date")),
            date_trunc("week", col("Date"))
        ).orderBy("Date")

        month_window = Window.partitionBy(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        ).orderBy("Date")

        year_window = Window.partitionBy(
            date_trunc("year", col("Date"))
        ).orderBy("Date")

        # Calculate daily returns
        df_returns = df.withColumn(
            "daily_return",
            ((col("Close") - col("Open")) / col("Open") * 100)
        )

        # Calculate average returns for different periods
        df_returns = df_returns.withColumn(
            "weekly_avg_return",
            mean("daily_return").over(week_window)
        ).withColumn(
            "monthly_avg_return",
            mean("daily_return").over(month_window)
        ).withColumn(
            "yearly_avg_return",
            mean("daily_return").over(year_window)
        )

        return df_returns

    except Exception as e:
        print(f"Error calculating returns: {str(e)}")
        return None


def analyze_stock_performance(df: DataFrame) -> dict:
    """
    Comprehensive stock performance analysis.
    """
    try:
        # Get price averages
        price_avgs = calculate_price_averages(df)

        # Calculate changes and returns
        df_with_changes = calculate_price_changes(df)
        df_with_returns = calculate_returns(df_with_changes)

        # Get highest daily returns
        top_returns = df_with_returns.orderBy(desc("daily_return")).limit(5)

        # Calculate period averages
        period_stats = df_with_returns.agg(
            mean("daily_return").alias("avg_daily_return"),
            mean("weekly_avg_return").alias("avg_weekly_return"),
            mean("monthly_avg_return").alias("avg_monthly_return"),
            mean("yearly_avg_return").alias("avg_yearly_return"),
            stddev("daily_return").alias("daily_return_volatility")
        ).collect()[0]

        return {
            "price_averages": price_avgs,
            "top_returns": top_returns.collect(),
            "period_stats": {
                "Average Daily Return": f"{float(period_stats['avg_daily_return']):.2f}%",
                "Average Weekly Return": f"{float(period_stats['avg_weekly_return']):.2f}%",
                "Average Monthly Return": f"{float(period_stats['avg_monthly_return']):.2f}%",
                "Average Yearly Return": f"{float(period_stats['avg_yearly_return']):.2f}%",
                "Daily Volatility": f"{float(period_stats['daily_return_volatility']):.2f}%"
            }
        }

    except Exception as e:
        print(f"Error in performance analysis: {str(e)}")
        return None


def find_best_performing_stock(
        spark: SparkSession,
        start_date: str,
        period: str = 'month',
        top_n: int = 5
) -> DataFrame | None:
    """Find the best performing stocks for a given time period."""
    try:
        # Get all available stocks
        all_stocks = []
        for category in STOCK_CATEGORIES.values():
            all_stocks.extend(category.keys())

        # Calculate end date and days for data fetch
        start_ts = pd.to_datetime(start_date)
        if period.lower() == 'month':
            end_ts = start_ts + pd.Timedelta(days=30)
            days_to_fetch = 30
        else:  # year
            end_ts = start_ts + pd.Timedelta(days=365)
            days_to_fetch = 365

        end_date = end_ts.strftime("%Y-%m-%d")

        # Initialize results collection
        results = []

        # Process each stock
        for ticker in all_stocks:
            try:
                # Get stock data only for the required period
                df = get_stock_history(ticker, spark, days=days_to_fetch)
                if df is None:
                    continue

                # Filter for the specified period
                period_data = df.filter(
                    (col("Date") >= start_date) &
                    (col("Date") <= end_date)
                )

                if period_data.count() == 0:
                    continue

                # Get first and last prices directly
                first_last = period_data.agg(
                    min("Date").alias("first_date"),
                    max("Date").alias("last_date")
                ).collect()[0]

                # Get prices for first and last dates
                prices = period_data.filter(
                    (col("Date") == first_last["first_date"]) |
                    (col("Date") == first_last["last_date"])
                ).orderBy("Date").collect()

                if len(prices) < 2:
                    continue

                start_price = float(prices[0]["Close"])
                end_price = float(prices[-1]["Close"])

                # Calculate return
                period_return = ((end_price - start_price) / start_price) * 100

                results.append((
                    ticker,
                    start_price,
                    end_price,
                    period_return
                ))

            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue

        # Create DataFrame from results
        schema = StructType([
            StructField("ticker", StringType(), True),
            StructField("start_price", DoubleType(), True),
            StructField("end_price", DoubleType(), True),
            StructField("return_rate", DoubleType(), True)
        ])

        if not results:
            print("No results found for the specified period")
            return None

        results_df = spark.createDataFrame(results, schema)

        # Get top performers
        return results_df.orderBy(desc("return_rate")).limit(top_n)

    except Exception as e:
        print(f"Error finding best performers: {str(e)}")
        return None


def calculate_moving_average(
        df: DataFrame,
        column_name: str,
        window_size: int,
        partition_cols: list = None
) -> DataFrame:
    """
    Calculate moving average for a specified column.
    
    Args:
        df (DataFrame): Input DataFrame
        column_name (str): Column to calculate moving average for
        window_size (int): Number of periods for moving average
        partition_cols (list): Optional list of columns to partition by
        
    Returns:
        DataFrame: DataFrame with added moving average column
    """
    try:
        # Repartition for better performance
        df = df.repartition(
            date_trunc("year", col("Date")),
            date_trunc("month", col("Date"))
        )

        # Create window specification
        if partition_cols:
            window_spec = Window.partitionBy(partition_cols) \
                .orderBy("Date") \
                .rowsBetween(-(window_size - 1), 0)
        else:
            window_spec = Window.partitionBy(
                date_trunc("year", col("Date")),
                date_trunc("month", col("Date"))
            ).orderBy("Date") \
                .rowsBetween(-(window_size - 1), 0)

        # Calculate moving average
        ma_col_name = f"{column_name}_{window_size}MA"
        df_with_ma = df.withColumn(
            ma_col_name,
            mean(col(column_name)).over(window_spec)
        )

        return df_with_ma

    except Exception as e:
        print(f"Error calculating moving average: {str(e)}")
        return df


def explore_data(spark: SparkSession, ticker: str, days: int = 365):
    """Main exploration function."""
    st.title("Data Exploration")

    try:
        # grab historical data
        hist_df = get_stock_history(ticker, spark, days)
        if hist_df is None:
            st.warning("No historical data available.")
            return

        # price chart
        st.subheader("Price History")
        fig = plot_stock_price_history(hist_df, ticker)
        st.plotly_chart(fig, use_container_width=True)

        # raw data if someone wants to check
        with st.expander("View Raw Data"):
            st.dataframe(
                hist_df.orderBy(desc("Date"))
                .limit(100)  # dont show everything
            )

        # quality check
        st.subheader("Data Quality Analysis")
        qual_stats = analyze_data_quality(hist_df)
        if qual_stats:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Observations", qual_stats["Total Observations"])
                st.metric("Data Frequency", qual_stats["Data Frequency"])

            with col2:
                st.write("Date Range")
                st.write(f"Start: {qual_stats['Date Range']['Start']}")
                st.write(f"End: {qual_stats['Date Range']['End']}")

            st.write("Missing Values")
            missing_df = pd.DataFrame(
                qual_stats["Missing Values"].items(),
                columns=["Column", "Missing Count"]
            )
            st.dataframe(missing_df)

        # basic stats display
        st.subheader("Basic Statistics")
        stats = get_basic_stats(hist_df)
        if stats:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("##### Price")
                for key, value in stats['Price'].items():
                    st.metric(key, value)

            with col2:
                st.write("##### Volume")
                for key, value in stats['Volume'].items():
                    st.metric(key, value)

            with col3:
                st.write("##### Returns")
                for key, value in stats['Returns'].items():
                    st.metric(key, value)

        # comparison stuff
        st.subheader("Stock Comparison")

        # get other stocks we can compare with
        other_stocks = get_all_stocks_except(ticker)

        compare_stock = st.selectbox(
            "Select a stock to compare with",
            options=other_stocks,
            format_func=lambda
                x: f"{x} - {next((name for cat in STOCK_CATEGORIES.values() for t, name in cat.items() if t == x), x)}",
            key="compare_stock"
        )

        if compare_stock:
            compare_df = get_stock_history(compare_stock, spark, days)
            if compare_df is not None:
                # show comparison plot
                fig_compare = plot_stock_comparison(hist_df, compare_df, ticker, compare_stock)
                st.plotly_chart(fig_compare, use_container_width=True)

                # correlation stats
                corr_stats = calculate_stock_correlation(hist_df, compare_df)
                if corr_stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Price Correlation",
                            f"{corr_stats['price_correlation']:.2%}"
                        )
                    with col2:
                        st.metric(
                            "Volume Correlation",
                            f"{corr_stats['volume_correlation']:.2%}"
                        )
                    with col3:
                        st.metric(
                            "Common Trading Days",
                            corr_stats['common_dates']
                        )

        # warn about past performance
        st.warning("""
        ⚠️ **Performance Analysis Warning**
        - Past performance does not indicate future results
        - Historical data may not reflect current market conditions
        - Performance metrics should not be used in isolation
        - Market conditions and company fundamentals can change rapidly
        """)

        # advanced stuff section
        st.subheader("Advanced Analytics")

        perf_stats = analyze_stock_performance(hist_df)
        if perf_stats:
            st.info("""
            ⚠️ **Historical Returns Notice**
            - Returns shown are historical and not indicative of future performance
            - Trading based solely on historical patterns carries significant risk
            - Market conditions vary and past patterns may not repeat
            """)

            # show period stats
            st.write("##### Price Performance")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Period Returns")
                for metric, value in perf_stats["period_stats"].items():
                    st.metric(metric, value)

            with col2:
                st.write("Top Daily Returns")
                for day in perf_stats["top_returns"]:
                    st.metric(
                        f"Return on {day['Date']}",
                        f"{day['daily_return']:.2f}%",
                        f"Close: ${day['Close']:.2f}"
                    )

            # rest of the function remains unchanged...

        # Add Best Performers section
        st.subheader("Best Performing Stocks")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Select start date",
                value=pd.Timestamp.now() - pd.Timedelta(days=30)
            )
        with col2:
            period = st.selectbox(
                "Select period",
                options=["month", "year"],
                index=0
            )

        if st.button("Find Best Performers"):
            with st.spinner("Analyzing stock performance..."):
                top_performers = find_best_performing_stock(
                    spark,
                    start_date.strftime("%Y-%m-%d"),
                    period
                )

                if top_performers is not None:
                    st.write(f"Top Performers for {period} starting {start_date}")

                    # Display results
                    st.dataframe(
                        top_performers.select(
                            "ticker",
                            format_number("start_price", 2).alias("Start Price ($)"),
                            format_number("end_price", 2).alias("End Price ($)"),
                            format_number("return_rate", 2).alias("Return Rate (%)")
                        )
                    )
                else:
                    st.error("Unable to calculate top performers")

        # Add Moving Averages section
        st.subheader("Moving Averages")

        col1, col2 = st.columns(2)

        with col1:
            ma_column = st.selectbox(
                "Select column for moving average",
                options=["Open", "Close", "High", "Low"],
                index=1  # Default to Close price
            )

        with col2:
            window_size = st.slider(
                "Select window size (days)",
                min_value=5,
                max_value=200,
                value=20,
                step=5
            )

        if ma_column and window_size:
            # Calculate moving average
            hist_df_ma = calculate_moving_average(hist_df, ma_column, window_size)

            # Create visualization
            ma_data = hist_df_ma.orderBy("Date").collect()

            fig = go.Figure()

            # Add original price line
            fig.add_trace(
                go.Scatter(
                    x=[row["Date"] for row in ma_data],
                    y=[row[ma_column] for row in ma_data],
                    name=ma_column,
                    line=dict(color="#00B5F7")
                )
            )

            # Add moving average line
            ma_col_name = f"{ma_column}_{window_size}MA"
            fig.add_trace(
                go.Scatter(
                    x=[row["Date"] for row in ma_data],
                    y=[row[ma_col_name] for row in ma_data],
                    name=f"{window_size}-day MA",
                    line=dict(color="#FF6B6B")
                )
            )

            fig.update_layout(
                title=f"{ticker} - {ma_column} Price with {window_size}-day Moving Average",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_dark",
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show statistics
            with st.expander("Moving Average Statistics"):
                stats_df = hist_df_ma.select(
                    ma_column,
                    ma_col_name,
                    ((col(ma_column) - col(ma_col_name)) / col(ma_col_name) * 100).alias("deviation_pct")
                ).agg(
                    mean(col(ma_column)).alias("avg_price"),
                    mean(col(ma_col_name)).alias("avg_ma"),
                    mean(col("deviation_pct")).alias("avg_deviation"),
                    stddev(col("deviation_pct")).alias("deviation_volatility")
                ).collect()[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Price", f"${stats_df['avg_price']:.2f}")
                    st.metric("Average MA", f"${stats_df['avg_ma']:.2f}")
                with col2:
                    st.metric("Average Deviation", f"{stats_df['avg_deviation']:.2f}%")
                    st.metric("Deviation Volatility", f"{stats_df['deviation_volatility']:.2f}%")

    except Exception as e:
        st.error(f"Error in data exploration: {str(e)}")

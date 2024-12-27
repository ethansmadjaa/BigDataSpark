import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, isnan, avg, expr, stddev, lag
from pyspark.sql.window import Window


def add_technical_indicators(df: DataFrame, spark: SparkSession):
    """Add technical indicators to the dataframe."""
    # TODO: adjust the number of te tittle based on whats appearing here
    st.subheader("3. Technical Indicators")

    with st.expander("📈 Price-Based Indicators", expanded=True):
        st.markdown("""
        ### Understanding Technical Indicators
        
        ⚠️ **IMPORTANT RISK DISCLAIMER**
        - Technical indicators are tools, not guaranteed predictions
        - Always use multiple indicators and additional analysis
        - Past patterns may not repeat in the future
        - Market conditions can change rapidly
        
        Technical indicators help traders analyze price movements and identify potential trading opportunities.
        Here are the indicators available in this analysis:
        """)
        
        # Add warning for each indicator
        st.markdown("""
        ⚠️ **Trading Risk Notice**
        Each indicator has limitations and should not be used in isolation:
        - False signals can occur
        - Market conditions affect indicator reliability
        - Different timeframes may show conflicting signals
        """)
        
        # Simple Moving Averages (SMA)
        st.write("##### Simple Moving Averages (SMA)")
        st.markdown("""
        **What is SMA?**  
        A Simple Moving Average calculates the average price over a specified period. It helps identify trends by smoothing out price fluctuations.
        - Short-term SMAs (20 days) show immediate trends
        - Medium-term SMAs (50 days) show intermediate trends
        - Long-term SMAs (200 days) show long-term trends
        
        *Trading signals often occur when shorter SMAs cross longer SMAs.*
        """)
        sma_periods = st.multiselect(
            "Select SMA periods (days):",
            options=[5, 10, 20, 50, 100, 200],
            default=[20, 50, 200],
            help="Common SMA periods are 20 (month), 50 (quarter) and 200 (year) days"
        )

        # EMA
        st.write("##### Exponential Moving Average (EMA)")
        st.markdown("""
        **What is EMA?**  
        An Exponential Moving Average gives more weight to recent prices, making it more responsive to new information than SMA.
        - 12-day EMA: Popular for short-term trends
        - 26-day EMA: Common for medium-term analysis
        
        *EMAs react faster to price changes than SMAs.*
        """)
        ema_periods = st.multiselect(
            "Select EMA periods (days):",
            options=[12, 26, 50, 200],
            default=[12, 26],
            help="EMA gives more weight to recent prices. Common periods are 12 and 26 days"
        )

        # Bollinger Bands
        st.write("##### Bollinger Bands")
        st.markdown("""
        **What are Bollinger Bands?**  
        Bollinger Bands consist of three lines:
        - Middle Band: 20-day SMA
        - Upper Band: SMA + (Standard Deviation × 2)
        - Lower Band: SMA - (Standard Deviation × 2)
        
        *When prices move outside the bands, it may indicate overbought or oversold conditions.*
        """)
        bb_period = st.slider(
            "Bollinger Bands period:",
            min_value=5,
            max_value=50,
            value=20,
            help="Standard period is 20 days"
        )
        bb_stddev = st.slider(
            "Number of standard deviations:",
            min_value=1,
            max_value=3,
            value=2,
            help="Standard is 2 standard deviations"
        )

        # ROC
        st.write("##### Price Rate of Change (ROC)")
        st.markdown("""
        **What is ROC?**  
        Rate of Change measures the percentage change in price between the current price and the price n periods ago.
        - Positive ROC: Price is higher than n periods ago
        - Negative ROC: Price is lower than n periods ago
        
        *Extreme ROC values might indicate overbought or oversold conditions.*
        """)
        roc_period = st.slider(
            "ROC period (days):",
            min_value=1,
            max_value=100,
            value=14,
            help="Measures price change over specified period"
        )

        # Momentum
        st.write("##### Price Momentum")
        st.markdown("""
        **What is Price Momentum?**  
        Momentum measures the speed of price change by comparing current price to a past price.
        - Positive Momentum: Upward price pressure
        - Negative Momentum: Downward price pressure
        
        *Strong momentum in either direction might indicate trend continuation.*
        """)
        momentum_period = st.slider(
            "Momentum period (days):",
            min_value=1,
            max_value=100,
            value=10,
            help="Measures price momentum over specified period"
        )

        if st.button("Calculate Selected Indicators"):
            with st.spinner("Calculating technical indicators..."):
                df_indicators = df

                # Calculate SMAs
                if sma_periods:
                    for period in sma_periods:
                        window_spec = (
                            Window.orderBy("Date")
                            .rowsBetween(-(period - 1), 0)
                        )
                        df_indicators = df_indicators.withColumn(
                            f"SMA_{period}",
                            avg("Close").over(window_spec)
                        )

                # Calculate EMAs
                if ema_periods:
                    for period in ema_periods:
                        smoothing = 2.0 / (period + 1)
                        window_spec = Window.orderBy("Date")
                        df_indicators = df_indicators.withColumn(
                            f"EMA_{period}",
                            # EMA calculation using window functions
                            expr(f"""
                                first_value(Close) over (
                                    order by Date 
                                    rows between {period - 1} preceding and {period - 1} preceding
                                )
                            """)
                        )

                # Calculate Bollinger Bands
                if bb_period:
                    window_spec = (
                        Window.orderBy("Date")
                        .rowsBetween(-(bb_period - 1), 0)
                    )
                    df_indicators = df_indicators.withColumn(
                        f"BB_middle_{bb_period}",
                        avg("Close").over(window_spec)
                    ).withColumn(
                        f"BB_std_{bb_period}",
                        stddev("Close").over(window_spec)
                    ).withColumn(
                        f"BB_upper_{bb_period}",
                        col(f"BB_middle_{bb_period}") + (bb_stddev * col(f"BB_std_{bb_period}"))
                    ).withColumn(
                        f"BB_lower_{bb_period}",
                        col(f"BB_middle_{bb_period}") - (bb_stddev * col(f"BB_std_{bb_period}"))
                    )

                # Calculate ROC
                if roc_period:
                    window_spec = Window.orderBy("Date")
                    df_indicators = df_indicators.withColumn(
                        f"ROC_{roc_period}",
                        ((col("Close") - lag("Close", roc_period).over(window_spec)) /
                         lag("Close", roc_period).over(window_spec) * 100)
                    )

                # Calculate Momentum
                if momentum_period:
                    window_spec = Window.orderBy("Date")
                    df_indicators = df_indicators.withColumn(
                        f"Momentum_{momentum_period}",
                        col("Close") - lag("Close", momentum_period).over(window_spec)
                    )

                # Visualize indicators
                st.subheader("Technical Indicators Visualization")
                
                # Get data for plotting
                plot_data = df_indicators.orderBy("Date").collect()
                dates = [row["Date"] for row in plot_data]
                prices = [row["Close"] for row in plot_data]
                
                # Create tabs for different indicator groups
                tab1, tab2, tab3 = st.tabs(["Moving Averages", "Bollinger Bands", "Momentum Indicators"])
                
                with tab1:
                    st.warning("""
                    ⚠️ Moving averages are lagging indicators and may not predict future movements.
                    Use in conjunction with other analysis tools.
                    """)
                    # Moving Averages Plot
                    fig_ma = go.Figure()
                    
                    # Add price
                    fig_ma.add_trace(
                        go.Scatter(
                            x=dates,
                            y=prices,
                            name="Price",
                            line=dict(color="#00B5F7", width=2)
                        )
                    )
                    
                    # Add SMAs
                    if sma_periods:
                        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
                        for period, color in zip(sma_periods, colors):
                            sma_values = [row[f"SMA_{period}"] for row in plot_data]
                            fig_ma.add_trace(
                                go.Scatter(
                                    x=dates,
                                    y=sma_values,
                                    name=f"SMA {period}",
                                    line=dict(color=color, width=1.5)
                                )
                            )
                    
                    # Add EMAs
                    if ema_periods:
                        colors = ["#FFE66D", "#96CEB4"]
                        for period, color in zip(ema_periods, colors):
                            ema_values = [row[f"EMA_{period}"] for row in plot_data]
                            fig_ma.add_trace(
                                go.Scatter(
                                    x=dates,
                                    y=ema_values,
                                    name=f"EMA {period}",
                                    line=dict(color=color, width=1.5, dash='dash')
                                )
                            )
                    
                    fig_ma.update_layout(
                        title="Price with Moving Averages",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        template="plotly_dark",
                        height=500
                    )
                    st.plotly_chart(fig_ma, use_container_width=True)
                
                with tab2:
                    st.warning("""
                    ⚠️ Bollinger Bands can give false signals in trending markets.
                    Price touching the bands does not guarantee a reversal.
                    """)
                    # Bollinger Bands Plot
                    if bb_period:
                        fig_bb = go.Figure()
                        
                        # Add price
                        fig_bb.add_trace(
                            go.Scatter(
                                x=dates,
                                y=prices,
                                name="Price",
                                line=dict(color="#00B5F7", width=2)
                            )
                        )
                        
                        # Add Bollinger Bands
                        bb_middle = [row[f"BB_middle_{bb_period}"] for row in plot_data]
                        bb_upper = [row[f"BB_upper_{bb_period}"] for row in plot_data]
                        bb_lower = [row[f"BB_lower_{bb_period}"] for row in plot_data]
                        
                        fig_bb.add_trace(
                            go.Scatter(
                                x=dates,
                                y=bb_middle,
                                name="Middle Band",
                                line=dict(color="#95A5A6", width=1)
                            )
                        )
                        
                        fig_bb.add_trace(
                            go.Scatter(
                                x=dates,
                                y=bb_upper,
                                name=f"Upper Band ({bb_stddev}σ)",
                                line=dict(color="#95A5A6", width=1, dash='dot')
                            )
                        )
                        
                        fig_bb.add_trace(
                            go.Scatter(
                                x=dates,
                                y=bb_lower,
                                name=f"Lower Band ({bb_stddev}σ)",
                                line=dict(color="#95A5A6", width=1, dash='dot'),
                                fill='tonexty'
                            )
                        )
                        
                        fig_bb.update_layout(
                            title=f"Bollinger Bands ({bb_period} periods)",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template="plotly_dark",
                            height=500
                        )
                        st.plotly_chart(fig_bb, use_container_width=True)
                    else:
                        st.info("Please select Bollinger Bands period to view this chart")
                
                with tab3:
                    st.warning("""
                    ⚠️ Momentum indicators may give false signals, especially in ranging markets.
                    Consider market context when interpreting these signals.
                    """)
                    # Momentum Indicators Plot
                    if roc_period or momentum_period:
                        fig_mom = make_subplots(rows=2, cols=1, shared_xaxes=True)
                        
                        if roc_period:
                            roc_values = [row[f"ROC_{roc_period}"] for row in plot_data]
                            fig_mom.add_trace(
                                go.Scatter(
                                    x=dates,
                                    y=roc_values,
                                    name=f"ROC ({roc_period})",
                                    line=dict(color="#F39C12")
                                ),
                                row=1, col=1
                            )
                        
                        if momentum_period:
                            momentum_values = [row[f"Momentum_{momentum_period}"] for row in plot_data]
                            fig_mom.add_trace(
                                go.Scatter(
                                    x=dates,
                                    y=momentum_values,
                                    name=f"Momentum ({momentum_period})",
                                    line=dict(color="#E74C3C")
                                ),
                                row=2, col=1
                            )
                        
                        fig_mom.update_layout(
                            title="Momentum Indicators",
                            template="plotly_dark",
                            height=600,
                            showlegend=True
                        )
                        
                        fig_mom.update_xaxes(title_text="Date", row=2, col=1)
                        fig_mom.update_yaxes(title_text="ROC (%)", row=1, col=1)
                        fig_mom.update_yaxes(title_text="Momentum", row=2, col=1)
                        
                        st.plotly_chart(fig_mom, use_container_width=True)
                    else:
                        st.info("Please select ROC or Momentum period to view this chart")

                return df_indicators

    return df


def preprocess_data(spark: SparkSession, stock: str, days: int):
    """Preprocess stock data for analysis."""
    st.title("Data Preprocessing")
    
    st.warning("""
    ⚠️ **Data Processing Notice**
    - Data cleaning may affect analysis accuracy
    - Missing value treatment can impact results
    - Processed data may not reflect real-time market conditions
    """)

    # Get the data from exploration
    from utils.stock_utils import get_stock_history
    df = get_stock_history(stock, spark, days)

    if df is None:
        return None

    # 1. Data Quality Analysis
    st.subheader("1. Data Quality Analysis")

    # Count total rows
    total_rows = df.count()
    st.write(f"Total number of records: **{total_rows}**")

    # Check for missing values
    missing_counts = []
    for column in df.columns:
        missing = df.filter(
            col(column).isNull() |
            isnan(col(column))
        ).count()
        missing_pct = (missing / total_rows) * 100
        missing_counts.append({
            "Column": column,
            "Missing Values": missing,
            "Percentage": f"{missing_pct:.2f}%"
        })

    # Display missing values
    st.write("Missing Values Analysis:")
    missing_df = spark.createDataFrame(missing_counts)
    st.dataframe(missing_df)

    # Visualize missing values
    if any(row["Missing Values"] > 0 for row in missing_counts):
        fig = go.Figure(data=[
            go.Bar(
                x=[row["Column"] for row in missing_counts],
                y=[row["Missing Values"] for row in missing_counts],
                text=[row["Percentage"] for row in missing_counts],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Column",
            yaxis_title="Number of Missing Values",
            template="plotly_dark"
        )

        st.plotly_chart(fig)

        # 2. Handle Missing Values
        st.subheader("2. Missing Values Treatment")

        method = st.radio(
            "Select how to handle missing values:",
            ["Drop incomplete rows",
             "Fill with column mean",
             "Fill with previous value"]
        )

        if st.button("Apply Missing Value Treatment"):
            with st.spinner("Processing..."):
                if method == "Drop incomplete rows":
                    df_clean = df.dropna()
                    st.success(f"Dropped {total_rows - df_clean.count()} rows with missing values")

                elif method == "Fill with column mean":
                    # Only fill numeric columns
                    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
                    df_clean = df

                    for col_name in numeric_cols:
                        if any(row["Column"] == col_name and row["Missing Values"] > 0
                               for row in missing_counts):
                            mean_val = df.select(avg(col(col_name))).first()[0]
                            df_clean = df_clean.fillna(mean_val, subset=[col_name])
                            st.info(f"Filled missing values in {col_name} with mean: {mean_val:.2f}")

                else:  # Fill with previous value
                    df_clean = df.fillna(method="previous")
                    st.success("Filled missing values with previous values")

                # Show sample of cleaned data
                st.subheader("Preview of Cleaned Data")
                st.dataframe(df_clean.limit(5).toPandas())

                return df_clean
    else:
        st.success("No missing values found in the dataset! 🎉")

    add_technical_indicators(df, spark)

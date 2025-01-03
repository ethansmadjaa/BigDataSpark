import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, isnan, avg, expr, stddev, lag
from pyspark.sql.window import Window


# TODO: refactor this huge file into smaller modules
# TODO: fix the weird bug with bollinger bands calculation
# TODO: add RSI and MACD indicators
# TODO: optimize the window calculations - too slow rn

def add_technical_indicators(df: DataFrame, spark: SparkSession):
    """Add technical indicators to the dataframe."""
    st.subheader("2. Technical Indicators")

    with st.expander("📈 Price-Based Indicators", expanded=True):
        # SMA section
        st.write("##### Simple Moving Averages (SMA)")
        st.markdown("""
            <div class="tooltip">ℹ️ About Simple Moving Average (SMA)
                <span class="tooltiptext">
                <strong>How to Use SMA:</strong><br>
                • Trend Direction: Price above SMA = Uptrend<br>
                • Support/Resistance: SMAs often act as price barriers<br>
                • Common Periods:<br>
                  - 20 days: Short-term trend<br>
                  - 50 days: Medium-term trend<br>
                  - 200 days: Long-term trend<br>
                • Golden Cross: Short SMA crosses above Long SMA (Bullish)
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        sma_prds = st.multiselect(
            "Select SMA periods (days):",
            options=[5, 10, 20, 50, 100, 200],
            default=[20, 50, 200]
        )

        # EMA section
        st.write("##### Exponential Moving Average (EMA)")
        st.markdown("""
            <div class="tooltip">ℹ️ About Exponential Moving Average (EMA)
                <span class="tooltiptext">
                <strong>EMA vs SMA:</strong><br>
                • Reacts faster to price changes<br>
                • More weight to recent prices<br>
                • Common Uses:<br>
                  - 12 & 26 day: MACD calculation<br>
                  - 9 day: Short-term signals<br>
                • Crossovers often signal trend changes
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        ema_prds = st.multiselect(
            "Select EMA periods (days):",
            options=[12, 26, 50, 200],
            default=[12, 26]
        )

        # Bollinger Bands section
        st.write("##### Bollinger Bands")
        st.markdown("""
            <div class="tooltip">ℹ️ About Bollinger Bands
                <span class="tooltiptext">
                <strong>Trading Signals:</strong><br>
                • Price near upper band: Potentially overbought<br>
                • Price near lower band: Potentially oversold<br>
                • Bands squeezing: Volatility decreasing<br>
                • Bands expanding: Volatility increasing<br>
                • Price breaking bands: Strong momentum
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        bb_prd = st.slider(
            "Bollinger Bands period:",
            min_value=5,
            max_value=50,
            value=20
        )

        bb_std = st.slider(
            "Number of standard deviations:",
            min_value=1,
            max_value=3,
            value=2
        )

        # ROC section
        st.write("##### Price Rate of Change (ROC)")
        st.markdown("""
            <div class="tooltip">ℹ️ About Rate of Change (ROC)
                <span class="tooltiptext">
                <strong>Understanding ROC:</strong><br>
                • Measures momentum and trend strength<br>
                • Positive ROC: Upward momentum<br>
                • Negative ROC: Downward momentum<br>
                • Zero line crossings: Potential trend changes<br>
                • Extreme values: Possible reversal points
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        roc_prd = st.slider(
            "ROC period (days):",
            min_value=1,
            max_value=100,
            value=14,
            help="Measures price change over specified period"
        )

        # momentum stuff
        st.write("##### Price Momentum")
        st.markdown("""
        **What is Price Momentum?**  
        Momentum measures the speed of price change by comparing current price to a past price.
        - Positive Momentum: Upward price pressure
        - Negative Momentum: Downward price pressure
        
        *Strong momentum in either direction might indicate trend continuation.*
        """)
        mom_prd = st.slider(
            "Momentum period (days):",
            min_value=1,
            max_value=100,
            value=10,
            help="Measures price momentum over specified period"
        )

        # calculate everything when user clicks
        if st.button("Calculate Selected Indicators"):
            with st.spinner("Calculating technical indicators..."):
                df_indic = df

                # calc SMAs
                if sma_prds:
                    for prd in sma_prds:
                        win_spec = (
                            Window.orderBy("Date")
                            .rowsBetween(-(prd - 1), 0)
                        )
                        df_indic = df_indic.withColumn(
                            f"SMA_{prd}",
                            avg("Close").over(win_spec)
                        )

                # calc EMAs - bit hacky but works
                if ema_prds:
                    for prd in ema_prds:
                        smooth = 2.0 / (prd + 1)
                        win_spec = Window.orderBy("Date")
                        df_indic = df_indic.withColumn(
                            f"EMA_{prd}",
                            expr(f"""
                                first_value(Close) over (
                                    order by Date 
                                    rows between {prd - 1} preceding and {prd - 1} preceding
                                )
                            """)
                        )

                # calc bollinger - check if calcs are correct
                if bb_prd:
                    win_spec = (
                        Window.orderBy("Date")
                        .rowsBetween(-(bb_prd - 1), 0)
                    )
                    df_indic = df_indic.withColumn(
                        f"BB_middle_{bb_prd}",
                        avg("Close").over(win_spec)
                    ).withColumn(
                        f"BB_std_{bb_prd}",
                        stddev("Close").over(win_spec)
                    ).withColumn(
                        f"BB_upper_{bb_prd}",
                        col(f"BB_middle_{bb_prd}") + (bb_std * col(f"BB_std_{bb_prd}"))
                    ).withColumn(
                        f"BB_lower_{bb_prd}",
                        col(f"BB_middle_{bb_prd}") - (bb_std * col(f"BB_std_{bb_prd}"))
                    )

                # calc ROC
                if roc_prd:
                    win_spec = Window.orderBy("Date")
                    df_indic = df_indic.withColumn(
                        f"ROC_{roc_prd}",
                        ((col("Close") - lag("Close", roc_prd).over(win_spec)) /
                         lag("Close", roc_prd).over(win_spec) * 100)
                    )

                # calc momentum
                if mom_prd:
                    win_spec = Window.orderBy("Date")
                    df_indic = df_indic.withColumn(
                        f"Momentum_{mom_prd}",
                        col("Close") - lag("Close", mom_prd).over(win_spec)
                    )

                # viz stuff
                st.subheader("Technical Indicators Visualization")

                # get data ready for plots
                plot_data = df_indic.orderBy("Date").collect()
                dates = [row["Date"] for row in plot_data]
                prices = [row["Close"] for row in plot_data]

                # tabs for different indicators
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
                    if sma_prds:
                        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
                        for prd, color in zip(sma_prds, colors):
                            sma_values = [row[f"SMA_{prd}"] for row in plot_data]
                            fig_ma.add_trace(
                                go.Scatter(
                                    x=dates,
                                    y=sma_values,
                                    name=f"SMA {prd}",
                                    line=dict(color=color, width=1.5)
                                )
                            )

                    # Add EMAs
                    if ema_prds:
                        colors = ["#FFE66D", "#96CEB4"]
                        for prd, color in zip(ema_prds, colors):
                            ema_values = [row[f"EMA_{prd}"] for row in plot_data]
                            fig_ma.add_trace(
                                go.Scatter(
                                    x=dates,
                                    y=ema_values,
                                    name=f"EMA {prd}",
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
                    if bb_prd:
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
                        bb_middle = [row[f"BB_middle_{bb_prd}"] for row in plot_data]
                        bb_upper = [row[f"BB_upper_{bb_prd}"] for row in plot_data]
                        bb_lower = [row[f"BB_lower_{bb_prd}"] for row in plot_data]

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
                                name=f"Upper Band ({bb_std}σ)",
                                line=dict(color="#95A5A6", width=1, dash='dot')
                            )
                        )

                        fig_bb.add_trace(
                            go.Scatter(
                                x=dates,
                                y=bb_lower,
                                name=f"Lower Band ({bb_std}��)",
                                line=dict(color="#95A5A6", width=1, dash='dot'),
                                fill='tonexty'
                            )
                        )

                        fig_bb.update_layout(
                            title=f"Bollinger Bands ({bb_prd} periods)",
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
                    if roc_prd or mom_prd:
                        fig_mom = make_subplots(rows=2, cols=1, shared_xaxes=True)

                        if roc_prd:
                            roc_values = [row[f"ROC_{roc_prd}"] for row in plot_data]
                            fig_mom.add_trace(
                                go.Scatter(
                                    x=dates,
                                    y=roc_values,
                                    name=f"ROC ({roc_prd})",
                                    line=dict(color="#F39C12")
                                ),
                                row=1, col=1
                            )

                        if mom_prd:
                            momentum_values = [row[f"Momentum_{mom_prd}"] for row in plot_data]
                            fig_mom.add_trace(
                                go.Scatter(
                                    x=dates,
                                    y=momentum_values,
                                    name=f"Momentum ({mom_prd})",
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

                return df_indic

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

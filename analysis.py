import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lag, stddev, avg, min, max, 
    expr, sum, when, sqrt, corr
)
from pyspark.sql.window import Window
from utils.risk_warnings import show_specific_warning
import math


def analyze_data(spark_session: SparkSession, stock: str, days: int):
    st.title("Data Analysis")
    
    # Get data
    from utils.stock_utils import get_stock_history
    df = get_stock_history(stock, spark_session, days)
    
    if df is None:
        return
    
    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["Risk Analysis", "Coming Soon Features"])
    
    with tab1:
        st.subheader("Risk Analysis")
        show_specific_warning("risk_analysis")  # We'll add this warning type
        
        # Calculate daily returns
        window_spec = Window.orderBy("Date")
        df = df.withColumn(
            "daily_return",
            ((col("Close") - lag("Close", 1).over(window_spec)) / 
             lag("Close", 1).over(window_spec) * 100)
        )
        
        # 1. Volatility Measurement
        st.write("##### 1. Volatility Analysis")
        st.markdown("""
        Volatility measures the degree of variation in the stock's returns. 
        Higher volatility indicates higher risk.
        """)
        
        # Calculate annualized volatility
        volatility = df.select(stddev("daily_return")).first()[0]
        if volatility:
            annualized_vol = volatility * math.sqrt(252)  # 252 trading days
            st.metric(
                "Annualized Volatility",
                f"{annualized_vol:.2f}%",
                help="Higher values indicate more price swings and potentially higher risk"
            )
        
        # 2. Maximum Drawdown
        st.write("##### 2. Maximum Drawdown")
        st.markdown("""
        Maximum drawdown measures the largest peak-to-trough decline in the stock's price.
        It helps understand potential downside risk.
        """)
        
        # Calculate running maximum and drawdown
        df = df.withColumn(
            "running_max",
            expr("max(Close) over (order by Date rows between unbounded preceding and current row)")
        ).withColumn(
            "drawdown",
            ((col("Close") - col("running_max")) / col("running_max")) * 100
        )
        
        max_drawdown = df.select(min("drawdown")).first()[0]
        if max_drawdown:
            st.metric(
                "Maximum Drawdown",
                f"{max_drawdown:.2f}%",
                help="The worst decline from peak to trough"
            )
            
            # Visualize drawdown
            drawdown_data = df.select("Date", "drawdown").collect()
            fig_drawdown = go.Figure()
            fig_drawdown.add_trace(
                go.Scatter(
                    x=[row["Date"] for row in drawdown_data],
                    y=[row["drawdown"] for row in drawdown_data],
                    fill='tozeroy',
                    name="Drawdown"
                )
            )
            fig_drawdown.update_layout(
                title="Historical Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_drawdown, use_container_width=True)
        
        # 3. Value at Risk (VaR)
        st.write("##### 3. Value at Risk (VaR)")
        st.markdown("""
        VaR estimates the potential loss in value of the stock over a defined period.
        We calculate both 95% and 99% confidence levels.
        """)
        
        # Calculate VaR
        returns_data = df.select("daily_return").collect()
        returns = [row["daily_return"] for row in returns_data if row["daily_return"] is not None]
        if returns:
            returns.sort()
            var_95 = returns[int(len(returns) * 0.05)]
            var_99 = returns[int(len(returns) * 0.01)]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "95% VaR",
                    f"{abs(var_95):.2f}%",
                    help="Maximum daily loss with 95% confidence"
                )
            with col2:
                st.metric(
                    "99% VaR",
                    f"{abs(var_99):.2f}%",
                    help="Maximum daily loss with 99% confidence"
                )
        
        # 4. Return Distribution
        st.write("##### 4. Return Distribution")
        st.markdown("""
        The distribution of returns helps understand the range and frequency of price changes.
        A wider distribution indicates higher volatility.
        """)
        
        # Create return distribution plot
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name="Daily Returns"
            )
        )
        fig_dist.update_layout(
            title="Distribution of Daily Returns",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template="plotly_dark"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with tab2:
        st.subheader("🚀 Upcoming Features")
        
        st.markdown("""
        ### 📊 Statistical Analysis
        Advanced statistical measures to better understand stock behavior:
        - Distribution characteristics and normality tests
        - Skewness and kurtosis analysis
        - Advanced outlier detection
        - Statistical hypothesis testing
        
        ### 🤖 Machine Learning Integration
        AI-powered analysis tools:
        - Price prediction models
        - Trend classification
        - Anomaly detection
        - Stock clustering and similarity analysis
        
        ### 📰 Sentiment Analysis
        Understanding market sentiment through:
        - News sentiment impact analysis
        - Social media mentions tracking
        - Market sentiment indicators
        - Real-time sentiment scoring
        
        ---
        
        > These features are under development as part of our ongoing effort to provide
        > comprehensive stock analysis tools. Stay tuned for updates!
        
        ⚠️ *Note: All future features will include appropriate risk warnings and educational context.*
        """)
        
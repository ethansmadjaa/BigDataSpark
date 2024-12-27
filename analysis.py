import streamlit as st
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lag, stddev, avg, min, max, 
    expr, sum
)
from pyspark.sql.window import Window
from utils.risk_warnings import show_specific_warning
import math
import numpy as np
from scipy import stats as scipy_stats


def analyze_data(spark_session: SparkSession, stock: str, days: int):
    st.title("Data Analysis")
    
    # Get data
    from utils.stock_utils import get_stock_history
    df = get_stock_history(stock, spark_session, days)
    
    if df is None:
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Statistical Analysis", "Coming Soon Features"])
    
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
        st.subheader("Statistical Analysis")
        show_specific_warning("statistical")  # We'll add this warning type
        
        # Calculate daily returns if not already done
        if "daily_return" not in df.columns:
            window_spec = Window.orderBy("Date")
            df = df.withColumn(
                "daily_return",
                ((col("Close") - lag("Close", 1).over(window_spec)) / 
                 lag("Close", 1).over(window_spec) * 100)
            )
        
        # 1. Basic Statistics
        st.write("##### 1. Return Statistics")
        
        # Calculate basic statistics
        stats = df.select(
            avg("daily_return").alias("mean"),
            stddev("daily_return").alias("std"),
            expr("skewness(daily_return)").alias("skewness"),
            expr("kurtosis(daily_return)").alias("kurtosis")
        ).collect()[0]
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Mean Return",
                f"{stats['mean']:.2f}%",
                help="Average daily return"
            )
        with col2:
            st.metric(
                "Std Deviation",
                f"{stats['std']:.2f}%",
                help="Measure of return volatility"
            )
        with col3:
            st.metric(
                "Skewness",
                f"{stats['skewness']:.2f}",
                help="Measure of return distribution asymmetry"
            )
        with col4:
            st.metric(
                "Kurtosis",
                f"{stats['kurtosis']:.2f}",
                help="Measure of tail extremity"
            )
            
        # 2. Normality Analysis
        st.write("##### 2. Return Distribution Analysis")
        st.markdown("""
        Comparing the actual return distribution to a normal distribution helps understand
        if returns follow typical market assumptions.
        """)
        
        # Get returns data
        returns_data = df.select("daily_return").collect()
        returns = [row["daily_return"] for row in returns_data if row["daily_return"] is not None]
        
        if returns:
            
            # Create QQ plot
            fig_qq = go.Figure()
            
            # Calculate theoretical quantiles
            theoretical_q = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
            returns_sorted = np.sort(returns)
            
            # Add QQ plot
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_q,
                    y=returns_sorted,
                    mode='markers',
                    name='Returns',
                    marker=dict(color='#00B5F7')
                )
            )
            
            # Add diagonal line
            min_val = float(np.min([np.min(theoretical_q), np.min(returns_sorted)]))
            max_val = float(np.max([np.max(theoretical_q), np.max(returns_sorted)]))
            
            fig_qq.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Normal',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig_qq.update_layout(
                title="Q-Q Plot (Returns vs Normal Distribution)",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_qq, use_container_width=True)
            
            # Perform normality tests
            st.write("##### 3. Normality Tests")
            
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = scipy_stats.shapiro(returns)
            # Anderson-Darling test
            anderson_result = scipy_stats.anderson(returns)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Shapiro-Wilk p-value",
                    f"{shapiro_p:.4f}",
                    help="p < 0.05 suggests non-normal distribution"
                )
            with col2:
                st.metric(
                    "Anderson-Darling Statistic",
                    f"{anderson_result.statistic:.4f}",
                    help="Higher values suggest non-normal distribution"
                )
            
            # 4. Outlier Analysis
            st.write("##### 4. Outlier Detection")
            
            # Calculate z-scores
            z_scores = scipy_stats.zscore(returns)
            outliers = np.abs(z_scores) > 3
            
            # Create outlier plot
            fig_out = go.Figure()
            
            # Add regular points
            dates = [row["Date"] for row in df.select("Date").collect()]
            
            fig_out.add_trace(
                go.Scatter(
                    x=dates,
                    y=returns,
                    mode='markers',
                    name='Normal Returns',
                    marker=dict(color='#00B5F7', size=6)
                )
            )
            
            # Add outliers
            outlier_dates = [date for date, is_out in zip(dates, outliers) if is_out]
            outlier_returns = [ret for ret, is_out in zip(returns, outliers) if is_out]
            
            fig_out.add_trace(
                go.Scatter(
                    x=outlier_dates,
                    y=outlier_returns,
                    mode='markers',
                    name='Outliers',
                    marker=dict(color='red', size=8, symbol='x')
                )
            )
            
            fig_out.update_layout(
                title="Return Outliers (|z-score| > 3)",
                xaxis_title="Date",
                yaxis_title="Daily Return (%)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_out, use_container_width=True)
            
            # Display outlier statistics using numpy's sum instead of PySpark's
            n_outliers = np.sum(outliers)
            st.metric(
                "Number of Outliers",
                int(n_outliers),
                f"{(n_outliers/len(returns)*100):.1f}% of returns",
                help="Days with returns > 3 standard deviations from mean"
            )
    
    with tab3:
        st.subheader("🚀 Upcoming Features")
        
        st.markdown("""
        
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
        
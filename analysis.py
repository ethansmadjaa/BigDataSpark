import streamlit as st
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, stddev, avg, min, max, expr, sum
from pyspark.sql.window import Window
from utils.risk_warnings import show_specific_warning
import math
import numpy as np
from scipy import stats as scipy_stats


def calculate_daily_returns(df):
    """
    Calculate daily returns for the stock.
    
    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame containing stock data with columns: Date, Close
        
    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with additional column 'daily_return' showing percentage change
        
    Notes
    -----
    - Returns are calculated as: ((Close_t - Close_t-1) / Close_t-1) * 100
    - First day's return will be NULL due to no previous close price
    """
    window_spec = Window.orderBy("Date")
    return df.withColumn(
        "daily_return",
        ((col("Close") - lag("Close", 1).over(window_spec)) / 
         lag("Close", 1).over(window_spec) * 100)
    )


def analyze_volatility(df):
    """
    Analyze stock volatility using standard deviation of returns.
    
    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame containing 'daily_return' column
        
    Notes
    -----
    - Annualized volatility = Daily volatility * sqrt(252)
    - 252 is used as the number of trading days in a year
    - Higher volatility indicates higher risk
    
    Displays
    --------
    - Annualized volatility metric
    - Explanation of volatility interpretation
    """
    st.write("##### 1. Volatility Analysis")
    st.markdown("""
    Volatility measures the degree of variation in the stock's returns. 
    Higher volatility indicates higher risk.
    """)
    
    volatility = df.select(stddev("daily_return")).first()[0]
    if volatility:
        annualized_vol = volatility * math.sqrt(252)
        st.metric(
            "Annualized Volatility",
            f"{annualized_vol:.2f}%",
            help="Higher values indicate more price swings and potentially higher risk"
        )


def analyze_drawdown(df):
    """
    Calculate and visualize maximum drawdown from peak prices.
    
    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame with columns: Date, Close
        
    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with additional columns: running_max, drawdown
        
    Notes
    -----
    - Drawdown measures decline from peak value
    - Calculated as: ((Current_Price - Peak_Price) / Peak_Price) * 100
    - Helps assess downside risk and worst historical losses
    
    Displays
    --------
    - Maximum drawdown percentage
    - Visual representation of drawdown over time
    """
    st.write("##### 2. Maximum Drawdown")
    st.markdown("""
    Maximum drawdown measures the largest peak-to-trough decline in the stock's price.
    It helps understand potential downside risk.
    """)
    
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
        
        plot_drawdown(df)
    
    return df


def plot_drawdown(df):
    """Plot drawdown visualization."""
    drawdown_data = df.select("Date", "drawdown").collect()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[row["Date"] for row in drawdown_data],
            y=[row["drawdown"] for row in drawdown_data],
            fill='tozeroy',
            name="Drawdown"
        )
    )
    fig.update_layout(
        title="Historical Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


def calculate_var(returns):
    """
    Calculate Value at Risk (VaR) at different confidence levels.
    
    Parameters
    ----------
    returns : list
        List of daily return percentages
        
    Notes
    -----
    - VaR 95%: Loss threshold that should only be exceeded 5% of the time
    - VaR 99%: More conservative threshold, exceeded 1% of the time
    - Historical VaR method used (no distribution assumptions)
    
    Warning
    -------
    VaR is backward-looking and may not capture future risks accurately.
    Extreme events can exceed VaR estimates.
    
    Displays
    --------
    - 95% VaR metric
    - 99% VaR metric
    - Explanation of VaR interpretation
    """
    st.write("##### 3. Value at Risk (VaR)")
    st.markdown("""
    VaR estimates the potential loss in value of the stock over a defined period.
    We calculate both 95% and 99% confidence levels.
    """)
    
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


def plot_return_distribution(returns):
    """Plot return distribution."""
    st.write("##### 4. Return Distribution")
    st.markdown("""
    The distribution of returns helps understand the range and frequency of price changes.
    A wider distribution indicates higher volatility.
    """)
    
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name="Daily Returns"
        )
    )
    fig.update_layout(
        title="Distribution of Daily Returns",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


def analyze_basic_statistics(df):
    """Analyze basic statistical measures."""
    st.write("##### 1. Return Statistics")
    
    stats = df.select(
        avg("daily_return").alias("mean"),
        stddev("daily_return").alias("std"),
        expr("skewness(daily_return)").alias("skewness"),
        expr("kurtosis(daily_return)").alias("kurtosis")
    ).collect()[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Return", f"{stats['mean']:.2f}%", help="Average daily return")
    with col2:
        st.metric("Std Deviation", f"{stats['std']:.2f}%", help="Measure of return volatility")
    with col3:
        st.metric("Skewness", f"{stats['skewness']:.2f}", help="Measure of return distribution asymmetry")
    with col4:
        st.metric("Kurtosis", f"{stats['kurtosis']:.2f}", help="Measure of tail extremity")


def analyze_normality(returns):
    """
    Perform comprehensive normality analysis on returns.
    
    Parameters
    ----------
    returns : list
        List of daily return percentages
        
    Notes
    -----
    - Q-Q plot compares returns to theoretical normal distribution
    - Shapiro-Wilk test for normality (null hypothesis: data is normal)
    - Anderson-Darling test provides additional confirmation
    
    Statistical Tests
    ----------------
    - Shapiro-Wilk: p < 0.05 suggests non-normal distribution
    - Anderson-Darling: Higher statistic values indicate non-normality
    
    Warning
    -------
    Financial returns often deviate from normal distribution,
    showing fat tails and higher kurtosis.
    """
    st.write("##### 2. Return Distribution Analysis")
    st.markdown("""
    Comparing the actual return distribution to a normal distribution helps understand
    if returns follow typical market assumptions.
    """)
    
    plot_qq(returns)
    perform_normality_tests(returns)


def plot_qq(returns):
    """Create Q-Q plot."""
    fig = go.Figure()
    
    theoretical_q = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
    returns_sorted = np.sort(returns)
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_q,
            y=returns_sorted,
            mode='markers',
            name='Returns',
            marker=dict(color='#00B5F7')
        )
    )
    
    min_val = float(np.min([np.min(theoretical_q), np.min(returns_sorted)]))
    max_val = float(np.max([np.max(theoretical_q), np.max(returns_sorted)]))
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Normal',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(
        title="Q-Q Plot (Returns vs Normal Distribution)",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def perform_normality_tests(returns):
    """Perform statistical normality tests."""
    st.write("##### 3. Normality Tests")
    
    shapiro_stat, shapiro_p = scipy_stats.shapiro(returns)
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


def analyze_outliers(df, returns):
    """Analyze return outliers."""
    st.write("##### 4. Outlier Detection")
    
    z_scores = scipy_stats.zscore(returns)
    outliers = np.abs(z_scores) > 3
    
    plot_outliers(df, returns, outliers)
    
    n_outliers = np.sum(outliers)
    st.metric(
        "Number of Outliers",
        int(n_outliers),
        f"{(n_outliers/len(returns)*100):.1f}% of returns",
        help="Days with returns > 3 standard deviations from mean"
    )


def plot_outliers(df, returns, outliers):
    """Plot outliers visualization."""
    fig = go.Figure()
    
    dates = [row["Date"] for row in df.select("Date").collect()]
    
    # Add regular points
    fig.add_trace(
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
    
    fig.add_trace(
        go.Scatter(
            x=outlier_dates,
            y=outlier_returns,
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=8, symbol='x')
        )
    )
    
    fig.update_layout(
        title="Return Outliers (|z-score| > 3)",
        xaxis_title="Date",
        yaxis_title="Daily Return (%)",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_upcoming_features():
    """Display upcoming features."""
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


def analyze_data(spark_session: SparkSession, stock: str, days: int):
    """
    Main analysis function coordinating all analysis components.
    
    Parameters
    ----------
    spark_session : SparkSession
        Active Spark session for data processing
    stock : str
        Stock ticker symbol (e.g., 'AAPL')
    days : int
        Number of historical days to analyze
        
    Analysis Components
    ------------------
    1. Risk Analysis:
       - Volatility
       - Maximum Drawdown
       - Value at Risk
       - Return Distribution
       
    2. Statistical Analysis:
       - Basic Statistics
       - Normality Tests
       - Outlier Detection
       
    Notes
    -----
    - Each analysis component includes relevant risk warnings
    - All visualizations use dark theme for consistency
    - Data is processed using PySpark for scalability
    
    Warning
    -------
    All analyses are based on historical data and should not be used
    as the sole basis for investment decisions.
    """
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
        show_specific_warning("risk_analysis")
        
        df = calculate_daily_returns(df)
        analyze_volatility(df)
        df = analyze_drawdown(df)
        
        returns_data = df.select("daily_return").collect()
        returns = [row["daily_return"] for row in returns_data if row["daily_return"] is not None]
        
        if returns:
            calculate_var(returns)
            plot_return_distribution(returns)
    
    with tab2:
        st.subheader("Statistical Analysis")
        show_specific_warning("statistical")
        
        if "daily_return" not in df.columns:
            df = calculate_daily_returns(df)
        
        analyze_basic_statistics(df)
        
        returns_data = df.select("daily_return").collect()
        returns = [row["daily_return"] for row in returns_data if row["daily_return"] is not None]
        
        if returns:
            analyze_normality(returns)
            analyze_outliers(df, returns)
    
    with tab3:
        show_upcoming_features()
        
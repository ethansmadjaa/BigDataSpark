import streamlit as st
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, stddev, avg, min, max, expr, sum
from pyspark.sql.window import Window
from utils.risk_warnings import show_specific_warning
import math
import numpy as np
from scipy import stats as scipy_stats
from utils.stock_utils import get_stock_history


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


def calculate_drawdown(df):
    """Calculate drawdown without displaying."""
    window_spec = Window.orderBy("Date")
    df = df.withColumn(
        "rolling_max",
        max("Close").over(window_spec)
    ).withColumn(
        "drawdown",
        ((col("Close") - col("rolling_max")) / col("rolling_max")) * 100
    )
    return df


def analyze_drawdown(df, display=True):
    """Analyze drawdown with optional display."""
    df = calculate_drawdown(df)
    
    if display:
        st.write("##### 2. Maximum Drawdown Analysis")
        st.markdown("""
        Maximum drawdown measures the largest peak-to-trough decline, showing potential downside risk.
        """)
        
        max_drawdown = df.select(min("drawdown")).first()[0]
        st.metric(
            "Maximum Drawdown",
            f"{max_drawdown:.2f}%",
            help="Largest peak-to-trough decline. Lower values indicate higher historical risk."
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
    st.plotly_chart(fig, use_container_width=True, key="drawdown_plot")


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
    st.plotly_chart(fig, use_container_width=True, key="return_dist_plot")


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
    
    st.plotly_chart(fig, use_container_width=True, key="qq_plot")


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
    
    st.plotly_chart(fig, use_container_width=True, key="outliers_plot")


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


def get_trading_signal(df, returns):
    """Generate trading signal based on multiple analysis components."""
    df = calculate_drawdown(df)
    signals = {}
    
    # 1. Trend Analysis (40%)
    trend_score = 0
    
    # Volatility check
    volatility = df.select(stddev("daily_return")).first()[0]
    if volatility:
        annualized_vol = volatility * math.sqrt(252)
        trend_score += 20 if annualized_vol < 30 else -20
        signals['volatility'] = annualized_vol
    
    # Drawdown check
    max_drawdown = df.select(min("drawdown")).first()[0]
    if max_drawdown:
        trend_score += 20 if max_drawdown > -20 else -20
        signals['max_drawdown'] = max_drawdown
    
    # 2. Statistical Analysis (30%)
    stats_score = 0
    
    stats = df.select(
        avg("daily_return").alias("mean"),
        stddev("daily_return").alias("std"),
        expr("skewness(daily_return)").alias("skewness")
    ).collect()[0]
    
    mean_return = stats["mean"]
    stats_score += 15 if mean_return > 0 else -15
    signals['mean_return'] = mean_return
    
    skewness = stats["skewness"]
    stats_score += 15 if skewness > 0 else -15
    signals['skewness'] = skewness
    
    # 3. Risk Assessment (30%)
    risk_score = 0
    
    # VaR check
    var_95 = np.percentile(returns, 5)  # Using numpy instead of sorting
    risk_score += 15 if abs(var_95) < 3 else -15
    signals['var_95'] = var_95
    
    # Recent performance using numpy
    recent_returns = np.array(returns[-5:])  # Last 5 days
    avg_recent_return = np.mean(recent_returns)
    risk_score += 15 if avg_recent_return > 0 else -15
    signals['recent_performance'] = float(avg_recent_return)  # Convert from numpy type
    
    # Calculate final score
    final_score = trend_score * 0.4 + stats_score * 0.3 + risk_score * 0.3
    
    # Generate recommendation
    if final_score > 30:
        recommendation = "Strong Buy"
        color = "green"
    elif final_score > 0:
        recommendation = "Buy"
        color = "lightgreen"
    elif final_score > -30:
        recommendation = "Hold"
        color = "yellow"
    else:
        recommendation = "Sell"
        color = "red"
    
    return {
        'score': final_score,
        'recommendation': recommendation,
        'color': color,
        'signals': signals
    }


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
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Analysis", "Statistical Analysis", 
                                     "Trading Signal", "Coming Soon Features"])
    
    # Get data and calculate returns first as they're needed by multiple tabs
    df = get_stock_history(stock, spark_session, days)
    if df is None:
        return
        
    df = calculate_daily_returns(df)
    returns_data = df.select("daily_return").collect()
    returns = [row["daily_return"] for row in returns_data if row["daily_return"] is not None]
    
    with tab1:
        st.subheader("Risk Analysis")
        show_specific_warning("risk_analysis")
        
        df = calculate_daily_returns(df)
        analyze_volatility(df)
        df = analyze_drawdown(df, display=True)
        
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
        st.subheader("Trading Signal Analysis")
        show_specific_warning("trading_signal")
        
        st.markdown("""
        This trading signal is based on a combination of:
        - Trend Analysis (40%): Volatility and drawdown patterns
        - Statistical Analysis (30%): Mean returns and distribution characteristics
        - Risk Assessment (30%): VaR and recent performance
        """)
        
        if st.button("Generate Trading Signal", type="primary"):
            with st.spinner("Analyzing market conditions..."):
                # Use calculate_drawdown instead of analyze_drawdown
                df = calculate_drawdown(df)
                signal = get_trading_signal(df, returns)
                
                # Display recommendation
                st.markdown(f"""
                <div style='padding: 20px; border-radius: 5px; background-color: {signal['color']}20;
                           border: 2px solid {signal['color']}; margin: 10px 0;'>
                    <h2 style='color: {signal['color']}; margin: 0;'>
                        Recommendation: {signal['recommendation']}
                    </h2>
                    <p style='margin: 10px 0 0 0;'>Overall Score: {(signal['score'] + 100)/2:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display analysis components
                st.subheader("Analysis Components")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Annualized Volatility", 
                             f"{signal['signals']['volatility']:.2f}%")
                    st.metric("Maximum Drawdown", 
                             f"{signal['signals']['max_drawdown']:.2f}%")
                    
                with col2:
                    st.metric("Mean Daily Return", 
                             f"{signal['signals']['mean_return']:.2f}%")
                    st.metric("Return Skewness", 
                             f"{signal['signals']['skewness']:.2f}")
                    
                with col3:
                    st.metric("95% VaR", 
                             f"{abs(signal['signals']['var_95']):.2f}%")
                    st.metric("Recent Performance", 
                             f"{signal['signals']['recent_performance']:.2f}%")
    
    with tab4:
        show_upcoming_features()
    
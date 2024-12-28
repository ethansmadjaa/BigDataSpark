import streamlit as st

def show_risk_warning():
    """Display a prominent risk warning."""
    with st.sidebar.expander("⚠️ Risk Warning", expanded=True):
        st.markdown("""
        ### Important Risk Disclaimer
        
        **This is an educational project and should not be used for actual trading decisions.**
        
        Key Risk Factors:
        - Market risk: Stock prices can go down as well as up
        - Technical analysis limitations: Past patterns may not predict future movements
        - Data reliability: Real-time data may have delays or inaccuracies
        - Educational purpose: This tool is not financial advice
        
        Always:
        - Do your own research
        - Consider consulting financial professionals
        - Understand the risks involved in stock trading
        - Use multiple sources of information
        
        *By using this tool, you acknowledge these risks and limitations.*
        """) 

def show_specific_warning(warning_type: str):
    """Display context-specific risk warnings."""
    warnings = {
        "technical": """
        ⚠️ **Technical Analysis Risk**
        - Indicators may provide false or misleading signals
        - Market conditions can change unexpectedly
        - Technical analysis is one of many tools and should not be used alone
        """,
        "historical": """
        ⚠️ **Historical Data Risk**
        - Past performance does not predict future results
        - Market conditions change constantly
        - Historical patterns may not repeat
        """,
        "data_quality": """
        ⚠️ **Data Quality Notice**
        - Data may contain delays or inaccuracies
        - Some data points might be missing or incorrect
        - Real-time trading conditions may differ
        """,
        "risk_analysis": """
        ⚠️ **Risk Analysis Warning**
        - Past volatility does not guarantee future volatility
        - VaR calculations are estimates and may not capture extreme events
        - Maximum drawdown shows historical worst case but larger drops are possible
        - Risk metrics should not be used in isolation for investment decisions
        """,
        "statistical": """
        ⚠️ **Statistical Analysis Warning**
        - Statistical measures are based on historical data
        - Assumptions about normal distributions may not hold
        - Outliers can occur more frequently than expected
        - Past statistical patterns may not predict future behavior
        """,
        "trading_signal": """
        ⚠️ **Trading Signal Warning**
        - This signal is for educational purposes only
        - Based on historical data which may not predict future performance
        - Multiple factors should be considered before making investment decisions
        - Consult with financial professionals before trading
        - Market conditions can change rapidly
        """
    }
    return st.warning(warnings.get(warning_type, "")) 
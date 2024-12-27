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
        """
    }
    return st.warning(warnings.get(warning_type, "")) 
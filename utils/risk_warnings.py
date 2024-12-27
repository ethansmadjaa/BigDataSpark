import streamlit as st

def show_risk_warning():
    """Display a prominent risk warning."""
    with st.sidebar.expander("⚠️ Risk Warning", expanded=False):
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
import streamlit as st


# TODO: add more specific warnings for different market conditions
# TODO: translate warnings in other languages
# TODO: fix the weird formatting in sidebar
# TODO: make warnings less boring maybe add some emojis

def show_risk_warning():
    """Show the big warning in sidebar - legal stuff."""
    with st.sidebar.expander("⚠️ Risk Warning", expanded=True):
        st.markdown("""
        ### ⚠️ Hey, Listen Up!
        
        **This is just for learning! Don't use it for real trading!!**
        
        Stuff that can go wrong:
        - Markets are crazy: stonks go up AND down 📈📉
        - Technical analysis isn't magic: patterns don't always work
        - Data might be wrong or delayed (yeah it happens)
        - We're not financial advisors lol
        
        What you should do:
        - Do ur own research (srsly)
        - Talk to actual pros who know stuff
        - Know what you're getting into
        - Check multiple sources (not just us)
        
        *If you use this, you're on your own! Don't blame us if things go south.*
        """)


def show_specific_warning(warn_type: str):
    """Show diff warnings depending on what user's doing."""
    # dict of warnings - might need to add more later
    warnings = {
        "technical": """
        ⚠️ **Watch Out with Technical Analysis**
        - Indicators can lie to you sometimes
        - Markets do whatever they want
        - Don't just trust the lines on charts
        """,

        "historical": """
        ⚠️ **Past Performance = Not Future Results**
        - Just cuz it happened before doesn't mean it'll happen again
        - Markets change all the time
        - History doesn't always repeat (but it rhymes)
        """,

        "data_quality": """
        ⚠️ **Data Might Be Sus**
        - Could be delays or errors
        - Some data points might be missing
        - Real market might be different rn
        """,

        "risk_analysis": """
        ⚠️ **Risk Analysis - Handle with Care**
        - Past volatility doesn't predict future craziness
        - VaR is just a guess (and sometimes wrong)
        - Max drawdown could get worse
        - Don't yolo based on these numbers alone
        """,

        "statistical": """
        ⚠️ **Stats Can Be Tricky**
        - All based on old data
        - Normal distribution? Markets laugh at that
        - Weird stuff happens more than math says it should
        - Past patterns might mean nothing tomorrow
        """,

        "trading_signal": """
        ⚠️ **Don't Blindly Trust These Signals**
        - This is for education only!
        - Based on past data (which might not matter)
        - Look at other stuff before trading
        - Talk to real financial pros
        - Markets can flip on you real quick
        """
    }

    # show the warning if we have one for this type
    return st.warning(warnings.get(warn_type, ""))

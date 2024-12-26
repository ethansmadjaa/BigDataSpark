import streamlit as st
import gc

from analysis import analyze_data
from exploration import explore_data
from preprocessing import preprocess_data
from utils.constants import STOCK_CATEGORIES
from utils.spark_utils import create_spark_session, cleanup_spark_cache
from utils.stock_utils import get_stock_info, format_market_cap, get_ytd_days


def main():
    st.set_page_config(
        page_title="Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
            <style>
            .main { padding: 0rem 1rem; }
            .stTabs [data-baseweb="tab-list"] { gap: 2px; }
            .stTabs [data-baseweb="tab"] { padding: 10px 20px; }


            .stock-info {
                background-color: #262730;
                border-radius: 5px;
                padding: 15px;
                margin: 10px 0;
                border: 1px solid #464B5C;
            }
            .stock-info h4 {
                color: #FFFFFF;
                margin: 0 0 10px 0;
                font-size: 1.1em;
            }
            .stock-info p {
                color: #FAFAFA;
                margin: 5px 0;
                font-size: 0.9em;
            }
            .stock-info .highlight {
                color: #00CC96;
                font-weight: bold;
            }
            .stock-info .label {
                color: #9BA1B9;
                margin-right: 5px;
            }
            </style>
        """, unsafe_allow_html=True)

    st.title("📈 Stock Analysis Dashboard")

    st.sidebar.title("Stock Selection")

    selection_method = st.sidebar.radio(
        "How do you want to pick a stock?",
        ["Category", "Custom Ticker"],
        key="main_selection_method"
    )

    if selection_method == "Category":
        # If they want to browse categories, we show them organized groups of stocks
        category = st.sidebar.selectbox(
            "Pick a category",
            list(STOCK_CATEGORIES.keys()),
            key="main_category"
        )
        stock_options = STOCK_CATEGORIES[category]
        selected_stock = st.sidebar.selectbox(
            "Choose your stock",
            list(stock_options.keys()),
            format_func=lambda x: f"{x} - {stock_options[x]}",  # Show both symbol and name
            key="main_stock"
        )
    else:
        # If they want to type in their own stock symbol
        selected_stock = st.sidebar.text_input(
            "Type in a stock symbol",
            value="AAPL",
            max_chars=5,  # Stock symbols are usually 1-5 characters
            key="main_ticker"
        ).upper()  # Convert to uppercase since that's how stock symbols are written

    # Time range selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Time Range")

    # Define our preset time periods
    period_options = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "YTD": get_ytd_days(),  # Days since January 1st
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825
    }

    selected_period = st.sidebar.select_slider(
        "Pick a time range",
        options=list(period_options.keys())
    )
    days = period_options[selected_period]

    # Create Spark session first
    spark = create_spark_session()

    stock_df = get_stock_info(selected_stock, spark)
    if stock_df:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Stock Details")
        # Display the info in a nice formatted box
        current_stock_info = stock_df.where(stock_df.ticker == selected_stock).collect()

        if current_stock_info:
            name = current_stock_info[0]['name']
            current_price = current_stock_info[0]['current_price']
            price_change = current_stock_info[0]['price_change']
            volume = current_stock_info[0]['volume']
            market_cap = format_market_cap(current_stock_info[0]['market_cap'])
            sector = current_stock_info[0]['sector']

            # Create a safer display string with explicit null checks
            display_html = f"""
                <div class="stock-info">
                <h4>{selected_stock} - {name if name else 'N/A'}</h4>
                <p><span class="label">Price:</span> <span class="highlight">
                    {f'${current_price:.2f}' if current_price is not None else 'N/A'}</span></p>
                <p><span class="label">Change:</span> <span class="highlight">
                    {f'{price_change:.2f}%' if price_change is not None else 'N/A'}</span></p>
                <p><span class="label">Volume:</span> 
                    {f'{int(volume):,}' if volume is not None else 'N/A'}</p>
                <p><span class="label">Market Cap:</span> {market_cap if market_cap else 'N/A'}</p>
                <p><span class="label">Sector:</span> {sector if sector else 'N/A'}</p>
                </div>
            """

            st.sidebar.markdown(display_html, unsafe_allow_html=True)
        else:
            st.sidebar.warning("No data available for this stock")

    # Create three main tabs for different types of analysis
    tab1, tab2, tab3 = st.tabs(["Explore", "Process", "Analyze"])

    # Run the appropriate analysis based on which tab is selected
    with tab1:
        explore_data(spark, selected_stock, days)
    with tab2:
        preprocess_data(spark, selected_stock, days)
    with tab3:
        analyze_data(spark, selected_stock, days)

    # Clean up after operations
    cleanup_spark_cache(spark)
    gc.collect()

    # Always stop Spark session
    if 'spark' in locals():
        spark.stop()


# This is where the app starts running
if __name__ == "__main__":
    main()

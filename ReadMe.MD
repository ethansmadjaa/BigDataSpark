# Big Data Analysis Project

### Nasdaq Tech Stocks Analysis using Streamlit, Spark, and Python

This project aims to build a Python application that provides valuable insights into Nasdaq technology stocks using the Apache Spark framework. The application is intended to assist traders and investors in making informed decisions by exploring, preprocessing, and analyzing historical stock data. The project is built with **Streamlit** for visualization, **Apache Spark** for data processing, and **yfinance** for fetching stock market data.

## Project Overview

This project goes through the entire data analysis process, from data **exploration** to **preprocessing**, and finally to **analysis and visualization**. We leverage **Spark** to efficiently handle large datasets, while **Streamlit** allows us to create an interactive web interface for users.

The main goal of this application is to extract useful information from historical stock prices to aid in decision-making for investors looking to invest in these stocks.

## Objectives

- Apply big data concepts to create a structured data analysis pipeline
- Use Spark to build a scalable solution for analyzing large datasets of stock data
- Generate meaningful insights to assist traders in making informed investment decisions
- Use visualization tools (Plotly) to effectively present findings

## Project Structure

The project is modular, consisting of the following main components:

- **app.py**: Main application file with Streamlit interface and stock selection
- **exploration.py**: Handles data visualization and basic statistics
- **preprocessing.py**: Technical indicators and data cleaning
- **analysis.py**: Risk metrics and trading signals
- **utils/**
  - **spark_utils.py**: Spark session management
  - **stock_utils.py**: Stock data fetching and processing
  - **risk_warnings.py**: Risk disclaimers
  - **constants.py**: Stock categories and configuration

## Features

### 1. Data Exploration
- Interactive candlestick charts with volume
- Basic statistics (price, volume, returns)
- Moving averages visualization
- Stock comparison tools

### 2. Data Processing
- Missing value detection and handling
- Technical indicators calculation:
  - Simple Moving Averages (SMA)
  - Exponential Moving Averages (EMA)
  - Bollinger Bands
  - Price Rate of Change (ROC)
  - Momentum indicators

### 3. Analysis
- Risk metrics:
  - Volatility analysis
  - Maximum drawdown
  - Value at Risk (VaR)
- Statistical analysis:
  - Return distribution
  - Normality tests
  - Outlier detection
- Trading signals based on multiple factors

## Installation

### Prerequisites
- Python 3.11
- Java 8 or higher (for Apache Spark)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ethansmadjaa/BigDataSpark.git
````
```bash
cd BigDataSpark
```

2. Create a virtual environment:
### On Windows
```bash
python -m venv .venv
```
```Bash
.\.venv\Scripts\activate
```

### On MacOs/Linux
```bash
python3 -m venv .venv
```
```Bash
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Select a stock:
   - Choose from predefined categories
   - Or enter a custom ticker symbol

2. Choose analysis period:
   - 1 Month to 5 Years
   - Or Year-to-Date (YTD)

3. Explore different analyses:
   - Use the "Explore" tab for basic visualization
   - "Process" tab for technical indicators
   - "Analyze" tab for risk metrics and signals

## ⚠️ Risk Disclaimer

This tool is for **educational purposes only**:
- Not financial advice
- Past performance doesn't predict future results
- Technical analysis has limitations
- Consult financial professionals for investment decisions

## Contributors

- **Ethan SMADJA**
- **Tom URBAN**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Developed as part of the Big Data Frameworks course at ECE Paris - M1 DATA & AI.

import streamlit as st
import plotly.graph_objects as go
from pyspark.sql import SparkSession, DataFrame


def analyze_data(spark_session, stock, days):
    st.title("Data Analysis")
    st.text("Module in development ! 🚀")
    
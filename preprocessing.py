import streamlit as st
from pyspark.sql import SparkSession, DataFrame


def preprocess_data(spark_session, stock, days):
    st.title("Data preporcessing")
    st.text("Module in development ! 🚀")
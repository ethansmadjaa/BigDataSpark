import streamlit as st
from pyspark.sql import SparkSession


def create_spark_session() -> SparkSession:
    """
    Create a simple Spark session for our stock analysis.
    This is a basic configuration for learning purposes.
    """
    try:
        # TODO: Might need to adjust memory settings based on data size
        spark = (SparkSession.builder
                .appName("Stock Analysis Project")
                # Basic memory settings for local development
                .config("spark.driver.memory", "2g")
                # Using local mode for development and testing
                .master("local[*]")
                .getOrCreate())
        
        # Reduce logging noise
        spark.sparkContext.setLogLevel("ERROR")
        return spark
        
    except Exception as e:
        st.error(f"Failed to create Spark session: {str(e)}")
        raise


def stop_spark_session(spark: SparkSession):
    """Stop the Spark session when we're done."""
    if spark:
        spark.stop()


def cleanup_spark_cache(spark: SparkSession):
    """
    Clean up cached data to free memory.
    Note: This is important when working with limited RAM on my laptop!
    """
    spark.catalog.clearCache()

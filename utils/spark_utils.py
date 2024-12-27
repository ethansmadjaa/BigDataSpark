import streamlit as st
from pyspark.sql import SparkSession
import os


def create_spark_session() -> SparkSession:
    """
    Create a simple Spark session for our stock analysis.
    This is a basic configuration for learning purposes.
    """
    try:
        # Set environment variables for Java
        os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
        
        # Create basic Spark session with minimal configuration
        spark = (SparkSession.builder
                .appName("Stock Analysis Project")
                # Basic memory settings
                .config("spark.driver.memory", "1g")
                # Important for cloud deployment
                .config("spark.driver.bindAddress", "127.0.0.1")
                .config("spark.driver.host", "127.0.0.1")
                # Use local mode
                .master("local[*]")
                # Disable Spark UI for cloud
                .config("spark.ui.enabled", "false")
                .getOrCreate())
        
        # Reduce logging noise
        spark.sparkContext.setLogLevel("ERROR")
        
        return spark
        
    except Exception as e:
        st.error(f"Failed to create Spark session: {str(e)}")
        st.error("Please try refreshing the page. If the error persists, contact support.")
        raise


def stop_spark_session(spark: SparkSession):
    """Stop the Spark session when we're done."""
    if spark:
        try:
            spark.stop()
        except Exception as e:
            st.warning(f"Error stopping Spark: {str(e)}")


def cleanup_spark_cache(spark: SparkSession):
    """
    Clean up cached data to free memory.
    Note: This is important when working with limited RAM on my laptop!
    """
    if spark:
        try:
            spark.catalog.clearCache()
        except Exception as e:
            st.warning(f"Error clearing cache: {str(e)}")

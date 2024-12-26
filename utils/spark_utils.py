import streamlit as st
from pyspark.sql import SparkSession
from typing import Optional, Dict


def create_spark_session() -> SparkSession:
    """Create and configure a Spark session optimized for Streamlit Cloud."""
    try:
        # Create minimal Spark session for cloud deployment
        spark = (SparkSession.builder
                .appName("Stock Analysis")
                # Minimal memory configuration for cloud
                .config("spark.driver.memory", "1g")
                .config("spark.executor.memory", "1g")
                .config("spark.sql.shuffle.partitions", "4")
                # Essential configurations
                .config("spark.driver.host", "localhost")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.session.timeZone", "UTC")
                # Minimal JVM options
                .config("spark.driver.extraJavaOptions", 
                       '-XX:+UseG1GC -XX:MaxCodeCacheSize=512M')
                # Local mode with limited cores
                .master("local[2]")
                .getOrCreate())
        
        spark.sparkContext.setLogLevel("ERROR")
        return spark
        
    except Exception as e:
        print(f"Error creating Spark session: {str(e)}")
        raise


def stop_spark_session(spark: SparkSession):
    """
    Safely stop a Spark session.
    
    Args:
        spark (SparkSession): Active Spark session to stop
    """
    if spark is not None:
        try:
            spark.stop()
        except Exception as e:
            print(f"Error stopping Spark session: {str(e)}")


def cleanup_spark_cache(spark: SparkSession) -> None:
    """Utility function to clean up Spark cache."""
    spark.catalog.clearCache()

from pyspark.sql import SparkSession
from typing import Optional, Dict


def create_spark_session() -> SparkSession:
    """
    Create and configure a Spark session.
    
    Returns:
        SparkSession: Configured Spark session
    """
    try:
        # Set Java options for code cache
        import os
        os.environ['SPARK_SUBMIT_OPTS'] = '-XX:ReservedCodeCacheSize=512M -XX:MaxCodeCacheSize=512M'
        
        # Create Spark session with more memory and proper configuration
        spark = (SparkSession.builder
                .appName("Stock Analysis")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .config("spark.sql.session.timeZone", "UTC")
                .config("spark.driver.maxResultSize", "4g")
                .config("spark.memory.offHeap.enabled", "true")
                .config("spark.memory.offHeap.size", "4g")
                .config("spark.local.dir", "/tmp")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.shuffle.partitions", "10")
                # JVM options for better performance
                .config("spark.driver.extraJavaOptions", 
                       "-XX:+UseG1GC -XX:ReservedCodeCacheSize=512M -XX:MaxCodeCacheSize=512M")
                .config("spark.executor.extraJavaOptions",
                       "-XX:+UseG1GC -XX:ReservedCodeCacheSize=512M -XX:MaxCodeCacheSize=512M")
                # Add these configurations for stability
                .config("spark.driver.host", "localhost")
                .master("local[*]")
                .getOrCreate())
        
        # Set log level to reduce noise
        spark.sparkContext.setLogLevel("ERROR")
        
        return spark
        
    except Exception as e:
        print(f"Error creating Spark session: {str(e)}")
        raise  # Re-raise the exception to handle it in the calling code


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

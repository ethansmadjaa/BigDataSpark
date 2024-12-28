import streamlit as st
from pyspark.sql import SparkSession


# TODO: fix memory leak when processing large datasets
# TODO: add proper error handling for spark failures
# TODO: implement session pooling for better perf
# TODO: cleanup the config mess - too many hardcoded values

def create_spark_session() -> SparkSession:
    """
    Setup basic spark session for our analysis.
    Nothing fancy, just local mode for now.
    """
    try:
        # spark config - might need tweaking
        sprk = (SparkSession.builder
                .appName("Stock Analysis")
                # 2g seems ok on my laptop but might need more
                .config("spark.driver.memory", "2g")
                # local mode cuz we're poor lol
                .master("local[*]")
                .getOrCreate())

        # shut up spark, ur too noisy
        sprk.sparkContext.setLogLevel("ERROR")
        return sprk

    except Exception as e:
        # should prob handle diff exceptions differently but whatever
        st.error(f"Spark died: {str(e)}")
        raise


def stop_spark_session(sprk: SparkSession):
    """Kill spark when we're done."""
    if sprk:  # check if its not ded already
        sprk.stop()


def cleanup_spark_cache(sprk: SparkSession):
    """
    Clear spark cache cuz memory go brrr.
    My poor laptop can't handle too much data lmao.
    """
    # note: might wanna add some checks here
    sprk.catalog.clearCache()  # yeet the cache

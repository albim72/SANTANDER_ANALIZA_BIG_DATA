#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("StreamingTwoCountersExample")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    stream_df = (
        spark.readStream
        .format("rate")
        .option("rowsPerSecond", 10)
        .load()
    )

    data_df = (
        stream_df
        .withColumn("amount", (F.col("value") % 100) + 1)
    )

    data_flagged = (
        data_df
        .withColumn("is_big", F.col("amount") > 50)
        .withColumn("is_small", F.col("amount") <= 50)
    )

    agg_df = (
        data_flagged
        .withWatermark("timestamp", "30 seconds")
        .groupBy(
            F.window("timestamp", "10 seconds")
        )
        .agg(
            F.sum(F.when(F.col("is_big"), 1).otherwise(0)).alias("count_big"),
            F.sum(F.when(F.col("is_small"), 1).otherwise(0)).alias("count_small"),
            F.sum(F.when(F.col("is_big"), F.col("amount")).otherwise(0)).alias("sum_big"),
            F.sum(F.when(F.col("is_small"), F.col("amount")).otherwise(0)).alias("sum_small"),
            F.count("*").alias("total_count")
        )
        .withColumn(
            "big_ratio_pct",
            F.when(
                F.col("total_count") > 0,
                F.round(F.col("count_big") / F.col("total_count") * 100.0, 2)
            ).otherwise(F.lit(0.0))
        )
        .orderBy("window")
    )

    query = (
        agg_df
        .writeStream
        .outputMode("update")
        .format("console")
        .option("truncate", False)
        .option("numRows", 50)
        .start()
    )

    print("\nStreaming started.")
    print("Source: Spark 'rate' → generates values 0..∞ → amount 1–100.")
    print("Two counters computed in 10-second windows.\n")
    print("Stop with Ctrl+C.\n")

    query.awaitTermination()

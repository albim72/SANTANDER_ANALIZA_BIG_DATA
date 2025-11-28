#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BANK + BIG DATA + SQL (Spark) – jeden spójny przykład

Scenariusz:
-----------
Mamy 3 rozproszone źródła danych:

1) KLIENCI (customers) – CSV (np. CRM / MDM)
   Pola:
     - customer_id
     - first_name
     - last_name
     - segment       (np. RETAIL / VIP / SME)

2) TRANSAKCJE (transactions) – Parquet (hurtownia / lakehouse)
   Pola:
     - tx_id
     - customer_id
     - tx_timestamp
     - amount
     - channel       (ATM / POS / ONLINE)
     - country       (PL / DE / US / ...)

3) ALERTY AML (alerts) – JSON (system monitorujący)
   Pola:
     - alert_id
     - customer_id
     - risk_score    (0.0 – 1.0)
     - risk_class    (low / medium / high)
     - created_at

Cel:
----
- wygenerować syntetyczne dane w 3 formatach (CSV, Parquet, JSON),
- wczytać je do Sparka,
- zarejestrować jako tabele tymczasowe,
- wykonać ANALIZĘ w CZYSTYM SQL (Spark SQL) w stylu bankowym:
    * sumy transakcji klientów,
    * połączenie z alertami AML,
    * Top N klientów wg ryzyka / kwot,
    * ranking krajów.

W realnym środowisku:
- liczby rekordów będą x100 / x1000 większe,
- pliki będą na HDFS / S3 / Lakehouse,
- ale logika i SQL zostają takie same.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ==========================
# PARAMETRY „SKALI” DEMO
# ==========================
NUM_CUSTOMERS = 50_000     # liczba klientów
NUM_TRANSACTIONS = 200_000 # liczba transakcji
NUM_ALERTS = 10_000        # liczba alertów AML

BASE_DIR = "bank_bigdata_sql_demo"
CUSTOMERS_PATH = os.path.join(BASE_DIR, "customers_csv")    # CSV
TRANSACTIONS_PATH = os.path.join(BASE_DIR, "transactions_parquet")  # Parquet
ALERTS_PATH = os.path.join(BASE_DIR, "alerts_json")         # JSON


# ==========================
# FUNKCJE GENERUJĄCE DANE
# ==========================

def generate_customers(spark: SparkSession):
    """
    Generowanie klientów:
      - customer_id
      - first_name
      - last_name
      - segment (RETAIL / VIP / SME)
    Zapis do CSV (źródło 1).
    """
    print(f"[GEN] Generuję klientów ({NUM_CUSTOMERS:,})...".replace(",", " "))

    first_names = ["Jan", "Anna", "Ewa", "Piotr", "Krzysztof", "Maria", "Tomasz", "Agnieszka"]
    last_names = ["Nowak", "Kowalski", "Wiśniewski", "Wójcik", "Kamińska", "Lewandowski", "Zieliński"]
    segments = ["RETAIL", "VIP", "SME"]

    customers_df = (
        spark.range(1, NUM_CUSTOMERS + 1)
        .withColumnRenamed("id", "customer_id")
        .withColumn(
            "first_name",
            F.element_at(F.array(*[F.lit(x) for x in first_names]),
                         (F.rand(seed=1) * len(first_names)).cast("int") + 1)
        )
        .withColumn(
            "last_name",
            F.element_at(F.array(*[F.lit(x) for x in last_names]),
                         (F.rand(seed=2) * len(last_names)).cast("int") + 1)
        )
        .withColumn(
            "segment",
            F.element_at(F.array(*[F.lit(x) for x in segments]),
                         (F.rand(seed=3) * len(segments)).cast("int") + 1)
        )
    )

    (
        customers_df
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(CUSTOMERS_PATH)
    )

    print("[GEN] Klienci zapisani (CSV).")


def generate_transactions(spark: SparkSession):
    """
    Generowanie transakcji:
      - tx_id
      - customer_id (losowo z istniejących klientów)
      - tx_timestamp (sztuczna data/czas)
      - amount (kwota – 10–10_000)
      - channel (ATM / POS / ONLINE)
      - country (PL / DE / US / GB)
    Zapis do Parquet (źródło 2).
    """
    print(f"[GEN] Generuję transakcje ({NUM_TRANSACTIONS:,})...".replace(",", " "))

    channels = ["ATM", "POS", "ONLINE"]
    countries = ["PL", "DE", "US", "GB", "FR", "CZ"]

    tx_df = (
        spark.range(1, NUM_TRANSACTIONS + 1)
        .withColumnRenamed("id", "tx_id")
        .withColumn("customer_id", (F.rand(seed=4) * NUM_CUSTOMERS).cast("int") + 1)
        .withColumn("amount", F.round(F.rand(seed=5) * 9990 + 10, 2))
        .withColumn(
            "channel",
            F.element_at(F.array(*[F.lit(x) for x in channels]),
                         (F.rand(seed=6) * len(channels)).cast("int") + 1)
        )
        .withColumn(
            "country",
            F.element_at(F.array(*[F.lit(x) for x in countries]),
                         (F.rand(seed=7) * len(countries)).cast("int") + 1)
        )
        # sztuczny timestamp – dzień + losowe przesunięcie godzin
        .withColumn("tx_timestamp",
                    F.to_timestamp(
                        F.concat(F.lit("2025-11-27 "),
                                 F.lpad((F.rand(seed=8) * 24).cast("int"), 2, "0"),
                                 F.lit(":"),
                                 F.lpad((F.rand(seed=9) * 60).cast("int"), 2, "0"),
                                 F.lit(":00")
                                 )
                    )
        )
    )

    (
        tx_df
        .write
        .mode("overwrite")
        .parquet(TRANSACTIONS_PATH)
    )

    print("[GEN] Transakcje zapisane (Parquet).")


def generate_alerts(spark: SparkSession):
    """
    Generowanie alertów AML:
      - alert_id
      - customer_id
      - risk_score (0.0–1.0)
      - risk_class (low / medium / high)
      - created_at
    Zapis do JSON (źródło 3).
    """
    print(f"[GEN] Generuję alerty AML ({NUM_ALERTS:,})...".replace(",", " "))

    base_df = spark.range(1, NUM_ALERTS + 1).withColumnRenamed("id", "alert_id")

    alerts_df = (
        base_df
        # losowy klient
        .withColumn("customer_id", (F.rand(seed=10) * NUM_CUSTOMERS).cast("int") + 1)
        # ryzyko ciągłe 0–1
        .withColumn("risk_score", F.round(F.rand(seed=11), 3))
        # klasy ryzyka
        .withColumn(
            "risk_class",
            F.when(F.col("risk_score") >= 0.8, "high")
             .when(F.col("risk_score") >= 0.4, "medium")
             .otherwise("low")
        )
        .withColumn(
            "created_at",
            F.to_timestamp(
                F.concat(F.lit("2025-11-27 "),
                         F.lpad((F.rand(seed=12) * 24).cast("int"), 2, "0"),
                         F.lit(":"),
                         F.lpad((F.rand(seed=13) * 60).cast("int"), 2, "0"),
                         F.lit(":00")
                         )
            )
        )
    )

    (
        alerts_df
        .write
        .mode("overwrite")
        .json(ALERTS_PATH)
    )

    print("[GEN] Alerty AML zapisane (JSON).")


# ==========================
# ANALIZA SQL W STYLU BIG DATA
# ==========================

def run_sql_analysis(spark: SparkSession):
    """
    Wczytanie 3 źródeł (CSV, Parquet, JSON),
    rejestracja jako widoki SQL
    i wykonanie kilku zapytań w konwencji bankowej.
    """

    print("\n[LOAD] Wczytuję klientów (CSV)...")
    customers_df = (
        spark.read
        .option("header", "true")
        .csv(CUSTOMERS_PATH)
        .select(
            F.col("customer_id").cast("int").alias("customer_id"),
            "first_name",
            "last_name",
            "segment"
        )
    )

    print("[LOAD] Wczytuję transakcje (Parquet)...")
    tx_df = spark.read.parquet(TRANSACTIONS_PATH)

    print("[LOAD] Wczytuję alerty AML (JSON)...")
    alerts_df = spark.read.json(ALERTS_PATH)

    print("\n[INFO] Schemat customers:")
    customers_df.printSchema()
    print("[INFO] Schemat transactions:")
    tx_df.printSchema()
    print("[INFO] Schemat alerts:")
    alerts_df.printSchema()

    # Rejestracja jako widoki SQL (kluczowy krok)
    customers_df.createOrReplaceTempView("customers")
    tx_df.createOrReplaceTempView("transactions")
    alerts_df.createOrReplaceTempView("alerts")

    # ==========================================================
    # 1. SUMA i liczba transakcji per klient – SQL na Parquet+CSV
    # ==========================================================
    print("\n[SQL 1] Top 10 klientów wg sumy transakcji:")

    sql1 = """
        SELECT
            c.customer_id,
            c.first_name,
            c.last_name,
            c.segment,
            SUM(t.amount) AS total_amount,
            COUNT(*) AS tx_count
        FROM transactions t
        JOIN customers c
            ON t.customer_id = c.customer_id
        GROUP BY
            c.customer_id, c.first_name, c.last_name, c.segment
        ORDER BY total_amount DESC
        LIMIT 10
    """

    result1 = spark.sql(sql1)
    result1.show(truncate=False)

    # ==========================================================
    # 2. Połączenie z alertami AML – klienci z wysokim ryzykiem
    # ==========================================================
    print("\n[SQL 2] Klienci z alertami high-risk + sumy transakcji:")

    sql2 = """
        SELECT
            c.customer_id,
            c.first_name,
            c.last_name,
            c.segment,
            a.alert_id,
            a.risk_score,
            a.risk_class,
            SUM(t.amount) AS total_amount,
            COUNT(*) AS tx_count
        FROM alerts a
        JOIN customers c
            ON a.customer_id = c.customer_id
        LEFT JOIN transactions t
            ON t.customer_id = c.customer_id
        WHERE a.risk_class = 'high'
        GROUP BY
            c.customer_id,
            c.first_name,
            c.last_name,
            c.segment,
            a.alert_id,
            a.risk_score,
            a.risk_class
        ORDER BY a.risk_score DESC, total_amount DESC
        LIMIT 20
    """

    result2 = spark.sql(sql2)
    result2.show(truncate=False)

    # ==========================================================
    # 3. Ranking krajów wg wolumenu transakcji (AML / Fraud view)
    # ==========================================================
    print("\n[SQL 3] Ranking krajów wg sumy i liczby transakcji:")

    sql3 = """
        SELECT
            country,
            COUNT(*) AS tx_count,
            SUM(amount) AS total_amount,
            AVG(amount) AS avg_amount
        FROM transactions
        GROUP BY country
        ORDER BY total_amount DESC
    """

    result3 = spark.sql(sql3)
    result3.show(truncate=False)

    # ==========================================================
    # 4. „Bankowe” KPI: średnia wartość transakcji klienta z alertem
    # ==========================================================
    print("\n[SQL 4] Średnia kwota transakcji klientów z alertami high vs reszta:")

    sql4 = """
        WITH flagged_customers AS (
            SELECT DISTINCT customer_id
            FROM alerts
            WHERE risk_class = 'high'
        )
        SELECT
            CASE
                WHEN fc.customer_id IS NOT NULL THEN 'HIGH_RISK_CUSTOMER'
                ELSE 'OTHER_CUSTOMER'
            END AS customer_risk_group,
            COUNT(t.amount) AS tx_count,
            SUM(t.amount) AS total_amount,
            AVG(t.amount) AS avg_amount
        FROM transactions t
        LEFT JOIN flagged_customers fc ON t.customer_id = fc.customer_id
        GROUP BY
            CASE
                WHEN fc.customer_id IS NOT NULL THEN 'HIGH_RISK_CUSTOMER'
                ELSE 'OTHER_CUSTOMER'
            END
    """

    result4 = spark.sql(sql4)
    result4.show(truncate=False)

    print("\n[END] Analiza SQL w konwencji Big Data (bank) zakończona.")


# ==========================
# GŁÓWNY BLOK SKRYPTU
# ==========================

if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("BankBigDataSQLDemo")
        .getOrCreate()
    )

    # 1. Generowanie syntetycznych danych w 3 formatach
    generate_customers(spark)
    generate_transactions(spark)
    generate_alerts(spark)

    # 2. Analiza w czystym SQL nad rozproszonymi danymi
    run_sql_analysis(spark)

    # 3. Zatrzymanie Sparka
    spark.stop()

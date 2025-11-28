#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Przypadek BIG DATA:
Dane rozproszone z 3 różnych źródeł, które trzeba połączyć (PySpark)

Scenariusz biznesowy:
---------------------
Mamy 3 systemy / źródła danych:

1) System CRM  (źródło A)  -> lista klientów
   Format: CSV (np. z hurtowni, eksport z CRM)
   Pola:
     - customer_id
     - first_name
     - last_name
     - city

2) System zamówień (źródło B)  -> zamówienia klientów
   Format: Parquet (system analityczny / data lake)
   Pola:
     - order_id
     - customer_id
     - order_amount

3) System płatności (źródło C) -> informacje o opłaceniu zamówień
   Format: JSON (logi z bramki płatniczej, dane pół-strukturalne)
   Pola:
     - order_id
     - paid_ratio   (0.0 = nieopłacone, 0.5 = częściowo, 1.0 = w pełni opłacone)

Cel:
----
- zasymulować rozproszone dane (3 różne formaty i ścieżki),
- wczytać je do PySpark,
- połączyć po kluczach (customer_id, order_id),
- policzyć:
    * ile było zamówień,
    * ile przychodu jest w zamówieniach,
    * ile faktycznie zostało zapłacone,
    * TOP 10 klientów wg zaległych kwot,
    * TOP 10 miast wg przychodu.

W realnym BIG DATA:
- liczby rekordów byłyby rzędu milionów / dziesiątek milionów,
- pliki siedziałyby na HDFS/S3/Lakehouse.
Tutaj generujemy syntetyczny, ale analogiczny case.
"""

import os
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast


# ============================================================
# PARAMETRY "WIELKOŚCI" (dla demo możesz zmienić na mniejsze)
# ============================================================
NUM_CUSTOMERS = 100_000   # liczba klientów
NUM_ORDERS = 500_000      # liczba zamówień
# Płatności wygenerujemy 1:1 do zamówień (dla uproszczenia)

BASE_DIR = "data_big_three_sources"
CUSTOMERS_PATH = os.path.join(BASE_DIR, "customers_csv")     # CSV
ORDERS_PATH = os.path.join(BASE_DIR, "orders_parquet")       # Parquet
PAYMENTS_PATH = os.path.join(BASE_DIR, "payments_json")      # JSON


# ================================================
# FUNKCJE POMOCNICZE: GENEROWANIE DANYCH
# ================================================

def generate_customers(spark: SparkSession, path: str, num_customers: int) -> None:
    """
    Generuje tabelę klientów i zapisuje do CSV (źródło A).
    Używamy prostego generatora imion/nazwisk z listy.
    """
    print(f"[GEN] Generuję klientów -> {path} (n = {num_customers:,})".replace(",", " "))

    first_names: List[str] = ["Jan", "Anna", "Ewa", "Piotr", "Krzysztof", "Maria", "Tomasz", "Agnieszka"]
    last_names: List[str] = ["Nowak", "Kowalski", "Wiśniewski", "Wójcik", "Kamińska", "Lewandowski", "Zieliński"]
    cities: List[str] = ["Warszawa", "Kraków", "Gdańsk", "Poznań", "Wrocław", "Lublin", "Szczecin", "Katowice"]

    customers_df = (
        spark.range(1, num_customers + 1)
        .withColumnRenamed("id", "customer_id")
        .withColumn(
            "first_name",
            F.element_at(
                F.array(*[F.lit(x) for x in first_names]),
                (F.rand(seed=1) * len(first_names)).cast("int") + 1,
            )
        )
        .withColumn(
            "last_name",
            F.element_at(
                F.array(*[F.lit(x) for x in last_names]),
                (F.rand(seed=2) * len(last_names)).cast("int") + 1,
            )
        )
        .withColumn(
            "city",
            F.element_at(
                F.array(*[F.lit(x) for x in cities]),
                (F.rand(seed=3) * len(cities)).cast("int") + 1,
            )
        )
    )

    (
        customers_df
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(path)
    )

    print("[GEN] Klienci zapisani (CSV).")


def generate_orders(spark: SparkSession, path: str, num_orders: int, num_customers: int) -> None:
    """
    Generuje tabelę zamówień i zapisuje do PARQUET (źródło B).
    """
    print(f"[GEN] Generuję zamówienia -> {path} (n = {num_orders:,})".replace(",", " "))

    orders_df = (
        spark.range(1, num_orders + 1)
        .withColumnRenamed("id", "order_id")
        # losowo przydzielamy zamówienia do klientów
        .withColumn("customer_id", (F.rand(seed=4) * num_customers).cast("int") + 1)
        # losowa kwota zamówienia 100–2000
        .withColumn("order_amount", F.round(F.rand(seed=5) * 1900 + 100, 2))
    )

    (
        orders_df
        .write
        .mode("overwrite")
        .parquet(path)
    )

    print("[GEN] Zamówienia zapisane (Parquet).")


def generate_payments(spark: SparkSession, path: str, num_orders: int) -> None:
    """
    Generuje tabelę płatności i zapisuje do JSON (źródło C).

    paid_ratio:
      - ~10% zamówień nieopłaconych (0.0)
      - ~10% częściowo opłaconych (0.5)
      - ~80% w pełni opłaconych (1.0)
    """
    print(f"[GEN] Generuję płatności -> {path} (n = {num_orders:,})".replace(",", " "))

    payments_df = (
        spark.range(1, num_orders + 1)
        .withColumnRenamed("id", "order_id")
        .withColumn("r", F.rand(seed=6))
        .withColumn(
            "paid_ratio",
            F.when(F.col("r") < 0.10, F.lit(0.0))   # nieopłacone
             .when(F.col("r") < 0.20, F.lit(0.5))   # częściowe
             .otherwise(F.lit(1.0))                 # w pełni opłacone
        )
        .drop("r")
    )

    (
        payments_df
        .write
        .mode("overwrite")
        .json(path)
    )

    print("[GEN] Płatności zapisane (JSON).")


# ================================================
# ANALIZA: ŁĄCZENIE 3 ŹRÓDEŁ I AGREGACJE
# ================================================

def run_analysis(spark: SparkSession) -> None:
    """
    Wczytuje 3 źródła danych (CSV, Parquet, JSON), łączy je
    i wykonuje analizy w kontekście Big Data.
    """

    print("\n[LOAD] Wczytuję klientów (CSV)...")
    customers_df = (
        spark.read
        .option("header", "true")
        .csv(CUSTOMERS_PATH)
        .select(
            F.col("customer_id").cast("int"),
            "first_name",
            "last_name",
            "city",
        )
    )

    print("[LOAD] Wczytuję zamówienia (Parquet)...")
    orders_df = spark.read.parquet(ORDERS_PATH)

    print("[LOAD] Wczytuję płatności (JSON)...")
    payments_df = spark.read.json(PAYMENTS_PATH)

    print("\n=== Schemat klientów ===")
    customers_df.printSchema()
    print("=== Schemat zamówień ===")
    orders_df.printSchema()
    print("=== Schemat płatności ===")
    payments_df.printSchema()

    # -------------------------------------------------------------
    # 1. Połączenie zamówień z klientami
    #    Rozsądne założenie: klienci (100k) << zamówienia (500k),
    #    więc używamy BROADCAST JOIN dla przyspieszenia.
    # -------------------------------------------------------------
    print("\n[JOIN] Łączę zamówienia z klientami (broadcast join po customer_id)...")

    orders_with_customers = (
        orders_df
        .join(
            broadcast(customers_df),
            on="customer_id",
            how="left"
        )
    )

    print("[INFO] Przykładowe połączone rekordy (klient + zamówienie):")
    orders_with_customers.show(5, truncate=False)

    # -------------------------------------------------------------
    # 2. Połączenie z płatnościami po order_id
    # -------------------------------------------------------------
    print("\n[JOIN] Łączę z płatnościami po order_id...")

    full_df = (
        orders_with_customers
        .join(payments_df, on="order_id", how="left")
        .withColumn(
            "paid_ratio",
            F.coalesce(F.col("paid_ratio"), F.lit(0.0))  # brak płatności -> 0.0
        )
        .withColumn(
            "paid_amount",
            F.round(F.col("order_amount") * F.col("paid_ratio"), 2)
        )
        .withColumn(
            "unpaid_amount",
            F.round(F.col("order_amount") - F.col("paid_amount"), 2)
        )
        .withColumn(
            "payment_status",
            F.when(F.col("paid_ratio") == 0.0, F.lit("UNPAID"))
             .when(F.col("paid_ratio") == 1.0, F.lit("FULLY_PAID"))
             .otherwise(F.lit("PARTIALLY_PAID"))
        )
    )

    print("[INFO] Przykładowe rekordy po połączeniu 3 źródeł:")
    full_df.select(
        "order_id", "customer_id", "first_name", "last_name",
        "city", "order_amount", "paid_amount", "unpaid_amount", "payment_status"
    ).show(10, truncate=False)

    # -------------------------------------------------------------
    # 3. Podstawowe metryki ogólne
    # -------------------------------------------------------------
    print("\n[METRICS] Podstawowe metryki globalne:")

    total_orders = full_df.count()
    total_order_amount = full_df.agg(F.sum("order_amount")).first()[0]
    total_paid_amount = full_df.agg(F.sum("paid_amount")).first()[0]
    total_unpaid_amount = full_df.agg(F.sum("unpaid_amount")).first()[0]

    print(f"Liczba zamówień:       {total_orders:,}".replace(",", " "))
    print(f"Suma kwot zamówień:    {total_order_amount:,.2f}".replace(",", " "))
    print(f"Suma kwot zapłaconych: {total_paid_amount:,.2f}".replace(",", " "))
    print(f"Suma zaległości:       {total_unpaid_amount:,.2f}".replace(",", " "))

    # podział wg statusu płatności
    print("\n[METRICS] Podział zamówień wg statusu płatności:")
    (
        full_df
        .groupBy("payment_status")
        .agg(
            F.count("*").alias("orders_count"),
            F.round(F.sum("order_amount"), 2).alias("total_order_amount"),
            F.round(F.sum("paid_amount"), 2).alias("total_paid_amount"),
            F.round(F.sum("unpaid_amount"), 2).alias("total_unpaid_amount"),
        )
        .orderBy("payment_status")
        .show(truncate=False)
    )

    # -------------------------------------------------------------
    # 4. TOP 10 klientów wg zaległych kwot
    # -------------------------------------------------------------
    print("\n[METRICS] TOP 10 klientów wg zaległych kwot:")

    (
        full_df
        .groupBy("customer_id", "first_name", "last_name", "city")
        .agg(
            F.round(F.sum("order_amount"), 2).alias("total_order_amount"),
            F.round(F.sum("paid_amount"), 2).alias("total_paid_amount"),
            F.round(F.sum("unpaid_amount"), 2).alias("total_unpaid_amount"),
        )
        .orderBy(F.desc("total_unpaid_amount"))
        .limit(10)
        .show(truncate=False)
    )

    # -------------------------------------------------------------
    # 5. TOP 10 miast wg przychodu (order_amount) i kwot zapłaconych
    # -------------------------------------------------------------
    print("\n[METRICS] TOP 10 miast wg sumarycznej wartości zamówień i płatności:")

    (
        full_df
        .groupBy("city")
        .agg(
            F.round(F.sum("order_amount"), 2).alias("total_order_amount"),
            F.round(F.sum("paid_amount"), 2).alias("total_paid_amount"),
            F.round(F.sum("unpaid_amount"), 2).alias("total_unpaid_amount"),
            F.count("*").alias("orders_count"),
        )
        .orderBy(F.desc("total_order_amount"))
        .limit(10)
        .show(truncate=False)
    )

    print("\n[END] Analiza danych z 3 rozproszonych źródeł zakończona.")


# ================================================
# GŁÓWNY BLOK SKRYPTU
# ================================================

if __name__ == "__main__":
    # Tworzymy katalog bazowy, jeśli nie istnieje
    os.makedirs(BASE_DIR, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("ThreeSourcesBigDataExample")
        .getOrCreate()
    )

    # 1. Generowanie danych (syntetyczne Big Data z 3 źródeł)
    generate_customers(spark, CUSTOMERS_PATH, NUM_CUSTOMERS)
    generate_orders(spark, ORDERS_PATH, NUM_ORDERS, NUM_CUSTOMERS)
    generate_payments(spark, PAYMENTS_PATH, NUM_ORDERS)

    # 2. Analiza – łączenie 3 źródeł i agregacje
    run_analysis(spark)

    # 3. Zakończenie pracy Sparka
    spark.stop()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Install faker library if not already installed
!pip install faker

"""
Big Data – dane pół-strukturalne JSON (2 000 000 rekordów) + analiza w PySpark
-------------------------------------------------------------------------------

1) Generowanie dużego pliku NDJSON (orders_2m.json):
   Każda linia to osobny obiekt JSON o strukturze:
   {
     "order_id": int,
     "customer": {
         "id": int,
         "name": str,
         "surname": str
     },
     "items": [
         {"product": str, "qty": int, "price": float},
         ...
     ],
     "city": str
   }

2) Analiza w PySpark:
   - odczyt JSON-a jako dane pół-strukturalne
   - eksplozja tablicy items
   - wyliczenie wartości zamówień
   - TOP miasta (liczba zamówień, suma wartości)
   - TOP produkty wg przychodu
   - średnia wartość zamówienia
"""

import json
import random
import os
from typing import List, Dict

from faker import Faker

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ========================================
# PARAMETRY GENERATORA DANYCH
# ========================================

NUM_RECORDS = 2_000_000
OUTPUT_FILE = "orders_2m.json"   # NDJSON: jedna linia = jedno zamówienie

fake = Faker("pl_PL")

# Lista przykładowych produktów do losowania
PRODUCTS = [
    ("Laptop", 3200, 6500),
    ("Telefon", 1200, 4200),
    ("Mysz", 40, 200),
    ("Klawiatura", 80, 600),
    ("Monitor", 500, 2500),
    ("Słuchawki", 100, 900),
    ("Powerbank", 50, 400),
    ("Kabel USB", 10, 80),
]


# ========================================
# FUNKCJE GENERUJĄCE DANE JSON
# ========================================

def generate_item() -> Dict:
    """
    Generuje pojedynczy element zamówienia (pozycja koszyka).
    """
    product_name, min_price, max_price = random.choice(PRODUCTS)
    qty = random.randint(1, 5)
    price = round(random.uniform(min_price, max_price), 2)

    return {
        "product": product_name,
        "qty": qty,
        "price": price,
    }


def generate_items_list() -> List[Dict]:
    """
    Generuje listę pozycji w zamówieniu (1–5 pozycji).
    Dla realizmu dodajemy ~5% pustych list (brak pozycji).
    """
    if random.random() < 0.05:
        return []

    num_items = random.randint(1, 5)
    return [generate_item() for _ in range(num_items)]


def generate_order(order_id: int) -> Dict:
    """
    Generuje pojedyncze zamówienie w formie słownika (dict),
    gotowe do zapisania jako JSON.
    """
    customer_id = random.randint(100_000, 999_999)

    return {
        "order_id": order_id,
        "customer": {
            "id": customer_id,
            "name": fake.first_name(),
            "surname": fake.last_name(),
        },
        "items": generate_items_list(),
        "city": fake.city(),
    }


def generate_json_file(
    output_path: str = OUTPUT_FILE,
    num_records: int = NUM_RECORDS,
    seed: int | None = 42,
) -> None:
    """
    Generuje plik NDJSON z zadanym rozmiarem.
    Każda linia w pliku jest osobnym obiektem JSON.
    """
    if seed is not None:
        random.seed(seed)

    print(
        f"[GENERATOR] Generuję {num_records:,} rekordów do pliku {output_path}..."
        .replace(",", " ")
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(1, num_records + 1):
            order = generate_order(i)
            f.write(json.dumps(order, ensure_ascii=False) + "\n")

            if i % 100_000 == 0:
                print(
                    f"[GENERATOR] Wygenerowano {i:,} rekordów..."
                    .replace(",", " ")
                )

    print(f"[GENERATOR] Gotowe. Plik zapisany jako: {output_path}")


# ========================================
# FUNKCJE ANALIZUJĄCE DANE W PYSPARK
# ========================================

def run_spark_analysis(json_path: str = OUTPUT_FILE) -> None:
    """
    Odczyt dużego pliku NDJSON w PySpark i wykonanie podstawowych analiz.
    """
    print(f"[SPARK] Start analizy pliku: {json_path}")

    spark = (
        SparkSession.builder
        .appName("BigJSONAnalysis")
        .getOrCreate()
    )

    # 1. Odczyt pół-strukturalnych danych JSON
    df = spark.read.json(json_path)

    print("[SPARK] Schemat danych:")
    df.printSchema()

    print("[SPARK] Przykładowe rekordy:")
    df.show(5, truncate=False)

    # 2. Eksplozja tablicy items (dane towarów w zamówieniu)
    df_items = (
        df
        .select("order_id", F.explode("items").alias("item"))
        .select(
            "order_id",
            F.col("item.product").alias("product"),
            F.col("item.qty").alias("qty"),
            F.col("item.price").alias("price"),
        )
    )

    print("[SPARK] Przykładowe pozycje zamówień (items):")
    df_items.show(5, truncate=False)

    # 3. Wyliczenie wartości zamówienia (sum(qty * price) dla każdego order_id)
    df_order_values = (
        df_items
        .groupBy("order_id")
        .agg(
            F.sum(F.col("qty") * F.col("price")).alias("order_value")
        )
    )

    # 4. Połączenie z oryginalną tabelą (żeby mieć np. miasto)
    df_orders_enriched = (
        df
        .select("order_id", "city")
        .join(df_order_values, on="order_id", how="left")
        .withColumn(
            "order_value",
            F.coalesce(F.col("order_value"), F.lit(0.0))
        )
    )

    # 5. Liczba rekordów / zamówień
    total_orders = df_orders_enriched.count()
    print(f"[SPARK] Liczba zamówień: {total_orders:,}".replace(",", " "))

    # 6. Średnia wartość zamówienia
    avg_order_value = (
        df_orders_enriched
        .agg(F.avg("order_value").alias("avg_order_value"))
        .collect()[0]["avg_order_value"]
    )
    print(f"[SPARK] Średnia wartość zamówienia: {avg_order_value:.2f}")

    # 7. TOP 10 miast wg liczby zamówień
    print("[SPARK] TOP 10 miast wg liczby zamówień:")
    (
        df_orders_enriched
        .groupBy("city")
        .agg(F.count("*").alias("orders_count"))
        .orderBy(F.desc("orders_count"))
        .show(10, truncate=False)
    )

    # 8. TOP 10 miast wg sumarycznej wartości zamówień
    print("[SPARK] TOP 10 miast wg sumarycznej wartości zamówień:")
    (
        df_orders_enriched
        .groupBy("city")
        .agg(F.sum("order_value").alias("total_revenue"))
        .orderBy(F.desc("total_revenue"))
        .show(10, truncate=False)
    )

    # 9. TOP 10 produktów wg przychodu
    print("[SPARK] TOP 10 produktów wg przychodu:")
    (
        df_items
        .groupBy("product")
        .agg(
            F.sum(F.col("qty") * F.col("price")).alias("revenue"),
            F.sum("qty").alias("total_qty"),
        )
        .orderBy(F.desc("revenue"))
        .show(10, truncate=False)
    )

    # 10. Opis statystyczny wartości zamówień (min, max, mean, stddev, percentyle)
    print("[SPARK] Opis statystyczny wartości zamówień:")
    df_orders_enriched.select("order_value").describe().show()

    # Można też policzyć przybliżone percentyle:
    approx_quantiles = df_orders_enriched.approxQuantile(
        "order_value",
        probabilities=[0.25, 0.5, 0.75, 0.9, 0.99],
        relativeError=0.01,
    )
    print("[SPARK] Przybliżone percentyle wartości zamówień (25%, 50%, 75%, 90%, 99%):")
    print(approx_quantiles)

    # Zakończenie sesji Spark
    spark.stop()
    print("[SPARK] Analiza zakończona.")


# ========================================
# GŁÓWNY BLOK URUCHAMIAJĄCY
# ========================================

if __name__ == "__main__":
    # 1. Generowanie pliku JSON (jeśli jeszcze nie istnieje)
    if not os.path.exists(OUTPUT_FILE):
        generate_json_file(OUTPUT_FILE, NUM_RECORDS)
    else:
        print(f"[MAIN] Plik {OUTPUT_FILE} już istnieje – pomijam generowanie.")

    # 2. Analiza w PySpark
    run_spark_analysis(OUTPUT_FILE)

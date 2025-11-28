# =====================================================================
# PRZYKŁAD BIG DATA: DANE NIESTRUKTURYZOWANE (LOGI HTTP) W PYTHON + PYSPARK
# =====================================================================
# Cel:
# 1. Wczytanie danych niestrukturyzowanych (surowy tekst – logi HTTP)
# 2. Parsowanie tekstu do postaci strukturalnej (DataFrame z kolumnami)
# 3. Prosta analiza: liczniki, TOP URL-e, odsetek błędów
# =====================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# -------------------------------------------------
# 1. Start SparkSession
# -------------------------------------------------
spark = (
    SparkSession.builder
    .appName("UnstructuredDataExample")
    .getOrCreate()
)

# -------------------------------------------------
# 2. Przykładowe dane niestrukturyzowane (logi HTTP)
#    W praktyce: zamiast listy poniżej byłby plik na HDFS/S3
# -------------------------------------------------
raw_logs = [
    '192.168.0.10 - - [27/Nov/2025:10:15:32 +0100] "GET /index.html HTTP/1.1" 200 1234',
    '192.168.0.11 - - [27/Nov/2025:10:16:01 +0100] "GET /produkty/15 HTTP/1.1" 200 8543',
    '192.168.0.12 - - [27/Nov/2025:10:16:15 +0100] "POST /api/login HTTP/1.1" 302 512',
    '192.168.0.13 - - [27/Nov/2025:10:16:45 +0100] "GET /nie-istnieje HTTP/1.1" 404 321',
    '192.168.0.14 - - [27/Nov/2025:10:17:02 +0100] "GET /index.html HTTP/1.1" 500 0'
]

# Tworzymy DataFrame z jedną kolumną "value" zawierającą całe linie logów
logs_df = spark.createDataFrame(raw_logs, "string").toDF("value")

print("=== Surowe dane niestrukturyzowane (tekst) ===")
logs_df.show(truncate=False)

# W wersji produkcyjnej byłoby np.:
# logs_df = spark.read.text("hdfs:///data/logs/access_log_*.txt")


# -------------------------------------------------
# 3. Wzorzec regex do parsowania linii loga
#    Typowy format: Common Log Format / Combined Log Format
# -------------------------------------------------
log_pattern = r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (.*?) (HTTP/\d\.\d)" (\d{3}) (\S+)'

# Wyjaśnienie grup:
# 1: IP klienta              -> (\S+)
# 2: znacznik czasu          -> \[([^\]]+)\]
# 3: metoda HTTP             -> "(\S+)
# 4: URL                     -> (.*?)
# 5: wersja HTTP             -> (HTTP/\d\.\d)
# 6: kod statusu             -> (\d{3})
# 7: rozmiar odpowiedzi      -> (\S+)


# -------------------------------------------------
# 4. Parsowanie: z tekstu do kolumn DataFrame
# -------------------------------------------------
parsed_df = (
    logs_df
    # IP klienta
    .withColumn("ip",        F.regexp_extract("value", log_pattern, 1))
    # znacznik czasu jako tekst (można potem sparsować na timestamp)
    .withColumn("timestamp", F.regexp_extract("value", log_pattern, 2))
    # metoda HTTP (GET, POST, itd.)
    .withColumn("method",    F.regexp_extract("value", log_pattern, 3))
    # URL zasobu
    .withColumn("url",       F.regexp_extract("value", log_pattern, 4))
    # wersja HTTP
    .withColumn("http_ver",  F.regexp_extract("value", log_pattern, 5))
    # kod statusu (int)
    .withColumn("status",    F.regexp_extract("value", log_pattern, 6).cast("int"))
    # rozmiar odpowiedzi (long)
    .withColumn("size",      F.regexp_extract("value", log_pattern, 7).cast("long"))
)

print("=== Dane po parsowaniu (strukturalne kolumny) ===")
parsed_df.select("ip", "timestamp", "method", "url", "status", "size").show(truncate=False)


# -------------------------------------------------
# 5. Analiza 1: Liczba żądań w podziale na kod statusu
# -------------------------------------------------
status_counts = (
    parsed_df
    .groupBy("status")
    .count()
    .orderBy("status")
)

print("=== Liczba żądań wg kodu statusu ===")
status_counts.show()


# -------------------------------------------------
# 6. Analiza 2: TOP URL-e wg liczby wywołań
# -------------------------------------------------
top_urls = (
    parsed_df
    .groupBy("url")
    .agg(F.count("*").alias("hits"))
    .orderBy(F.desc("hits"))
)

print("=== TOP URL-e wg liczby hitów ===")
top_urls.show(truncate=False)


# -------------------------------------------------
# 7. Analiza 3: Odsetek błędów (4xx + 5xx)
# -------------------------------------------------
total_requests = parsed_df.count()

errors_df = parsed_df.filter(
    (F.col("status") >= 400) & (F.col("status") < 600)
)
error_count = errors_df.count()

error_rate = error_count / total_requests if total_requests > 0 else 0.0

print("=== Podsumowanie błędów ===")
print(f"Liczba wszystkich żądań  : {total_requests}")
print(f"Liczba błędów (4xx/5xx)  : {error_count}")
print(f"Error rate               : {error_rate:.2%}")

# -------------------------------------------------
# 8. Zakończenie pracy Spark (opcjonalnie)
# -------------------------------------------------
# spark.stop()

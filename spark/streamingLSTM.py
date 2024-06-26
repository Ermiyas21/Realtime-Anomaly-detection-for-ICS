import pyspark
import numpy as np
import pandas as pd
from keras.models import load_model
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from influxdb_client import InfluxDBClient, Point, WriteOptions
import configparser
import time

# Initialize the ConfigParser
config = configparser.ConfigParser()
config.read('/sparkScripts/config.ini')

# Access the configuration variables
influxdb_url = config['InfluxDB']['influxdb_url']
token = config['InfluxDB']['token']
org = config['InfluxDB']['org']
bucket_ = config['InfluxDB']['bucket']
measurementlstm = config['InfluxDB']['measurementlstm']

# Define the window size for the LSTM
seq_size = 1

# Start Spark session
spark = SparkSession.builder.master('local').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Define the schema for the incoming data
schema = StructType([
    StructField("FIT101", StringType(), True),
    StructField("LIT101", StringType(), True),
    StructField("AIT201", StringType(), True),
    StructField("AIT202", StringType(), True),
    StructField("AIT203", StringType(), True),
    StructField("FIT201", StringType(), True),
    StructField("DPIT301", StringType(), True),
    StructField("FIT301", StringType(), True),
    StructField("LIT301", StringType(), True),
    StructField("AIT401", StringType(), True),
    StructField("AIT402", StringType(), True),
    StructField("FIT401", StringType(), True),
    StructField("LIT401", StringType(), True),
    StructField("AIT501", StringType(), True),
    StructField("AIT502", StringType(), True),
    StructField("AIT503", StringType(), True),
    StructField("AIT504", StringType(), True),
    StructField("FIT501", StringType(), True),
    StructField("FIT502", StringType(), True),
    StructField("FIT503", StringType(), True),
    StructField("FIT504", StringType(), True),
    StructField("PIT501", StringType(), True),
    StructField("PIT502", StringType(), True),
    StructField("PIT503", StringType(), True),
    StructField("FIT601", StringType(), True),
    StructField("MV101_1", StringType(), True),
    StructField("MV101_2", StringType(), True),
    StructField("P101_2", StringType(), True),
    StructField("P102_2", StringType(), True),
    StructField("MV201_1", StringType(), True),
    StructField("MV201_2", StringType(), True),
    StructField("P201_2", StringType(), True),
    StructField("P203_2", StringType(), True),
    StructField("P204_2", StringType(), True),
    StructField("P205_2", StringType(), True),
    StructField("P206_2", StringType(), True),
    StructField("MV301_1", StringType(), True),
    StructField("MV301_2", StringType(), True),
    StructField("MV302_1", StringType(), True),
    StructField("MV302_2", StringType(), True),
    StructField("MV303_1", StringType(), True),
    StructField("MV303_2", StringType(), True),
    StructField("MV304_1", StringType(), True),
    StructField("MV304_2", StringType(), True),
    StructField("P301_2", StringType(), True),
    StructField("P302_2", StringType(), True),
    StructField("P402_2", StringType(), True),
    StructField("P403_2", StringType(), True),
    StructField("UV401_2", StringType(), True),
    StructField("P501_2", StringType(), True),
    StructField("P602_2", StringType(), True),
    StructField("label", StringType(), True)
])

# Load the Kafka streaming data
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "thesis-test-topic") \
    .option("startingOffsets", "earliest") \
    .load()

df = df.withColumn("value", df["value"].cast(StringType()))
df = df.withColumn("data", from_json(col("value"), schema)).select("data.*")

# Schema mapping to cast types
schema_mapping = {
    "FIT101": "double", "LIT101": "double", "AIT201": "double", "AIT202": "double", "AIT203": "double", 
    "FIT201": "double", "DPIT301": "double", "FIT301": "double", "LIT301": "double", "AIT401": "double",
    "AIT402": "double", "FIT401": "double", "LIT401": "double", "AIT501": "double", "AIT502": "double",
    "AIT503": "double", "AIT504": "double", "FIT501": "double", "FIT502": "double", "FIT503": "double",
    "FIT504": "double", "PIT501": "double", "PIT502": "double", "PIT503": "double", "FIT601": "double",
    "MV101_1": "double", "MV101_2": "double", "P101_2": "double", "P102_2": "double", "MV201_1": "double",
    "MV201_2": "double", "P201_2": "double", "P203_2": "double", "P204_2": "double", "P205_2": "double",
    "P206_2": "double", "MV301_1": "double", "MV301_2": "double", "MV302_1": "double", "MV302_2": "double",
    "MV303_1": "double", "MV303_2": "double", "MV304_1": "double", "MV304_2": "double", "P301_2": "double",
    "P302_2": "double", "P402_2": "double", "P403_2": "double", "UV401_2": "double", "P501_2": "double",
    "P602_2": "double", "label": "double"
}

for col_name, col_type in schema_mapping.items():
    df = df.withColumn(col_name, df[col_name].cast(col_type))

# Load the pre-trained Autoencoder model
model = load_model('/sparkScripts/models/model_LSTM_SWaT_final16.keras')

def reshape_data_udf(features):
    """
    Reshape the features to match LSTM input requirements.
    """
    reshaped_data = features.reshape((features.shape[0], seq_size, features.shape[1]))
    return reshaped_data

def mc_dropout_predict(model, inputs, n_samples=100):
    """
    Perform Monte Carlo predictions using dropout at inference time.
    """
    predictions = np.array([model(inputs, training=True) for _ in range(n_samples)])
    return predictions

# Variables to measure latency and throughput
total_batches = 0
total_data_points = 0
total_processing_time = 0

def predict_and_detect(batch_df):
    global total_batches, total_data_points, total_processing_time
    
    # Start measuring batch processing time
    batch_start_time = time.time()
    
    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = batch_df.toPandas()
    features = np.array(pandas_df.drop(['label'], axis=1))

    # Reshape features for LSTM
    features = reshape_data_udf(features)

    # Perform predictions using the Autoencoder
    base_predictions = model.predict(features)

    # Perform MC dropout predictions to estimate uncertainty
    mc_predictions = mc_dropout_predict(model, features, n_samples=50)

    # Calculate mean and variance of MC predictions
    mean_predictions = mc_predictions.mean(axis=0)
    variance_predictions = mc_predictions.var(axis=0)

    # Calculate reconstruction error based on the base predictions
    reconstruction_error = np.mean(np.power(features - base_predictions, 2), axis=(1, 2))

    # Add results to DataFrame
    pandas_df['anomaly_score'] = reconstruction_error
    pandas_df['uncertainty'] = variance_predictions.mean(axis=(1, 2))  # Mean uncertainty across features

    # Define a threshold for anomaly detection
    threshold = 0.0065
    pandas_df['is_anomaly'] = (pandas_df['anomaly_score'] > threshold).astype(int)

    # Select only the necessary columns
    result_df = pandas_df[['label', 'anomaly_score', 'is_anomaly', 'uncertainty']]

    # End measuring batch processing time
    batch_end_time = time.time()

    # Calculate batch processing time
    batch_processing_time = batch_end_time - batch_start_time
    total_processing_time += batch_processing_time
    total_batches += 1
    
    # Measure latency and throughput
    latency = batch_processing_time
    avg_latency = total_processing_time / total_batches
    total_data_points += len(result_df)
    throughput = total_data_points / total_processing_time
    
    # Print latency and throughput
    print(f"Latency: {latency:.4f} seconds")
    print(f"Average Latency: {avg_latency:.4f} seconds")
    print(f"Throughput: {throughput:.2f} data points per second")
    
    # Convert back to Spark DataFrame
    return spark.createDataFrame(result_df)

def write_to_influxdb(df, bucket_, token):
    client = InfluxDBClient(url=influxdb_url, token=token, org=org)
    write_api = client.write_api(write_options=WriteOptions(batch_size=500, flush_interval=10_000, jitter_interval=2_000, retry_interval=5_000))
    data_points = [Point(measurementlstm).field(col, value) for row in df.collect() for col, value in row.asDict().items()]
    write_api.write(bucket=bucket_, org=org, record=data_points) 
    client.close()

query = df.writeStream.foreachBatch(lambda df, epoch_id: write_to_influxdb(predict_and_detect(df), bucket_, token)) \
    .outputMode("update") \
    .start()

query.awaitTermination() 


    #v spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 /sparkScripts/streamingLSTM.py
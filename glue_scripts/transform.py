import logging
import boto3
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_trunc, count, countDistinct, sum as _sum, round
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_spark_session():
    """Create and return a Spark session"""
    return SparkSession.builder \
        .appName("GlueParquetReader") \
        .getOrCreate()

def get_secret(secret_name, region_name="us-east-1", secrets_client=None):
    """
    Retrieve secrets from AWS Secrets Manager
    Args:
        secret_name: Name of the secret
        region_name: AWS region
        secrets_client: Optional pre-configured boto3 client (for testing)
    """
    try:
        logger.info(f"Retrieving secret: {secret_name}")
        client = secrets_client or boto3.session.Session().client(
            service_name='secretsmanager',
            region_name=region_name
        )

        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return json.loads(secret)
        else:
            logger.error("Secret not found in string format")
            raise Exception("Secret not in string format")
            
    except ClientError as e:
        logger.error(f"Error retrieving secret: {e}", exc_info=True)
        raise e

def transform_data(df):
    """
    Perform data transformations on the input DataFrame
    Args:
        df: Input Spark DataFrame
    Returns:
        Transformed Spark DataFrame
    """
    df.createOrReplaceTempView("mockrecord")
    sql_query = """
        SELECT 
            user_name,
            email,
            DATE_TRUNC('week', start_date_time) AS week_start,
            COUNT(*) AS total_calls,  
            COUNT(DISTINCT DATE(start_date_time)) AS practice_days,  
            ROUND(SUM(total_duration) / NULLIF(COUNT(DISTINCT DATE(start_date_time)), 0) / 60, 0) AS avg_duration_per_day,
            ROUND(COUNT(*) / NULLIF(COUNT(DISTINCT DATE(start_date_time)), 0), 2) AS calls_per_day
        FROM 
            mockrecord
        GROUP BY 
            user_name, email, DATE_TRUNC('week', start_date_time)
        ORDER BY 
            week_start, user_name, total_calls DESC
    """
    return df.sparkSession.sql(sql_query)

def write_to_postgres(df, secret_name="postgres/blindflavors", secrets_client=None):
    """
    Write DataFrame to PostgreSQL
    Args:
        df: DataFrame to write
        secret_name: Name of the secret containing connection details
        secrets_client: Optional pre-configured boto3 client (for testing)
    """
    try:
        secret = get_secret(secret_name, secrets_client=secrets_client)
        
        postgres_url = f"jdbc:postgresql://{secret['host']}:{secret['port']}/{secret['dbname']}"
        postgres_properties = {
            "user": secret['username'],
            "password": secret['password'],
            "driver": "org.postgresql.Driver"
        }

        df.write.mode("overwrite").jdbc(
            url=postgres_url, 
            table=secret['table'], 
            properties=postgres_properties
        )
        logger.info("Successfully wrote the summary table to PostgreSQL.")
    except Exception as e:
        logger.error(f"Failed to write to PostgreSQL: {e}", exc_info=True)
        raise

def process_file(spark, bucket_name, object_key):
    """
    Process a single file from S3
    Args:
        spark: Spark session
        bucket_name: S3 bucket name
        object_key: S3 object key
    """
    try:
        s3_path = f"s3://{bucket_name}/{object_key}"
        df = spark.read.parquet(s3_path)
        transformed_df = transform_data(df)
        transformed_df.show()
        write_to_postgres(transformed_df)
    except Exception as e:
        logger.error(f"Error processing file {object_key}: {e}", exc_info=True)
        raise

def main():
    """Main entry point for the script"""
    spark = create_spark_session()
    try:
        process_file(
            spark,
            bucket_name='data-engineering-sam0612',
            object_key='rawData/part-00000-120ab73d-7a35-4729-8084-2279486ed76f-c000.snappy.parquet'
        )
    finally:
        spark.stop()
        logger.info("Spark session stopped.")

if __name__ == "__main__":
    logger.info("Starting the main function.")
    main()
    logger.info("Glue job completed.")

    ## Hi Your script is working from bitbucket.
import pytest
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, TimestampType
from pyspark.sql.functions import col, round
from chispa.dataframe_comparer import assert_df_equality
from glue_scripts.transform import transform_data

def print_header(title):
    """Helper to print formatted section headers"""
    print(f"\n{'='*50}")
    print(f"{' '*10}{title.upper()}{' '*10}")
    print(f"{'='*50}\n")

def print_dataframe(df, name):
    """Helper function to print dataframe schema and content"""
    print_header(name)
    print("Schema:")
    print(df.schema)
    print("\nContent:")
    df.show(truncate=False, n=100)
    print(f"\nTotal rows: {df.count()}")

@pytest.fixture(scope="session")
def spark():
    print_header("Creating Spark Session")
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("test") \
        .getOrCreate()
    print("✅ Spark session created successfully")
    return spark

def test_transform_data(spark):
    print_header("Starting Test: Data Transformation")
    
    # Test input data with explanations
    print("\nTest Data Explanation:")
    print("- 2022-12-31 23:59:59 = Saturday (week starts 2022-12-26)")
    print("- 2023-01-01 00:00:00 = Sunday (same week in Spark)")
    print("- 2023-01-01 09:00:00 = Sunday (same week)")
    
    input_data = [
        ("user1", "user1@test.com", "2022-12-31 23:59:59", 3600),
        ("user1", "user1@test.com", "2023-01-01 00:00:00", 1800),
        ("user2", "user2@test.com", "2023-01-01 09:00:00", 7200)
    ]
    
    print("\nCreating input DataFrame with:")
    for i, row in enumerate(input_data, 1):
        print(f"Row {i}: User='{row[0]}', Date='{row[2]}', Duration={row[3]}s")
    
    input_df = spark.createDataFrame(
        input_data, 
        ["user_name", "email", "start_date_time", "total_duration"]
    )
    print_dataframe(input_df, "Input DataFrame")
    
    # Expected output with calculations explained
    print("\nExpected Results Calculation:")
    print("user1: 2 calls across 2 days (Dec 31 + Jan 1)")
    print("- Total calls: 2")
    print("- Practice days: 2")
    print("- Avg duration: (3600 + 1800)/2 = 2700s → 45 mins")
    print("- Calls per day: 2/2 = 1.0")
    print("\nuser2: 1 call on Jan 1")
    print("- Total calls: 1")
    print("- Practice days: 1")
    print("- Avg duration: 7200s → 120 mins")
    print("- Calls per day: 1/1 = 1.0")
    
    expected_data = [
        ("user1", "user1@test.com", datetime(2022, 12, 26), 2, 2, 45.0, 1.0),
        ("user2", "user2@test.com", datetime(2022, 12, 26), 1, 1, 120.0, 1.0)
    ]
    
    expected_schema = StructType([
        StructField("user_name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("week_start", TimestampType(), True),
        StructField("total_calls", LongType(), False),
        StructField("practice_days", LongType(), False),
        StructField("avg_duration_per_day", DoubleType(), True),
        StructField("calls_per_day", DoubleType(), True)
    ])
    
    expected_df = spark.createDataFrame(expected_data, schema=expected_schema)
    print_dataframe(expected_df, "Expected DataFrame")
    
    # Apply transformation
    print("\nApplying transform_data function...")
    result_df = transform_data(input_df)
    print_dataframe(result_df, "Transformed Result DataFrame")
    
    # Detailed comparison
    print_header("DataFrame Comparison")
    print("Week start dates should be Sundays (Spark's default week starts on Sunday)")
    
    # Assert equality
    print("\nStarting DataFrame comparison...")
    try:
        assert_df_equality(
            result_df,
            expected_df,
            ignore_nullable=True,
            ignore_column_order=True
        )
        print("✅ DataFrame comparison successful - all values match!")
    except AssertionError as e:
        print("❌ DataFrame comparison failed! Details below:")
        print(str(e))
        raise
    finally:
        print("\nTest completed")

def test_transform_empty_data(spark):
    print_header("Starting Test: Empty Data Transformation")
    
    empty_schema = StructType([
        StructField("user_name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("start_date_time", StringType(), True),
        StructField("total_duration", LongType(), True)
    ])
    
    print("\nCreating empty input DataFrame")
    input_df = spark.createDataFrame([], schema=empty_schema)
    print_dataframe(input_df, "Empty Input DataFrame")
    
    print("\nApplying transform_data to empty DataFrame...")
    result_df = transform_data(input_df)
    print_dataframe(result_df, "Empty Result DataFrame")
    
    # Expected output schema
    expected_schema = StructType([
        StructField("user_name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("week_start", TimestampType(), True),
        StructField("total_calls", LongType(), False),
        StructField("practice_days", LongType(), False),
        StructField("avg_duration_per_day", DoubleType(), True),
        StructField("calls_per_day", DoubleType(), True)
    ])
    
    expected_df = spark.createDataFrame([], schema=expected_schema)
    print_dataframe(expected_df, "Empty Expected DataFrame")
    
    # Assert equality
    print("\nStarting empty DataFrame comparison...")
    try:
        assert_df_equality(result_df, expected_df)
        print("✅ Empty DataFrame comparison successful")
    except AssertionError as e:
        print("❌ Empty DataFrame comparison failed!")
        print(str(e))
        raise
    finally:
        print("\nTest completed")
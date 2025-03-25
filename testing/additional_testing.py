import pytest
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, TimestampType
from pyspark.sql.functions import lit,col
from chispa.dataframe_comparer import assert_df_equality
from glue_scripts.transform import transform_data
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_header(title):
    """Helper to print formatted section headers"""
    logger.info(f"\n{'='*80}")
    logger.info(f"{' '*15}{title.upper()}{' '*15}")
    logger.info(f"{'='*80}\n")

def print_dataframe(df, name):
    """Helper function to print dataframe schema and content"""
    print_header(name)
    logger.info("Schema:")
    logger.info(df.schema)
    logger.info("\nContent:")
    df.show(truncate=False, n=100)
    logger.info(f"\nTotal rows: {df.count()}")

@pytest.fixture(scope="session")
def spark():
    print_header("Creating Spark Session")
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("pytest-spark") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.sql.session.timeZone", "UTC") \
        .getOrCreate()
    logger.info("✅ Spark session created successfully")
    yield spark
    spark.stop()
    logger.info("✅ Spark session stopped")

## --------------------------
## Core Transformation Tests
## --------------------------

def test_transform_data(spark):
    """Test normal case with valid data"""
    print_header("Starting Test: Data Transformation - Normal Case")
    
    input_data = [
        ("user1", "user1@test.com", "2022-12-31 23:59:59", 3600),  # Saturday
        ("user1", "user1@test.com", "2023-01-01 00:00:00", 1800),  # Sunday (same week)
        ("user2", "user2@test.com", "2023-01-01 09:00:00", 7200)   # Sunday
    ]
    
    input_df = spark.createDataFrame(
        input_data, 
        ["user_name", "email", "start_date_time", "total_duration"]
    )
    print_dataframe(input_df, "Input DataFrame")
    
    expected_data = [
        ("user1", "user1@test.com", datetime(2022, 12, 26), 2, 2, 45.0, 1.0),
        ("user2", "user2@test.com", datetime(2022, 12, 26), 1, 1, 120.0, 1.0)
    ]
    
    expected_df = spark.createDataFrame(
        expected_data,
        ["user_name", "email", "week_start", "total_calls", "practice_days", 
         "avg_duration_per_day", "calls_per_day"]
    )
    print_dataframe(expected_df, "Expected DataFrame")
    
    # Execute transformation
    result_df = transform_data(input_df)
    print_dataframe(result_df, "Transformed Result DataFrame")
    
    # Assertions
    assert_df_equality(
        result_df,
        expected_df,
        ignore_nullable=True,
        ignore_column_order=True,
        ignore_row_order=True
    )

## --------------------------
## Edge Case Tests
## --------------------------

def test_empty_data(spark):
    """Test with empty input DataFrame"""
    print_header("Starting Test: Empty Data Transformation")
    
    empty_schema = StructType([
        StructField("user_name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("start_date_time", StringType(), True),
        StructField("total_duration", LongType(), True)
    ])
    
    input_df = spark.createDataFrame([], schema=empty_schema)
    print_dataframe(input_df, "Empty Input DataFrame")
    
    result_df = transform_data(input_df)
    print_dataframe(result_df, "Empty Result DataFrame")
    
    # Verify schema matches expected output
    expected_schema = StructType([
        StructField("user_name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("week_start", TimestampType(), True),
        StructField("total_calls", LongType(), False),
        StructField("practice_days", LongType(), False),
        StructField("avg_duration_per_day", DoubleType(), True),
        StructField("calls_per_day", DoubleType(), True)
    ])
    
    assert result_df.schema == expected_schema
    assert result_df.count() == 0

def test_null_values(spark):
    """Test handling of null values in input"""
    print_header("Starting Test: Null Value Handling")
    
    input_data = [
        ("user1", None, "2023-01-01 00:00:00", 3600),
        (None, "user2@test.com", "2023-01-02 00:00:00", 1800),
        ("user3", "user3@test.com", None, 7200),
        ("user4", "user4@test.com", "2023-01-03 00:00:00", None)
    ]
    
    input_df = spark.createDataFrame(
        input_data, 
        ["user_name", "email", "start_date_time", "total_duration"]
    )
    print_dataframe(input_df, "Input DataFrame with Nulls")
    
    # Execute transformation
    result_df = transform_data(input_df)
    print_dataframe(result_df, "Transformed Result with Nulls")
    
    # Verify null handling behavior
    assert result_df.count() > 0  # Should process records with some nulls
    assert result_df.filter(col("user_name").isNull()).count() == 0  # Should drop null users
    assert result_df.filter(col("week_start").isNull()).count() == 0  # Should drop null dates

## --------------------------
## Stress/Volume Tests
## --------------------------

def test_large_dataset(spark):
    """Test with large dataset to check performance and memory handling"""
    print_header("Starting Test: Large Dataset")
    
    # Generate 10,000 random records
    users = [f"user{i}" for i in range(1, 101)]
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(0, 100)]
    
    input_data = []
    for _ in range(10000):
        user = random.choice(users)
        email = f"{user}@test.com"
        date = random.choice(dates).strftime("%Y-%m-%d %H:%M:%S")
        duration = random.randint(60, 3600)
        input_data.append((user, email, date, duration))
    
    input_df = spark.createDataFrame(
        input_data,
        ["user_name", "email", "start_date_time", "total_duration"]
    )
    logger.info(f"Created large DataFrame with {input_df.count()} rows")
    
    # Execute transformation with timing
    start_time = datetime.now()
    result_df = transform_data(input_df)
    elapsed = datetime.now() - start_time
    
    logger.info(f"Transformation completed in {elapsed.total_seconds():.2f} seconds")
    print_dataframe(result_df.limit(20), "Sample of Transformed Results (20 rows)")
    
    # Basic validation
    assert result_df.count() > 0
    assert len(result_df.columns) == 7
    assert result_df.filter(col("total_calls").isNull()).count() == 0

## --------------------------
## Business Logic Tests
## --------------------------

def test_week_boundary_cases(spark):
    """Test behavior around week boundaries"""
    print_header("Starting Test: Week Boundary Cases")
    
    # Data spanning two weeks with edge cases
    input_data = [
        # Week 1 (starts 2022-12-26)
        ("user1", "user1@test.com", "2022-12-31 23:59:59", 3600),  # Saturday
        ("user1", "user1@test.com", "2023-01-01 00:00:00", 1800),   # Sunday (new week)
        
        # Week 2 (starts 2023-01-02)
        ("user1", "user1@test.com", "2023-01-02 00:00:00", 2400),   # Monday
        ("user2", "user2@test.com", "2023-01-07 23:59:59", 1200),   # Saturday
        ("user2", "user2@test.com", "2023-01-08 00:00:00", 3000)    # Sunday (new week)
    ]
    
    input_df = spark.createDataFrame(
        input_data, 
        ["user_name", "email", "start_date_time", "total_duration"]
    )
    print_dataframe(input_df, "Week Boundary Input Data")
    
    expected_data = [
        ("user1", "user1@test.com", datetime(2022, 12, 26), 1, 1, 60.0, 1.0),  # Week 1
        ("user1", "user1@test.com", datetime(2023, 1, 2), 2, 2, 35.0, 1.0),    # Week 2 (avg of 1800+2400 = 2100s → 35min)
        ("user2", "user2@test.com", datetime(2023, 1, 2), 1, 1, 20.0, 1.0),    # Week 2
        ("user2", "user2@test.com", datetime(2023, 1, 8), 1, 1, 50.0, 1.0)     # Week 3
    ]
    
    expected_df = spark.createDataFrame(
        expected_data,
        ["user_name", "email", "week_start", "total_calls", "practice_days", 
         "avg_duration_per_day", "calls_per_day"]
    )
    print_dataframe(expected_df, "Expected Week Boundary Results")
    
    result_df = transform_data(input_df)
    print_dataframe(result_df, "Actual Week Boundary Results")
    
    assert_df_equality(
        result_df,
        expected_df,
        ignore_nullable=True,
        ignore_column_order=True,
        ignore_row_order=True
    )

def test_single_user_multiple_weeks(spark):
    """Test single user's data across multiple weeks"""
    print_header("Starting Test: Single User Across Multiple Weeks")
    
    input_data = [
        ("user1", "user1@test.com", "2023-01-01 00:00:00", 1800),  # Week 1
        ("user1", "user1@test.com", "2023-01-02 00:00:00", 2400),   # Week 1
        ("user1", "user1@test.com", "2023-01-08 00:00:00", 3000),   # Week 2
        ("user1", "user1@test.com", "2023-01-15 00:00:00", 3600)    # Week 3
    ]
    
    input_df = spark.createDataFrame(
        input_data, 
        ["user_name", "email", "start_date_time", "total_duration"]
    )
    
    result_df = transform_data(input_df)
    print_dataframe(result_df, "Multi-Week Results")
    
    # Verify we have 3 weeks of data for user1
    assert result_df.filter(col("user_name") == "user1").count() == 3
    # Verify each week has correct practice_days
    assert result_df.filter(col("week_start") == datetime(2023, 1, 1)).first().practice_days == 2
    assert result_df.filter(col("week_start") == datetime(2023, 1, 8)).first().practice_days == 1
    assert result_df.filter(col("week_start") == datetime(2023, 1, 15)).first().practice_days == 1

## --------------------------
## Data Quality Tests
## --------------------------

def test_negative_duration(spark):
    """Test handling of negative duration values"""
    print_header("Starting Test: Negative Duration Values")
    
    input_data = [
        ("user1", "user1@test.com", "2023-01-01 00:00:00", -3600),
        ("user2", "user2@test.com", "2023-01-02 00:00:00", 1800)
    ]
    
    input_df = spark.createDataFrame(
        input_data, 
        ["user_name", "email", "start_date_time", "total_duration"]
    )
    print_dataframe(input_df, "Input with Negative Duration")
    
    result_df = transform_data(input_df)
    print_dataframe(result_df, "Transformed Results")
    
    # Verify negative duration was either filtered or handled
    assert result_df.filter(col("user_name") == "user1").count() == 0  # Should exclude negative durations
    assert result_df.filter(col("user_name") == "user2").count() == 1  # Valid record remains

def test_malformed_dates(spark):
    """Test handling of malformed date strings"""
    print_header("Starting Test: Malformed Date Handling")
    
    input_data = [
        ("user1", "user1@test.com", "2023-01-01 00:00:00", 3600),  # Valid
        ("user2", "user2@test.com", "not-a-date", 1800),             # Invalid
        ("user3", "user3@test.com", "2023/01/01 00-00-00", 2400)    # Wrong format
    ]
    
    input_df = spark.createDataFrame(
        input_data, 
        ["user_name", "email", "start_date_time", "total_duration"]
    )
    print_dataframe(input_df, "Input with Malformed Dates")
    
    result_df = transform_data(input_df)
    print_dataframe(result_df, "Transformed Results")
    
    # Verify only valid records processed
    assert result_df.count() == 1
    assert result_df.first().user_name == "user1"
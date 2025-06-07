#!/usr/bin/env python3
"""
Test connections to all data infrastructure components.
This script verifies that all components are running and accessible.
"""

import os
import sys
import time
import psycopg2
import redis
import nats
import asyncio
from nats.js import JetStreamContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set default values if environment variables are not set
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "gcp_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "gcp_password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "gcp")

TIMESCALEDB_HOST = os.getenv("TIMESCALEDB_HOST", "timescaledb")
TIMESCALEDB_PORT = int(os.getenv("TIMESCALEDB_PORT", "5432"))
TIMESCALEDB_USER = os.getenv("TIMESCALEDB_USER", "gcp_user")
TIMESCALEDB_PASSWORD = os.getenv("TIMESCALEDB_PASSWORD", "gcp_password")
TIMESCALEDB_DB = os.getenv("TIMESCALEDB_DB", "gcp_timeseries")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "gcp_password")

NATS_HOST = os.getenv("NATS_HOST", "nats")
NATS_PORT = int(os.getenv("NATS_PORT", "4222"))
NATS_USER = os.getenv("NATS_USER", "gcp_user")
NATS_PASSWORD = os.getenv("NATS_PASSWORD", "gcp_password")


def test_postgres_connection():
    """Test connection to PostgreSQL."""
    print("\n=== Testing PostgreSQL Connection ===")
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DB
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"PostgreSQL Version: {version[0]}")
        
        # Test schemas
        cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'public');")
        schemas = cursor.fetchall()
        print("Available schemas:")
        for schema in schemas:
            print(f"  - {schema[0]}")
        
        # Test camera_management.cameras table
        try:
            cursor.execute("SELECT COUNT(*) FROM camera_management.cameras;")
            count = cursor.fetchone()
            print(f"Number of cameras: {count[0]}")
        except Exception as e:
            print(f"Error querying camera_management.cameras: {e}")
        
        cursor.close()
        conn.close()
        print("PostgreSQL connection test: SUCCESS")
        return True
    except Exception as e:
        print(f"PostgreSQL connection test: FAILED - {e}")
        return False


def test_timescaledb_connection():
    """Test connection to TimescaleDB."""
    print("\n=== Testing TimescaleDB Connection ===")
    try:
        conn = psycopg2.connect(
            host=TIMESCALEDB_HOST,
            port=TIMESCALEDB_PORT,
            user=TIMESCALEDB_USER,
            password=TIMESCALEDB_PASSWORD,
            dbname=TIMESCALEDB_DB
        )
        cursor = conn.cursor()
        
        # Check TimescaleDB version
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
        version = cursor.fetchone()
        print(f"TimescaleDB Version: {version[0]}")
        
        # Test hypertables
        cursor.execute("SELECT table_name FROM timescaledb_information.hypertables;")
        hypertables = cursor.fetchall()
        print("Available hypertables:")
        for table in hypertables:
            print(f"  - {table[0]}")
        
        cursor.close()
        conn.close()
        print("TimescaleDB connection test: SUCCESS")
        return True
    except Exception as e:
        print(f"TimescaleDB connection test: FAILED - {e}")
        return False


def test_redis_connection():
    """Test connection to Redis."""
    print("\n=== Testing Redis Connection ===")
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        info = r.info()
        print(f"Redis Version: {info['redis_version']}")
        print(f"Redis Mode: {info['redis_mode']}")
        print(f"Connected Clients: {info['connected_clients']}")
        
        # Test set and get
        r.set("test_key", "test_value")
        value = r.get("test_key")
        print(f"Test key value: {value}")
        
        # Clean up
        r.delete("test_key")
        
        print("Redis connection test: SUCCESS")
        return True
    except Exception as e:
        print(f"Redis connection test: FAILED - {e}")
        return False


async def test_nats_connection():
    """Test connection to NATS."""
    print("\n=== Testing NATS Connection ===")
    try:
        # Connect to NATS
        nc = await nats.connect(
            f"nats://{NATS_USER}:{NATS_PASSWORD}@{NATS_HOST}:{NATS_PORT}"
        )
        
        # Get server info
        server_info = nc.connected_server_info
        print(f"NATS Server: {server_info.server_name}")
        print(f"NATS Version: {server_info.version}")
        print(f"NATS Protocol: {server_info.proto}")
        
        # Test JetStream
        js = nc.jetstream()
        
        # Create a test stream
        try:
            await js.add_stream(name="test_stream", subjects=["test.subject"])
            print("Created test stream")
        except Exception as e:
            print(f"Stream may already exist: {e}")
        
        # Publish a message
        ack = await js.publish("test.subject", b"test message")
        print(f"Published message, sequence: {ack.seq}")
        
        # Subscribe and receive the message
        sub = await js.subscribe("test.subject")
        msg = await sub.next_msg(timeout=1)
        print(f"Received message: {msg.data.decode()}")
        
        # Clean up
        await sub.unsubscribe()
        await js.delete_stream("test_stream")
        
        # Close connection
        await nc.close()
        
        print("NATS connection test: SUCCESS")
        return True
    except Exception as e:
        print(f"NATS connection test: FAILED - {e}")
        return False


async def main():
    """Run all connection tests."""
    print("=== Data Infrastructure Connection Tests ===")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    postgres_success = test_postgres_connection()
    timescaledb_success = test_timescaledb_connection()
    redis_success = test_redis_connection()
    nats_success = await test_nats_connection()
    
    print("\n=== Test Summary ===")
    print(f"PostgreSQL: {'SUCCESS' if postgres_success else 'FAILED'}")
    print(f"TimescaleDB: {'SUCCESS' if timescaledb_success else 'FAILED'}")
    print(f"Redis: {'SUCCESS' if redis_success else 'FAILED'}")
    print(f"NATS: {'SUCCESS' if nats_success else 'FAILED'}")
    
    if postgres_success and timescaledb_success and redis_success and nats_success:
        print("\nAll connection tests passed!")
        return 0
    else:
        print("\nSome connection tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

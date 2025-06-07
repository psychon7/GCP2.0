#!/usr/bin/env python3
"""
GCP Data Infrastructure Client

This module provides a client for connecting to the GCP data infrastructure components:
- PostgreSQL
- TimescaleDB
- Redis
- NATS

Usage:
    from gcp_data_client import GCPDataClient
    
    # Create a client
    client = GCPDataClient()
    
    # Get a PostgreSQL connection
    with client.get_postgres_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM camera_management.cameras")
            rows = cursor.fetchall()
            for row in rows:
                print(row)
    
    # Get a TimescaleDB connection
    with client.get_timescaledb_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM entropy.raw_samples LIMIT 10")
            rows = cursor.fetchall()
            for row in rows:
                print(row)
    
    # Get a Redis connection
    redis_client = client.get_redis_client()
    redis_client.set("test_key", "test_value")
    value = redis_client.get("test_key")
    print(value)
    
    # Get a NATS connection (async)
    async def nats_example():
        nc = await client.get_nats_connection()
        js = nc.jetstream()
        await js.publish("test.subject", b"test message")
        await nc.close()
    
    asyncio.run(nats_example())
"""

import os
import asyncio
import psycopg2
import redis
import nats
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GCPDataClient:
    """Client for connecting to GCP data infrastructure components."""
    
    def __init__(self, env_file=None):
        """Initialize the client.
        
        Args:
            env_file: Path to .env file (optional)
        """
        if env_file:
            load_dotenv(env_file)
        
        # PostgreSQL configuration
        self.postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'user': os.getenv('POSTGRES_USER', 'gcp_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'gcp_password'),
            'dbname': os.getenv('POSTGRES_DB', 'gcp')
        }
        
        # TimescaleDB configuration
        self.timescaledb_config = {
            'host': os.getenv('TIMESCALEDB_HOST', 'timescaledb'),
            'port': int(os.getenv('TIMESCALEDB_PORT', '5432')),
            'user': os.getenv('TIMESCALEDB_USER', 'gcp_user'),
            'password': os.getenv('TIMESCALEDB_PASSWORD', 'gcp_password'),
            'dbname': os.getenv('TIMESCALEDB_DB', 'gcp_timeseries')
        }
        
        # Redis configuration
        self.redis_config = {
            'host': os.getenv('REDIS_HOST', 'redis'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'password': os.getenv('REDIS_PASSWORD', 'gcp_password'),
            'decode_responses': True
        }
        
        # NATS configuration
        self.nats_config = {
            'host': os.getenv('NATS_HOST', 'nats'),
            'port': int(os.getenv('NATS_PORT', '4222')),
            'user': os.getenv('NATS_USER', 'gcp_user'),
            'password': os.getenv('NATS_PASSWORD', 'gcp_password')
        }
    
    def get_postgres_connection(self):
        """Get a PostgreSQL connection.
        
        Returns:
            psycopg2.connection: PostgreSQL connection
        """
        return psycopg2.connect(**self.postgres_config)
    
    def get_timescaledb_connection(self):
        """Get a TimescaleDB connection.
        
        Returns:
            psycopg2.connection: TimescaleDB connection
        """
        return psycopg2.connect(**self.timescaledb_config)
    
    def get_redis_client(self):
        """Get a Redis client.
        
        Returns:
            redis.Redis: Redis client
        """
        return redis.Redis(**self.redis_config)
    
    async def get_nats_connection(self):
        """Get a NATS connection.
        
        Returns:
            nats.NATS: NATS connection
        """
        return await nats.connect(
            f"nats://{self.nats_config['user']}:{self.nats_config['password']}@{self.nats_config['host']}:{self.nats_config['port']}"
        )


# Example usage
if __name__ == "__main__":
    client = GCPDataClient()
    
    # Test PostgreSQL connection
    try:
        with client.get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                print(f"PostgreSQL version: {version[0]}")
        print("PostgreSQL connection successful")
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
    
    # Test TimescaleDB connection
    try:
        with client.get_timescaledb_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
                version = cursor.fetchone()
                print(f"TimescaleDB version: {version[0]}")
        print("TimescaleDB connection successful")
    except Exception as e:
        print(f"TimescaleDB connection failed: {e}")
    
    # Test Redis connection
    try:
        redis_client = client.get_redis_client()
        info = redis_client.info()
        print(f"Redis version: {info['redis_version']}")
        print("Redis connection successful")
    except Exception as e:
        print(f"Redis connection failed: {e}")
    
    # Test NATS connection
    async def test_nats():
        try:
            nc = await client.get_nats_connection()
            server_info = nc.connected_server_info
            print(f"NATS version: {server_info.version}")
            await nc.close()
            print("NATS connection successful")
        except Exception as e:
            print(f"NATS connection failed: {e}")
    
    asyncio.run(test_nats())

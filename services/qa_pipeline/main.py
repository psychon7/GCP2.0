"""
Main entry point for the QA Pipeline service.

This module provides command-line and service functionality for running statistical
tests on entropy samples and calculating quality scores.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import nats
import tomli
from pydantic import ValidationError

from framework.data_types import EntropySample, OverallQualityScore, TestConfiguration
from framework.test_runner import TestRunner


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("qa_pipeline")


class QAPipelineService:
    """Main service class for the QA Pipeline."""
    
    def __init__(self, config_path: str = "config/qa_config.toml"):
        """
        Initialize the QA Pipeline service.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.test_runner = self._initialize_test_runner()
        self.nats_client = None
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from a TOML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Parsed configuration dictionary
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            tomli.TOMLDecodeError: If the TOML file is invalid
        """
        try:
            with open(config_path, "rb") as f:
                return tomli.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            logger.info("Creating default configuration...")
            self._create_default_config(config_path)
            with open(config_path, "rb") as f:
                return tomli.load(f)
    
    def _create_default_config(self, config_path: str) -> None:
        """
        Create a default configuration file.
        
        Args:
            config_path: Path to save the configuration file
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Default configuration
        default_config = """
# QA Pipeline Configuration

[service]
# Service settings
name = "qa_pipeline"
log_level = "INFO"

# NATS Connection settings
[nats]
enabled = true
servers = ["nats://localhost:4222"]
subject_input = "qa.input"
subject_output = "qa.output"
queue_group = "qa_workers"
connect_timeout = 5.0
reconnect_attempts = 5
max_reconnect_attempts = -1  # -1 means unlimited

# Test settings
[tests]
# Minimum number of bits required for tests
min_sample_size = 100

# Available tests
[[tests.test_configs]]
test_id = "nist_frequency.NistFrequencyTest"
enabled = true
parameters = { }
threshold = 0.01  # p-value threshold
weight = 1.0  # weight in the overall score

# Add more tests as needed
# [[tests.test_configs]]
# test_id = "chi_squared.ChiSquaredTest"
# enabled = true
# parameters = { }
# threshold = 0.01
# weight = 1.0

# Node status monitoring
[node_monitoring]
enabled = true
rolling_window_samples = 100  # Number of samples to keep for rolling average
threshold_score = 70.0  # Minimum acceptable rolling score
flagging_period = 10  # Number of consecutive below-threshold scores to flag a node
"""
        
        with open(config_path, "w") as f:
            f.write(default_config.lstrip())
        
        logger.info(f"Default configuration created at {config_path}")
    
    def _initialize_test_runner(self) -> TestRunner:
        """
        Initialize the test runner with configured tests.
        
        Returns:
            Configured TestRunner instance
        """
        test_configs = []
        
        # Convert configuration dict to TestConfiguration objects
        for test_config_dict in self.config.get("tests", {}).get("test_configs", []):
            if test_config_dict.get("enabled", True):
                try:
                    test_configs.append(TestConfiguration(
                        test_id=test_config_dict["test_id"],
                        enabled=test_config_dict.get("enabled", True),
                        parameters=test_config_dict.get("parameters", {}),
                        threshold=test_config_dict.get("threshold"),
                        weight=test_config_dict.get("weight", 1.0)
                    ))
                except (ValidationError, KeyError) as e:
                    logger.error(f"Invalid test configuration: {e}")
        
        return TestRunner(test_configs)
    
    def process_sample(self, sample: EntropySample) -> OverallQualityScore:
        """
        Process an entropy sample through the test pipeline.
        
        Args:
            sample: The entropy sample to process
            
        Returns:
            Overall quality score with individual test results
        """
        # Check if the sample meets minimum size requirements
        min_sample_size = self.config.get("tests", {}).get("min_sample_size", 100)
        
        # Calculate bits depending on the data format
        if isinstance(sample.data, bytes):
            bits = len(sample.data) * 8
        elif isinstance(sample.data, str) and sample.data_format == "binary":
            bits = len(sample.data)
        elif isinstance(sample.data, str) and sample.data_format == "hex":
            bits = len(sample.data) * 4
        elif isinstance(sample.data, list):
            bits = len(sample.data)
        else:
            bits = 0
        
        if bits < min_sample_size:
            logger.warning(f"Sample size {bits} bits is below minimum {min_sample_size}")
            # Could return a partial result or error here
        
        # Process the sample through the test runner
        start_time = time.time()
        result = self.test_runner.run_all_tests(sample)
        processing_time = time.time() - start_time
        
        logger.info(
            f"Sample {sample.sample_id} processed in {processing_time:.2f}s with "
            f"score {result.score:.2f} ({result.passed_tests}/{result.total_tests} tests passed)"
        )
        
        return result
    
    async def connect_nats(self) -> None:
        """
        Connect to NATS server for message processing.
        
        Raises:
            Exception: If connection fails after retries
        """
        if not self.config.get("nats", {}).get("enabled", False):
            logger.info("NATS integration disabled in configuration")
            return
        
        nats_config = self.config.get("nats", {})
        servers = nats_config.get("servers", ["nats://localhost:4222"])
        connect_timeout = nats_config.get("connect_timeout", 5.0)
        
        logger.info(f"Connecting to NATS servers: {servers}")
        
        try:
            self.nats_client = await nats.connect(
                servers=servers,
                connect_timeout=connect_timeout
            )
            logger.info("Connected to NATS server")
        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            raise
    
    async def subscribe_to_input(self) -> None:
        """
        Subscribe to the input subject for entropy samples.
        
        Raises:
            ValueError: If NATS client is not connected
        """
        if not self.nats_client:
            raise ValueError("NATS client not connected")
        
        nats_config = self.config.get("nats", {})
        subject = nats_config.get("subject_input", "qa.input")
        queue_group = nats_config.get("queue_group", "qa_workers")
        
        logger.info(f"Subscribing to {subject} (queue group: {queue_group})")
        
        subscription = await self.nats_client.subscribe(
            subject=subject,
            queue=queue_group,
            cb=self._process_message
        )
        
        logger.info(f"Subscribed to {subject}")
        
        return subscription
    
    async def _process_message(self, msg) -> None:
        """
        Process an incoming NATS message.
        
        Args:
            msg: NATS message object
        """
        try:
            # Parse the message
            data = json.loads(msg.data.decode())
            
            # Generate a sample ID if not provided
            sample_id = data.get("sample_id", str(uuid.uuid4()))
            
            # Create an entropy sample from the message data
            sample = EntropySample(
                sample_id=sample_id,
                data=data.get("data"),
                data_format=data.get("data_format", "binary"),
                source_id=data.get("source_id"),
                timestamp=datetime.fromtimestamp(data.get("timestamp", time.time())),
                metadata=data.get("metadata", {})
            )
            
            # Process the sample
            result = self.process_sample(sample)
            
            # Publish the result
            await self._publish_result(result)
            
            # Acknowledge the message
            await msg.ack()
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Depending on the error, might want to: nack, requeue, publish an error, etc.
    
    async def _publish_result(self, result: OverallQualityScore) -> None:
        """
        Publish test results to the output subject.
        
        Args:
            result: The test results to publish
            
        Raises:
            ValueError: If NATS client is not connected
        """
        if not self.nats_client:
            raise ValueError("NATS client not connected")
        
        nats_config = self.config.get("nats", {})
        subject = nats_config.get("subject_output", "qa.output")
        
        # Convert the result to a JSON-serializable dictionary
        result_dict = result.dict()
        
        # Convert datetime objects to ISO format strings
        result_dict["timestamp"] = result_dict["timestamp"].isoformat()
        
        # Convert test results (which may contain datetime objects)
        for i, test_result in enumerate(result_dict["test_results"]):
            if "timestamp" in test_result and test_result["timestamp"]:
                result_dict["test_results"][i]["timestamp"] = test_result["timestamp"].isoformat()
        
        # Publish the result
        await self.nats_client.publish(
            subject=subject,
            payload=json.dumps(result_dict).encode()
        )
        
        logger.debug(f"Published result for sample {result.sample_id} to {subject}")
    
    async def run(self) -> None:
        """Run the QA Pipeline service until interrupted."""
        try:
            # Connect to NATS if enabled
            if self.config.get("nats", {}).get("enabled", False):
                await self.connect_nats()
                subscription = await self.subscribe_to_input()
                
                # Keep the service running
                logger.info("QA Pipeline service running, press Ctrl+C to exit")
                while True:
                    await asyncio.sleep(1.0)
                    
        except KeyboardInterrupt:
            logger.info("Service interrupted")
        finally:
            # Cleanup
            if self.nats_client:
                await self.nats_client.drain()
                logger.info("NATS connection closed")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="QA Pipeline for entropy data")
    parser.add_argument(
        "--config", 
        default="config/qa_config.toml", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-file", 
        help="Process a single file with entropy data"
    )
    parser.add_argument(
        "--format", 
        choices=["binary", "hex", "json"],
        default="binary", 
        help="Format of the input file"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", 
        help="Logging level"
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    logging.getLogger("qa_pipeline").setLevel(getattr(logging, args.log_level))
    
    # Create the service
    service = QAPipelineService(config_path=args.config)
    
    # If a test file was specified, process it
    if args.test_file:
        with open(args.test_file, "rb") as f:
            data = f.read()
        
        # Create a sample from the file
        if args.format == "binary":
            sample = EntropySample(
                sample_id=str(uuid.uuid4()),
                data=data,
                data_format="binary"
            )
        elif args.format == "hex":
            sample = EntropySample(
                sample_id=str(uuid.uuid4()),
                data=data.decode().strip(),
                data_format="hex"
            )
        elif args.format == "json":
            json_data = json.loads(data.decode())
            sample = EntropySample(**json_data)
        
        # Process the sample and print results
        result = service.process_sample(sample)
        print(json.dumps(result.dict(), indent=2, default=str))
        
    else:
        # Run as a service
        await service.run()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3

"""
Main entry point for the QA Pipeline service.

Provides both CLI functionality for one-off testing and a NATS integration for
service-oriented operation.
"""

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import nats
import tomli
from pydantic import ValidationError

from framework.data_types import EntropySample, TestConfiguration, OverallQualityScore
from framework.test_runner import TestRunner
from scoring.quality_tracker import QualityTracker
from scoring.nats_handler import QualityScoringNatsHandler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("qa_pipeline")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a TOML file.
    
    Args:
        config_path: Path to the TOML config file, or None to use default
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the config file does not exist
        ValueError: If the config file can't be parsed
    """
    if not config_path:
        # Use default config path
        config_path = os.path.join(os.path.dirname(__file__), 
                                  'config', 'qa_config.toml')
    
    # Ensure the path exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        return config
    except tomli.TOMLDecodeError as e:
        raise ValueError(f"Error parsing config file: {e}")


def setup_quality_system(config: Dict[str, Any]) -> tuple[QualityTracker, QualityScoringNatsHandler]:
    """Set up the quality scoring system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (QualityTracker, QualityScoringNatsHandler)
    """
    # Create quality tracker
    quality_tracker = QualityTracker(config)
    
    # Create NATS handler
    nats_handler = QualityScoringNatsHandler(config, quality_tracker)
    
    return quality_tracker, nats_handler


def setup_test_runner(config: Dict[str, Any]) -> TestRunner:
    """Set up the test runner.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TestRunner instance
    """
    test_configs = []
    
    # Convert configuration dict to TestConfiguration objects
    for test_config_dict in config.get("tests", {}).get("test_configs", []):
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


async def process_entropy_sample(test_runner: TestRunner, sample: EntropySample) -> OverallQualityScore:
    """Process a single entropy sample through the test pipeline.
    
    Args:
        test_runner: The test runner to use
        sample: The entropy sample to test
        
    Returns:
        OverallQualityScore with the test results
    """
    # Run asynchronously in case some tests are time-consuming
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, test_runner.run_all_tests, sample)
    
    logger.debug(f"Sample processed with score {result.score:.2f} "
                f"({result.passed_tests}/{result.total_tests} tests passed)")
    
    return result


async def handle_entropy_sample(msg, test_runner: TestRunner, nc, quality_tracker: Optional[QualityTracker] = None):
    """Handle an entropy sample message from NATS.
    
    Args:
        msg: The NATS message
        test_runner: TestRunner instance
        nc: NATS client
        quality_tracker: Optional QualityTracker instance
    """
    subject = msg.subject
    reply_subject = msg.reply
    data = msg.data.decode()
    
    try:
        # Parse the message
        message_data = json.loads(data)
        
        # Extract the sample and metadata
        sample_data = message_data.get("sample")
        correlation_id = message_data.get("correlation_id", "")
        
        if not sample_data:
            logger.error("Received message without sample data")
            return
            
        # Create an entropy sample
        sample = EntropySample(
            data=sample_data.get("data"),
            data_format=sample_data.get("format", "binary"),
            timestamp=sample_data.get("timestamp"),
            metadata=sample_data.get("metadata", {})
        )
        
        # Process the sample
        result = await process_entropy_sample(test_runner, sample)
        
        # Update quality tracker if node_id is provided
        if quality_tracker and sample.metadata and "node_id" in sample.metadata:
            node_id = sample.metadata["node_id"]
            sample_id = sample.metadata.get("id", str(sample.timestamp))
            quality_tracker.record_score(node_id, sample_id, result)
        
        # Publish the result
        await nc.publish(
            f"{reply_subject}",
            json.dumps({
                "correlation_id": correlation_id,
                "result": result.dict()
            }).encode()
        )
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in message: {data}")
    except Exception as e:
        logger.error(f"Error processing entropy sample: {e}")


async def run_service(config: Dict[str, Any]):
    """Run the QA Pipeline as a service.
    
    Connects to NATS and subscribes to entropy sample messages.
    
    Args:
        config: Service configuration
    """
    # Get NATS configuration
    nats_config = config.get("nats", {})
    servers = nats_config.get("servers", ["nats://localhost:4222"])
    subject_prefix = nats_config.get("subject_prefix", "gcp")
    
    # Create test runner
    test_runner = setup_test_runner(config)
    
    # Set up quality scoring system
    quality_tracker, quality_nats_handler = setup_quality_system(config)
    
    # Connect to NATS
    try:
        logger.info(f"Connecting to NATS servers: {servers}")
        nc = await nats.connect(servers=servers)
        
        # Also connect quality scoring handler
        await quality_nats_handler.connect()
    except Exception as e:
        logger.error(f"Failed to connect to NATS: {e}")
        return
        
    logger.info("Connected to NATS")
    
    # Subscribe to subject with wildcard
    entropy_subject = f"{subject_prefix}.entropy.samples.*"
    
    async def message_handler(msg):
        nonlocal test_runner, quality_tracker
        await handle_entropy_sample(msg, test_runner, nc, quality_tracker)
    
    # Subscribe to entropy samples
    sub = await nc.subscribe(entropy_subject, cb=message_handler)
    logger.info(f"Subscribed to {entropy_subject}")
    
    # Start periodic tasks for quality monitoring
    periodic_task = asyncio.create_task(
        quality_nats_handler.run_periodic_tasks()
    )
    
    # Run until interrupted
    try:
        logger.info("QA Pipeline service is running. Press Ctrl+C to exit...")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        # Clean up
        periodic_task.cancel()
        try:
            await periodic_task
        except asyncio.CancelledError:
            pass
            
        await sub.unsubscribe()
        await quality_nats_handler.disconnect()
        await nc.close()
        logger.info("Shutdown complete")


def process_file(file_path: str, config_path: Optional[str] = None) -> None:
    """Process an entropy data file and print results.
    
    Args:
        file_path: Path to the entropy data file
        config_path: Optional path to the configuration file
    """
    # Load config
    config = load_config(config_path)
    
    # Set up test runner
    test_runner = setup_test_runner(config)
    
    try:
        # Read file
        with open(file_path, "rb") as f:
            data = f.read()
        
        # Create sample
        sample = EntropySample(
            data=data,
            data_format="binary",
            timestamp=datetime.now().isoformat()
        )
        
        # Process sample
        result = test_runner.run_all_tests(sample)
        
        # Print results
        print(f"\nOverall Quality Score: {result.score:.2f}/100")
        print(f"Tests passed: {result.passed_tests}/{result.total_tests}")
        print("\nIndividual Test Results:")
        
        for test_result in result.individual_scores:
            status = "✓ PASS" if test_result.passed else "✗ FAIL"
            print(f"  {test_result.test_name}: {test_result.score:.2f} - {status}")
            
            # Print key statistics if present
            if test_result.statistics:
                for key, value in test_result.statistics.items():
                    if key in ["p_value", "chi_squared", "runs", "frequency"]:
                        print(f"    {key}: {value}")
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error processing file: {e}")


def main():
    """Main entry point for the QA Pipeline CLI."""
    parser = argparse.ArgumentParser(description="QA Pipeline for entropy testing")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Service command
    service_parser = subparsers.add_parser("service", help="Run as a service")
    service_parser.add_argument(
        "--config", "-c",
        help="Path to the configuration file",
        default=None
    )
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a single file")
    process_parser.add_argument(
        "file",
        help="Path to the entropy data file"
    )
    process_parser.add_argument(
        "--config", "-c",
        help="Path to the configuration file",
        default=None
    )
    
    args = parser.parse_args()
    
    if args.command == "service":
        # Load config
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
        
        # Run the service
        try:
            asyncio.run(run_service(config))
        except KeyboardInterrupt:
            logger.info("Service stopped by user")
        except Exception as e:
            logger.error(f"Service error: {e}")
            sys.exit(1)
    
    elif args.command == "process":
        process_file(args.file, args.config)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

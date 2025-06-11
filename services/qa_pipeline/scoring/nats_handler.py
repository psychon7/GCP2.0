"""
NATS integration for the quality scoring system.

Handles publishing node status updates and subscribing to relevant topics.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import nats
from nats.aio.client import Client as NatsClient
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.data_types import NodeStatus, NodeStatusRecord, QualityScore
from scoring.quality_tracker import QualityTracker

logger = logging.getLogger(__name__)


class QualityScoringNatsHandler:
    """
    Handles NATS integration for the quality scoring system.
    
    Subscribes to sample scoring results and publishes node status updates.
    """
    
    def __init__(self, config: Dict[str, Any], quality_tracker: QualityTracker):
        """
        Initialize the NATS handler.
        
        Args:
            config: Configuration dictionary
            quality_tracker: Quality tracker instance to use
        """
        self.config = config
        self.tracker = quality_tracker
        
        # NATS client
        self.nats_client: Optional[NatsClient] = None
        self.connected = False
        
        # NATS topics/subjects
        nats_config = config.get("nats", {})
        self.base_subject = nats_config.get("subject_prefix", "gcp")
        self.scores_subject = f"{self.base_subject}.qa.scores"
        self.status_subject = f"{self.base_subject}.qa.node_status"
        self.alerts_subject = f"{self.base_subject}.qa.alerts"
        
        # Track subscriptions
        self.subscriptions = []
        
        # Register for status change notifications
        self.tracker.register_status_change_callback(self.handle_status_change)
        
    async def connect(self):
        """
        Connect to NATS server and set up subscriptions.
        """
        if self.connected:
            return
            
        try:
            # Get NATS connection options from config
            nats_config = self.config.get("nats", {})
            servers = nats_config.get("servers", ["nats://localhost:4222"])
            
            # Connect to NATS
            self.nats_client = await nats.connect(servers=servers)
            logger.info(f"Connected to NATS servers: {servers}")
            self.connected = True
            
            # Set up subscriptions
            await self.setup_subscriptions()
            
        except ErrNoServers as e:
            logger.error(f"Failed to connect to NATS servers: {e}")
            self.connected = False
        except Exception as e:
            logger.error(f"Unexpected error connecting to NATS: {e}")
            self.connected = False
    
    async def disconnect(self):
        """
        Disconnect from NATS server.
        """
        if self.nats_client:
            try:
                # Clean up subscriptions
                for sub in self.subscriptions:
                    await sub.unsubscribe()
                self.subscriptions = []
                
                # Close NATS connection
                await self.nats_client.close()
                logger.info("Disconnected from NATS")
                
            except Exception as e:
                logger.error(f"Error disconnecting from NATS: {e}")
            finally:
                self.connected = False
                self.nats_client = None
    
    async def setup_subscriptions(self):
        """
        Set up NATS subscriptions.
        """
        if not self.nats_client or not self.connected:
            logger.error("Cannot setup subscriptions - not connected to NATS")
            return
        
        try:
            # Subscribe to quality scores
            sub = await self.nats_client.subscribe(
                f"{self.scores_subject}.>", 
                cb=self.on_quality_score
            )
            self.subscriptions.append(sub)
            logger.info(f"Subscribed to {self.scores_subject}.>")
            
        except Exception as e:
            logger.error(f"Error setting up NATS subscriptions: {e}")
    
    async def on_quality_score(self, msg):
        """
        Handle incoming quality score messages.
        
        Args:
            msg: NATS message
        """
        try:
            # Parse message data
            data = json.loads(msg.data.decode())
            
            # Extract required fields
            node_id = data.get("node_id")
            sample_id = data.get("sample_id")
            quality_score = data.get("quality_score")
            
            if not all([node_id, sample_id, quality_score]):
                logger.warning(f"Received incomplete quality score message: {data}")
                return
                
            # Record the score
            self.tracker.record_score(node_id, sample_id, quality_score)
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in quality score message")
        except Exception as e:
            logger.error(f"Error processing quality score message: {e}")
    
    def handle_status_change(self, node_id: str, status_record: NodeStatusRecord):
        """
        Handle a node status change.
        
        Args:
            node_id: ID of the node
            status_record: Updated status record
        """
        # Convert to serializable dict
        status_dict = status_record.dict()
        
        # Convert enums to strings
        status_dict["status"] = str(status_dict["status"])
        if status_dict["previous_status"]:
            status_dict["previous_status"] = str(status_dict["previous_status"])
            
        # Convert datetimes to ISO strings
        status_dict["since"] = status_dict["since"].isoformat()
        status_dict["last_updated"] = status_dict["last_updated"].isoformat()
        
        # Create message
        message = {
            "type": "node_status_change",
            "timestamp": datetime.now().isoformat(),
            "node_id": node_id,
            "status": status_dict
        }
        
        # Publish asynchronously
        asyncio.create_task(self.publish_status_update(message))
        
        # Also publish as alert if flagged
        if status_record.status == NodeStatus.FLAGGED:
            alert_message = {
                "type": "node_flagged_alert",
                "timestamp": datetime.now().isoformat(),
                "node_id": node_id,
                "reason": status_record.message,
                "score": status_record.current_score,
                "consecutive_failures": status_record.consecutive_below_threshold
            }
            asyncio.create_task(self.publish_alert(alert_message))
            
    async def publish_status_update(self, message: Dict[str, Any]):
        """
        Publish a status update message to NATS.
        
        Args:
            message: Message to publish
        """
        if not self.nats_client or not self.connected:
            logger.warning("Cannot publish status update - not connected to NATS")
            return
            
        try:
            # Create subject with node ID
            subject = f"{self.status_subject}.{message['node_id']}"
            
            # Publish message
            await self.nats_client.publish(
                subject,
                json.dumps(message).encode()
            )
            logger.debug(f"Published status update to {subject}")
            
        except ErrConnectionClosed:
            logger.error("NATS connection closed, trying to reconnect...")
            await self.connect()
        except Exception as e:
            logger.error(f"Error publishing status update: {e}")
            
    async def publish_alert(self, message: Dict[str, Any]):
        """
        Publish an alert message to NATS.
        
        Args:
            message: Message to publish
        """
        if not self.nats_client or not self.connected:
            logger.warning("Cannot publish alert - not connected to NATS")
            return
            
        try:
            # Create subject with node ID and alert type
            subject = f"{self.alerts_subject}.{message['node_id']}.{message['type']}"
            
            # Publish message
            await self.nats_client.publish(
                subject,
                json.dumps(message).encode()
            )
            logger.info(f"Published alert to {subject}: {message['reason']}")
            
        except ErrConnectionClosed:
            logger.error("NATS connection closed, trying to reconnect...")
            await self.connect()
        except Exception as e:
            logger.error(f"Error publishing alert: {e}")
            
    async def publish_rolling_metrics(self, node_id: str):
        """
        Publish rolling metrics for a node.
        
        Args:
            node_id: ID of the node
        """
        if not self.nats_client or not self.connected:
            return
            
        try:
            # Get rolling metrics
            metrics = self.tracker.get_rolling_metrics(node_id)
            if not metrics:
                return
                
            # Convert to serializable dict
            metrics_dict = metrics.dict()
            
            # Convert datetimes to ISO strings
            metrics_dict["window_start"] = metrics_dict["window_start"].isoformat()
            metrics_dict["window_end"] = metrics_dict["window_end"].isoformat()
            
            # Create message
            message = {
                "type": "rolling_metrics",
                "timestamp": datetime.now().isoformat(),
                "node_id": node_id,
                "metrics": metrics_dict
            }
            
            # Publish message
            subject = f"{self.base_subject}.qa.metrics.{node_id}"
            await self.nats_client.publish(
                subject,
                json.dumps(message).encode()
            )
            
        except Exception as e:
            logger.error(f"Error publishing rolling metrics: {e}")
            
    async def run_periodic_tasks(self, interval_seconds: int = 60):
        """
        Run periodic tasks for the quality scoring system.
        
        Args:
            interval_seconds: Interval between task runs in seconds
        """
        while True:
            try:
                # Check for offline nodes
                self.tracker.check_offline_nodes()
                
                # Publish rolling metrics for all nodes
                for node_id in self.tracker.node_scores.keys():
                    await self.publish_rolling_metrics(node_id)
                    
            except Exception as e:
                logger.error(f"Error in periodic tasks: {e}")
                
            # Wait for next interval
            await asyncio.sleep(interval_seconds)

"""
FastAPI application for the Quality Assurance Pipeline.

Provides endpoints for submitting entropy data and querying node status.
"""
import asyncio
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import nats

# Assumes the service is run from the `services/qa_pipeline` directory
# so that sibling modules (scoring, config) can be imported.
import sys
from pathlib import Path
# Add the project root to the sys.path to allow for absolute imports
# This allows running `uvicorn api.main:app` from the `services/qa_pipeline` directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scoring.quality_tracker import QualityTracker
from scoring.data_types import NodeState, RollingMetrics
from config import load_config

# --- Globals ---
quality_tracker: Optional[QualityTracker] = None
nats_client: Optional[nats.aio.client.Client] = None
app_settings = load_config()

# --- FastAPI App ---
app = FastAPI(
    title="GCP Quality Assurance Pipeline API",
    description="API for submitting entropy samples and monitoring node quality.",
    version="1.0.0",
)

# --- Pydantic Models ---

class EntropySubmission(BaseModel):
    """Model for submitting a new entropy sample."""
    node_id: str = Field(..., description="The unique identifier for the entropy source node.")
    sample_data: List[float] = Field(..., description="The raw entropy data sample, as a list of floats.")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the submission.")


# --- Dependency Injection ---

async def get_nats_client() -> nats.aio.client.Client:
    """FastAPI dependency to get a connected NATS client."""
    global nats_client
    if nats_client is None or not nats_client.is_connected:
        # This will fail if startup event hasn't run or failed.
        raise HTTPException(status_code=503, detail="NATS client not available.")
    return nats_client

def get_quality_tracker() -> QualityTracker:
    """FastAPI dependency to get the singleton QualityTracker instance."""
    global quality_tracker
    if quality_tracker is None:
        raise RuntimeError("QualityTracker not initialized. Check startup logic.")
    return quality_tracker


# --- API Endpoints ---

@app.post("/v1/submit", status_code=202, summary="Submit Entropy Sample")
async def submit_entropy_sample(
    submission: EntropySubmission,
    nc: nats.aio.client.Client = Depends(get_nats_client)
):
    """
    Accepts an entropy sample and publishes it to the NATS messaging
    system for asynchronous processing by the QA pipeline worker.
    """
    subject = app_settings.nats.get("subject_prefix", "entropy.sample") + f".{submission.node_id}"
    payload = submission.model_dump_json().encode("utf-8")

    try:
        await nc.publish(subject, payload)
        return {"status": "accepted", "node_id": submission.node_id, "subject": subject}
    except Exception as e:
        logging.error(f"Failed to publish to NATS subject {subject}: {e}")
        raise HTTPException(status_code=500, detail="Failed to publish sample to processing queue.")


@app.get("/v1/status/{node_id}", response_model=NodeState, summary="Get Node Status")
async def get_node_status(
    node_id: str,
    tracker: QualityTracker = Depends(get_quality_tracker)
):
    """
    Retrieves the current quality status and history for a specific node.
    """
    node_state = tracker.get_node_state(node_id)
    if node_state is None:
        # Return a default "UNKNOWN" state for nodes not yet seen
        return NodeState(node_id=node_id)
    return node_state

@app.get("/v1/metrics/{node_id}", response_model=Optional[RollingMetrics], summary="Get Node Metrics")
async def get_node_metrics(
    node_id: str,
    tracker: QualityTracker = Depends(get_quality_tracker)
):
    """
    Retrieves the current rolling metrics for a specific node.
    Returns null if no metrics are available.
    """
    return tracker.get_rolling_metrics(node_id)


# --- Startup and Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Initializes the QualityTracker and connects to NATS on startup."""
    global quality_tracker, nats_client
    logging.basicConfig(level=logging.INFO)
    logging.info("QA Pipeline API starting up...")

    # Initialize QualityTracker
    logging.info("Initializing QualityTracker...")
    qt_config = app_settings.get('quality_tracker', {})
    quality_tracker = QualityTracker(qt_config, status_change_callback=None)

    # Connect to NATS
    try:
        nats_url = app_settings.get('nats', {}).get('url', 'nats://localhost:4222')
        nats_client = await nats.connect(nats_url)
        logging.info(f"Successfully connected to NATS at {nats_url}")
    except Exception as e:
        logging.error(f"Fatal: Could not connect to NATS on startup: {e}")
        # Allow app to start, but endpoints will fail.
        nats_client = None

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully closes the NATS connection on shutdown."""
    global nats_client
    if nats_client and nats_client.is_connected:
        logging.info("Draining and closing NATS connection...")
        await nats_client.drain()
    logging.info("QA Pipeline API shut down.")

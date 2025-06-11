"""
Asynchronous worker for the Quality Assurance Pipeline.

This worker subscribes to a NATS subject, receives entropy samples,
runs them through the quality scoring system, and updates the node status.
"""
import asyncio
import logging
import signal

import nats
from pydantic import ValidationError

# Assumes the service is run from the `services/qa_pipeline` directory
# so that sibling modules (scoring, config) can be imported.
import sys
from pathlib import Path
# Add the project root to the sys.path to allow for absolute imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scoring.quality_tracker import QualityTracker
from scoring.data_types import QualityScore
from api.main import EntropySubmission # Re-use the submission model from the API
from config import load_config

# --- Globals ---
stop_event = asyncio.Event()

# --- Main Worker Logic ---

async def run_worker():
    """Connects to NATS, subscribes to subjects, and processes messages."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("QA Pipeline Worker starting up...")

    # Load configuration
    app_settings = load_config()
    qt_config = app_settings.get('quality_tracker', {})
    nats_config = app_settings.get('nats', {})
    nats_url = nats_config.get('url', 'nats://localhost:4222')
    subject = nats_config.get("subject_prefix", "entropy.sample") + ".*>"

    # Initialize QualityTracker
    # In a real-world scenario, the state might be shared with the API (e.g., via Redis)
    # For this example, the worker maintains its own state.
    quality_tracker = QualityTracker(qt_config)
    logging.info("QualityTracker initialized.")

    # Connect to NATS
    try:
        nc = await nats.connect(nats_url)
        logging.info(f"Connected to NATS at {nats_url}")
    except Exception as e:
        logging.fatal(f"Failed to connect to NATS: {e}")
        return

    async def message_handler(msg):
        """Callback to process incoming entropy samples."""
        try:
            # 1. Decode and validate the message payload
            submission = EntropySubmission.model_validate_json(msg.data)
            logging.info(f"Received sample from node '{submission.node_id}' on subject '{msg.subject}'.")

            # 2. Perform quality scoring (simplified for this example)
            # In a real implementation, this would call the statistical test framework.
            score_value = sum(submission.sample_data) % 101
            score = QualityScore(
                node_id=submission.node_id,
                sample_id=f"sample_{submission.timestamp.isoformat()}",
                score=float(score_value),
                timestamp=submission.timestamp,
                test_scores={"placeholder_test": float(score_value)}
            )

            # 3. Record the score with the tracker
            quality_tracker.record_score(score)
            logging.info(f"Recorded score {score.score:.2f} for node '{score.node_id}'.")

        except ValidationError as e:
            logging.error(f"Failed to validate message payload: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in message_handler: {e}")

    # Subscribe to the subject
    sub = await nc.subscribe(subject, cb=message_handler)
    logging.info(f"Subscribed to subject: {subject}")

    # Wait for the stop signal
    await stop_event.wait()

    # Graceful shutdown
    logging.info("Shutting down worker...")
    await sub.unsubscribe()
    await nc.drain()
    logging.info("NATS connection closed. Worker shut down.")

def shutdown_handler(sig, frame):
    """Signal handler to trigger graceful shutdown."""
    logging.info(f"Received shutdown signal: {sig}. Triggering stop event.")
    stop_event.set()


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Run the main worker coroutine
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logging.info("Worker stopped by user.")

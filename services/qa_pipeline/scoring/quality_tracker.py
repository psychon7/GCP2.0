"""
Quality tracker system for monitoring node entropy quality over time.
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np

# Local imports
from .data_types import (
    NodeState,
    NodeStatus,
    QualityScore,
    RollingQualityMetrics,
)

logger = logging.getLogger(__name__)


class QualityTracker:
    """
    Tracks the quality scores of multiple entropy nodes in real-time.
    """

    def __init__(self, config: dict, status_change_callback: Optional[Callable[[str, NodeState], None]] = None):
        """Initialize the QualityTracker."""
        self.config = config
        self.enabled = config.get("enabled", True)
        self.window_size = config.get("window_size", 100)
        self.threshold_score = config.get("threshold_score", 70.0)
        self.warning_threshold = config.get("warning_threshold", 85.0)
        self.flagging_period = config.get("flagging_period", 5)
        self.offline_timeout = config.get("offline_timeout_seconds", 60)

        self.node_scores: Dict[str, deque] = {}
        self.node_states: Dict[str, NodeState] = {}

        self.status_change_callbacks: List[Callable[[str, NodeState], None]] = []
        if status_change_callback:
            self.add_status_change_listener(status_change_callback)

    def add_status_change_listener(
        self, callback: Callable[[str, NodeState], None]
    ):
        """Register a callback to be notified of node status changes."""
        self.status_change_callbacks.append(callback)

    def _notify_status_change(self, node_id: str, node_state: NodeState):
        """Notify all registered listeners about a status change."""
        for callback in self.status_change_callbacks:
            try:
                callback(node_id, node_state)
            except Exception as e:
                logger.error(
                    f"Error in status change callback for node {node_id}: {e}",
                    exc_info=True,
                )

    def _get_or_create_node_state(self, node_id: str) -> NodeState:
        """Get or create the state object for a given node."""
        if node_id not in self.node_states:
            logger.info(f"First time seeing node {node_id}, creating new state record.")
            self.node_states[node_id] = NodeState(node_id=node_id)
            self.node_scores[node_id] = deque(maxlen=self.window_size)
        return self.node_states[node_id]

    def record_score(self, score_record: QualityScore) -> QualityScore:
        """
        Record a new quality score for a node and update its status.
        """
        if not self.enabled:
            return score_record

        node_id = score_record.node_id
        node_state = self._get_or_create_node_state(node_id)

        # Update activity and score history
        self.node_scores[node_id].append(score_record)
        node_state.last_activity = score_record.timestamp
        node_state.last_score = score_record.score

        # Evaluate the node's status based on the new score
        self._evaluate_node_status(node_id, score_record.score)

        return score_record

    def _evaluate_node_status(self, node_id: str, current_score: float):
        """
        Evaluate and update node status based on the latest score.
        """
        node_state = self.node_states[node_id]
        old_status = node_state.current_status

        # Reset consecutive below-threshold counter if score is good
        if current_score >= self.threshold_score:
            node_state.consecutive_below_threshold = 0
        else:
            node_state.consecutive_below_threshold += 1

        # Determine new status
        new_status = old_status
        message = ""

        if node_state.consecutive_below_threshold >= self.flagging_period:
            new_status = NodeStatus.FLAGGED
            message = (
                f"Node flagged: {node_state.consecutive_below_threshold} consecutive "
                f"scores below threshold ({self.threshold_score})."
            )
        elif current_score < self.threshold_score:
            new_status = NodeStatus.WARNING
            message = f"Node quality warning: score {current_score:.2f} is below threshold {self.threshold_score:.2f}."
        elif current_score < self.warning_threshold:
            new_status = NodeStatus.WARNING
            message = f"Node quality degrading: score {current_score:.2f} is below warning threshold {self.warning_threshold:.2f}."
        else:
            new_status = NodeStatus.OK
            message = f"Node quality OK: score {current_score:.2f}."

        # Update status if it has changed
        if new_status != old_status:
            node_state.update_status(new_status, message)
            self._notify_status_change(node_id, node_state)
            logger.info(f"Node {node_id} status changed from {old_status.value} to {new_status.value}: {message}")

    def check_offline_nodes(self):
        """
        Check for nodes that haven't sent data recently and mark them as offline.
        """
        if not self.enabled:
            return

        now = datetime.now()
        offline_threshold_time = now - timedelta(seconds=self.offline_timeout)

        for node_id, node_state in self.node_states.items():
            if (
                node_state.last_activity < offline_threshold_time
                and node_state.current_status != NodeStatus.OFFLINE
            ):
                message = f"Node offline: No activity since {node_state.last_activity.isoformat()}"
                logger.warning(message)
                node_state.update_status(NodeStatus.OFFLINE, message)
                self._notify_status_change(node_id, node_state)

    def get_node_state(self, node_id: str) -> Optional[NodeState]:
        """Get the current state for a node."""
        return self.node_states.get(node_id)

    def get_all_node_states(self) -> Dict[str, NodeState]:
        """Get state for all tracked nodes."""
        return self.node_states.copy()

    def get_rolling_metrics(self, node_id: str) -> Optional[RollingQualityMetrics]:
        """
        Calculate rolling metrics for a node based on historical scores.
        """
        if node_id not in self.node_scores or not self.node_scores[node_id]:
            return None

        scores = self.node_scores[node_id]
        if not scores:
            return None

        score_values = [s.score for s in scores]
        timestamps = [s.timestamp for s in scores]

        avg_score = np.mean(score_values)
        min_score = np.min(score_values)
        max_score = np.max(score_values)
        std_dev = np.std(score_values) if len(score_values) > 1 else 0.0

        test_metrics = {}
        all_tests = set()
        for score in scores:
            all_tests.update(score.test_scores.keys())

        for test_id in all_tests:
            test_values = [
                s.test_scores[test_id] for s in scores if test_id in s.test_scores
            ]
            if test_values:
                test_metrics[test_id] = {
                    "avg": float(np.mean(test_values)),
                    "min": float(np.min(test_values)),
                    "max": float(np.max(test_values)),
                    "std_dev": float(np.std(test_values)) if len(test_values) > 1 else 0.0,
                }

        return RollingQualityMetrics(
            node_id=node_id,
            window_start=min(timestamps),
            window_end=max(timestamps),
            rolling_avg_score=float(avg_score),
            min_score=float(min_score),
            max_score=float(max_score),
            std_dev=float(std_dev),
            num_samples=len(scores),
            test_metrics=test_metrics,
        )

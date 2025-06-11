"""
Unit tests for the quality scoring system.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Adjust path to import from the services directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.qa_pipeline.scoring.quality_tracker import QualityTracker
from services.qa_pipeline.scoring.data_types import NodeStatus, NodeState, QualityScore


class TestQualityTracker(unittest.TestCase):
    """Test cases for the QualityTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "enabled": True,
            "window_size": 10,
            "threshold_score": 70.0,
            "flagging_period": 3,
            "warning_threshold": 85.0,
            "offline_timeout_seconds": 60,
        }
        self.status_change_mock = MagicMock()
        self.tracker = QualityTracker(self.config, status_change_callback=self.status_change_mock)

    def _create_score(self, node_id: str, score: float, sample_id: str = "sample") -> QualityScore:
        """Helper to create a QualityScore object."""
        return QualityScore(
            node_id=node_id,
            sample_id=sample_id,
            score=score,
            timestamp=datetime.now(),
            test_scores={"test1": score}
        )

    def test_record_score_and_initial_state(self):
        """Test recording a quality score and creating initial node state."""
        score_record = self._create_score("node1", 90.0)
        self.tracker.record_score(score_record)

        # Check that score was recorded
        self.assertEqual(len(self.tracker.node_scores["node1"]), 1)
        self.assertEqual(self.tracker.node_scores["node1"][0].score, 90.0)

        # Check that the node state was created
        node_state = self.tracker.get_node_state("node1")
        self.assertIsNotNone(node_state)
        self.assertEqual(node_state.node_id, "node1")
        self.assertEqual(node_state.current_status, NodeStatus.OK)
        # History should contain initial UNKNOWN and the new OK status
        self.assertEqual(len(node_state.status_history), 2)
        self.assertEqual(node_state.status_history[0].status, NodeStatus.OK)
        self.assertEqual(node_state.status_history[1].status, NodeStatus.UNKNOWN)

    def test_status_changes_from_ok_to_warning_to_flagged(self):
        """Test status transitions: OK -> WARNING -> FLAGGED."""
        node_id = "node2"

        # 1. Start with a good score -> OK
        self.tracker.record_score(self._create_score(node_id, 95.0, "s1"))
        node_state = self.tracker.get_node_state(node_id)
        self.assertEqual(node_state.current_status, NodeStatus.OK)
        self.status_change_mock.assert_called_with(node_id, node_state)

        # 2. Record a score below warning threshold -> WARNING
        self.tracker.record_score(self._create_score(node_id, 80.0, "s2"))
        self.assertEqual(node_state.current_status, NodeStatus.WARNING)
        self.assertEqual(node_state.consecutive_below_threshold, 0)  # Still above main threshold
        self.status_change_mock.assert_called_with(node_id, node_state)

        # 3. Record scores below flagging threshold -> WARNING -> FLAGGED
        self.tracker.record_score(self._create_score(node_id, 65.0, "s3"))  # Below 70
        self.assertEqual(node_state.current_status, NodeStatus.WARNING)
        self.assertEqual(node_state.consecutive_below_threshold, 1)

        self.tracker.record_score(self._create_score(node_id, 60.0, "s4"))  # Below 70
        self.assertEqual(node_state.current_status, NodeStatus.WARNING)
        self.assertEqual(node_state.consecutive_below_threshold, 2)

        # This one should trigger the flag
        self.tracker.record_score(self._create_score(node_id, 55.0, "s5"))  # Below 70
        self.assertEqual(node_state.current_status, NodeStatus.FLAGGED)
        self.assertEqual(node_state.consecutive_below_threshold, 3)
        self.status_change_mock.assert_called_with(node_id, node_state)

        # Check history (UNKNOWN -> OK -> WARNING -> FLAGGED)
        # Note: The two WARNING scores don't trigger a state *change*, so only one is recorded
        self.assertEqual(len(node_state.status_history), 4)
        self.assertEqual(node_state.status_history[0].status, NodeStatus.FLAGGED)
        self.assertEqual(node_state.status_history[1].status, NodeStatus.WARNING)
        self.assertEqual(node_state.status_history[2].status, NodeStatus.OK)

    def test_status_recovery_from_flagged_to_ok(self):
        """Test that node status recovers from FLAGGED to OK."""
        node_id = "node3"

        # Put node in FLAGGED state
        for i in range(self.config["flagging_period"]):
            self.tracker.record_score(self._create_score(node_id, 60.0, f"s{i}"))

        node_state = self.tracker.get_node_state(node_id)
        self.assertEqual(node_state.current_status, NodeStatus.FLAGGED)

        # Record a good score
        self.tracker.record_score(self._create_score(node_id, 95.0, "s_good"))

        # Check that status is now OK and counter is reset
        self.assertEqual(node_state.current_status, NodeStatus.OK)
        self.assertEqual(node_state.consecutive_below_threshold, 0)
        self.status_change_mock.assert_called_with(node_id, node_state)

    def test_check_offline_nodes(self):
        """Test detecting offline nodes."""
        node_id = "node4"
        self.tracker.record_score(self._create_score(node_id, 90.0))

        node_state = self.tracker.get_node_state(node_id)
        self.assertEqual(node_state.current_status, NodeStatus.OK)

        # Manually set last activity to be older than the timeout
        timeout = self.config["offline_timeout_seconds"]
        node_state.last_activity = datetime.now() - timedelta(seconds=timeout + 1)

        self.tracker.check_offline_nodes()

        self.assertEqual(node_state.current_status, NodeStatus.OFFLINE)
        self.status_change_mock.assert_called_with(node_id, node_state)
        self.assertIn("offline", node_state.status_history[0].message.lower())

    def test_get_rolling_metrics(self):
        """Test calculating rolling metrics."""
        node_id = "node5"
        scores = [60.0, 70.0, 80.0, 90.0, 85.0]

        for i, score in enumerate(scores):
            self.tracker.record_score(self._create_score(node_id, score, f"s{i}"))

        metrics = self.tracker.get_rolling_metrics(node_id)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.node_id, node_id)
        self.assertEqual(metrics.num_samples, len(scores))
        self.assertAlmostEqual(metrics.rolling_avg_score, sum(scores) / len(scores), places=1)
        self.assertEqual(metrics.min_score, min(scores))
        self.assertEqual(metrics.max_score, max(scores))

        # Check test metrics
        self.assertIn("test1", metrics.test_metrics)
        self.assertAlmostEqual(metrics.test_metrics["test1"]["avg"], sum(scores) / len(scores), places=1)


if __name__ == "__main__":
    unittest.main()

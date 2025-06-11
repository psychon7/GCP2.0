"""
Data types for the Quality Scoring System.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, model_validator


class NodeStatus(str, Enum):
    """Status values for entropy nodes."""
    
    OK = "ok"
    WARNING = "warning"
    FLAGGED = "flagged"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class QualityScore(BaseModel):
    """Single quality score measurement for a node."""
    
    node_id: str = Field(..., description="ID of the entropy source/node")
    sample_id: str = Field(..., description="ID of the sample that was tested")
    score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score from 0-100")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this score was calculated")
    test_scores: Dict[str, float] = Field(default_factory=dict, description="Individual scores from each test")


class RollingQualityMetrics(BaseModel):
    """Rolling quality metrics for a node over a time window."""
    
    node_id: str = Field(..., description="ID of the entropy source/node")
    window_start: datetime = Field(..., description="Start of the time window")
    window_end: datetime = Field(..., description="End of the time window")
    rolling_avg_score: float = Field(..., ge=0.0, le=100.0, description="Average quality score over the window")
    min_score: float = Field(..., ge=0.0, le=100.0, description="Minimum score in the window")
    max_score: float = Field(..., ge=0.0, le=100.0, description="Maximum score in the window")
    std_dev: float = Field(..., ge=0.0, description="Standard deviation of scores in the window")
    num_samples: int = Field(..., ge=0, description="Number of samples in this window")
    test_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Test-specific metrics (avg, min, max, std_dev)")


class StatusHistoryItem(BaseModel):
    """Represents a single event in a node's status history."""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    status: NodeStatus
    message: Optional[str] = None


class NodeState(BaseModel):
    """
    Represents the current state and status history of a single entropy node.
    This object is managed by the QualityTracker.
    """
    
    node_id: str
    
    # Current status
    current_status: NodeStatus = NodeStatus.UNKNOWN
    status_since: datetime = Field(default_factory=datetime.now)
    
    # Details for flagging logic
    consecutive_below_threshold: int = 0
    last_activity: datetime = Field(default_factory=datetime.now)
    last_score: Optional[float] = None
    
    # History
    status_history: List[StatusHistoryItem] = Field(default_factory=list)
    
    # Optional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='before')
    @classmethod
    def ensure_initial_history(cls, values):
        """Ensure that the node's history starts with its initial status."""
        history = values.get('status_history', [])
        if not history:
            initial_status = values.get('current_status', NodeStatus.UNKNOWN)
            initial_timestamp = values.get('status_since', datetime.now())
            history.append(StatusHistoryItem(status=initial_status, timestamp=initial_timestamp, message="Initial state tracking started."))
            values['status_history'] = history
        return values

    def update_status(self, new_status: NodeStatus, message: Optional[str] = None):
        """
        Updates the node's status and records the change in its history.
        
        Args:
            new_status: The new status to set.
            message: An optional message describing the reason for the change.
        """
        if new_status != self.current_status:
            now = datetime.now()
            self.status_history.insert(0, StatusHistoryItem(status=new_status, timestamp=now, message=message))
            self.current_status = new_status
            self.status_since = now


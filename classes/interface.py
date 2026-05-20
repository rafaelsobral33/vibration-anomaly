from datetime import datetime
from typing import Sequence, List, Optional

from pydantic import BaseModel, Field


# --- Domain / data transfer types ---


class DataPoint(BaseModel):
    timestamp: datetime = Field(
        ..., description="Unix timestamp of the time the data point was collected"
    )
    uptime: bool = Field(..., description="Whether the data point is during uptime")
    vel_x: float = Field(
        ..., description="Vibration velocity component along the X axis"
    )
    vel_y: float = Field(
        ..., description="Vibration velocity component along the Y axis"
    )
    vel_z: float = Field(
        ..., description="Vibration velocity component along the Z axis"
    )

    acc_x: float = Field(
        ..., description="Vibration acceleration component along the X axis"
    )
    acc_y: float = Field(
        ..., description="Vibration acceleration component along the Y axis"
    )
    acc_z: float = Field(
        ..., description="Vibration acceleration component along the Z axis"
    )


class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity",
    )

    @property
    def length(self) -> int:
        return len(self.data)

    @property
    def last_timestamp(self) -> datetime:
        if not self.data:
            raise ValueError("TimeSeries has no data")
        return self.data[-1].timestamp

    @property
    def first_timestamp(self) -> datetime:
        if not self.data:
            raise ValueError("TimeSeries has no data")
        return self.data[0].timestamp


# --- Model / pipeline types (shared across modules) ---


class Weights(BaseModel):
    fitted: bool = False
    mean_vector: list[float] | None = None
    inv_covariance: list[list[float]] | None = None
    dynamic_threshold: float | None = None

class ModelParams(BaseModel):
    window_anomaly_ratio: float = 0.5
    mahalanobis_mean_threshold: float = 10
    fault_threshold: int = 20

class AlertEngineParams(BaseModel):
    hours_to_clear: int = 12
    increase_to_realert: float = 3.0
    consecutive_anomalies_to_alert: int = 2

class PipelineParams(BaseModel):
    model_window_size_hours: float = 4.0
    window_overlap_hours: float = 0.0


class PredictOutput(BaseModel):
    anomaly_status: bool
    anomaly_score: float
    anomaly_responsibles: list[str]
    timestamp: datetime


class AlertDecision(BaseModel):
    alert: bool
    timestamp: datetime
    message: str


class TrueIncident(BaseModel):
    start: datetime
    end: datetime

from .interface import AlertDecision, PredictOutput, AlertEngineParams
from datetime import timedelta, datetime
from pathlib import Path
import yaml

DEFAULT_ALERT_ENGINE_PARAMS = Path("hyperparameters/alertengine_hyperparams.yaml")

def load_alertengine_params(path: Path = DEFAULT_ALERT_ENGINE_PARAMS) -> AlertEngineParams:
    """Load pipeline hyperparameters from YAML. Falls back to PipelineParams defaults if file missing."""
    if not path.exists():
        return AlertEngineParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return AlertEngineParams(**data)

class AlertEngine:
    def __init__(self):
        self.locked = False
        self.last_alert_timestamp = None
        self.anomaly_score = None
        self.consecutive_alerts = 0
        self.params = load_alertengine_params(DEFAULT_ALERT_ENGINE_PARAMS)

    def current_score(self, prediction: PredictOutput) -> float:
        return prediction.anomaly_score
    
    def current_time(self, prediction: PredictOutput) -> datetime:
        return prediction.timestamp

    def _get_responsibles(self, prediction: PredictOutput) -> list[str]:
        return prediction.anomaly_responsibles

    def _has_alert(self, prediction: PredictOutput) -> bool:
        return prediction.anomaly_status

    def predict(self, prediction: PredictOutput) -> AlertDecision:
        # Extracts if there is an alert from the prediction
        has_anomaly = self._has_alert(prediction)
        # Extracts the timestamp of the prediction
        current_time = self.current_time(prediction)
        # Extracts the current score from the prediction
        current_score = self.current_score(prediction)
        # Extracts the anomaly responsibles, if any
        current_responsibles = self._get_responsibles(prediction)
        
        # Checks if the alert is locked (alert already happened)
        if self.locked:
            # Confirms if the anomaly is still happening or stopped
            if has_anomaly:
                # Saves the last timestamp in which an anomaly was identified
                self.last_alert_timestamp = current_time
                
                # Checks if the anomaly has worsened significantly so that a re-alert is triggered
                if self.anomaly_score is not None and current_score >= self.params.increase_to_realert*self.anomaly_score:
                    # Saves the new threshold for re-alert so it just re-alerts if it worsenes even further
                    self.anomaly_score = current_score 
                    # Concatenates the responsibles for the alert message
                    causes_str = ", ".join(current_responsibles)

                    # Returns the re-alert with the message explaining what caused it
                    return AlertDecision(
                        alert=True,  
                        timestamp=current_time,
                        message=f"Anomaly worsened significantly. Score increased to {current_score:.2f}. Causes: {causes_str}.",
                    )

                # If the anomaly hasn't worsened significantly, doesn't emit the alert
                return AlertDecision(
                    alert=False,  
                    timestamp=current_time,
                    message="System already entered abnormal state earlier. Updating persistent anomaly timestamp.",
                )
            else:
                # Compute the time since the last anomaly so that it can be considered finished
                time_since_last_alert = current_time - self.last_alert_timestamp
                
                # Checks if the time elapsed is enough to consider the anomaly finished
                if time_since_last_alert >= timedelta(hours=self.params.hours_to_clear):
                    # Frees all variables
                    self.locked = False
                    self.last_alert_timestamp = None
                    self.anomaly_score = None 
                    self.consecutive_alerts = 0 

                    # Returns a message with the information that the anomaly is considered finished
                    return AlertDecision(
                        alert=False,
                        timestamp=current_time,
                        message=f"No persistent abnormal vibration. Lock released after {self.params.hours_to_clear} hours of normal state.",
                    )
                else:
                    # Keeps waiting for the minimum time without anomaly to allow new alerts
                    return AlertDecision(
                        alert=False,
                        timestamp=current_time,
                        message=f"System is locked. Waiting for {self.params.hours_to_clear} hours of continuous normal state to release.",
                    )

        # Checks if the machine has entered an anomaly state
        if has_anomaly:
            # Saves the total of consecutive anomaly identified without acting upon it
            self.consecutive_alerts += 1
            causes_str = ", ".join(current_responsibles)

            # If the total of consecutive anomalies identified surpasses a threshold, then alert it to avoid false positives
            if self.consecutive_alerts >= self.params.consecutive_anomalies_to_alert:
                # Saves the variables relative to the alert and locks the alert engine
                self.locked = True
                self.last_alert_timestamp = current_time
                self.anomaly_score = current_score            
                
                # Returns a message with the information about the causes of the anomaly
                return AlertDecision(
                    alert=True,
                    timestamp=current_time,
                    message=f"Abnormal vibration confirmed. Initial score of {self.anomaly_score:.2f}. Causes: {causes_str}.",
                )
            else:
                # Returns a message with the information that another consecutive anomaly was found, but it will not be acted upon yet
                return AlertDecision(
                    alert=False,
                    timestamp=current_time,
                    message=f"{self.consecutive_alerts} consecutive anomalies detected (Causes: {causes_str}). Waiting for confirmation to alert.",
                )

        # If no anomaly is identified, refreshes the consecutive alert counter
        if self.consecutive_alerts > 0:
            self.consecutive_alerts = 0
            # Returns a message that explains that the previous anomaly didn't generate any alert.
            return AlertDecision(
                alert=False,
                timestamp=current_time,
                message="Previous anomaly not confirmed. Treated as a transient spike/false positive.",
            )

        self.consecutive_alerts = 0
        
        return AlertDecision(
            alert=False,
            timestamp=current_time,
            message="No persistent abnormal vibration.",
        )
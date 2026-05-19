from .interface import AlertDecision, PredictOutput
from datetime import timedelta

class AlertEngine:
    def __init__(self):
        self.locked = False
        self.last_alert_timestamp = None
        self.hours_to_reset = 12

    def _has_alert(self, prediction: PredictOutput) -> bool:
        return prediction.anomaly_status

    def predict(self, prediction: PredictOutput) -> AlertDecision:
        current_time = prediction.timestamp
        has_anomaly = self._has_alert(prediction)
        
        if self.locked:
            if has_anomaly:
                self.last_alert_timestamp = current_time
                return AlertDecision(
                    alert=False,  
                    timestamp=current_time,
                    message="System already entered abnormal state earlier. Updating persistent anomaly timestamp.",
                )
            else:
                time_since_last_alert = current_time - self.last_alert_timestamp
                
                if time_since_last_alert >= timedelta(self.hours_to_reset):
                    self.locked = False
                    self.last_alert_timestamp = None

                    return AlertDecision(
                        alert=False,
                        timestamp=current_time,
                        message="No persistent abnormal vibration. Lock released after 12 hours of normal state.",
                    )
                else:
                    return AlertDecision(
                        alert=False,
                        timestamp=current_time,
                        message="System is locked. Waiting for 12 hours of continuous normal state to release.",
                    )

        if has_anomaly:
            print("lock")
            self.last_alert_timestamp = current_time
            self.locked = True
            return AlertDecision(
                alert=True,
                timestamp=current_time,
                message="Abnormal vibration detected.",
            )

        # Sem anomalia e sem lock
        return AlertDecision(
            alert=False,
            timestamp=current_time,
            message="No persistent abnormal vibration.",
        )
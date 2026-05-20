from .interface import AlertDecision, PredictOutput
from datetime import timedelta

class AlertEngine:
    def __init__(self):
        self.locked = False
        self.last_alert_timestamp = None
        self.anomaly_score = None
        self.hours_to_reset = 12
        self.consecutive_alerts = 0 

    def _has_alert(self, prediction: PredictOutput) -> bool:
        return prediction.anomaly_status

    def predict(self, prediction: PredictOutput) -> AlertDecision:
        current_time = prediction.timestamp
        has_anomaly = self._has_alert(prediction)
        
        if self.locked:
            if has_anomaly:
                current_score = prediction.anomaly_score
                self.last_alert_timestamp = current_time
                
                if self.anomaly_score is not None and current_score >= 3 * self.anomaly_score:
                    self.anomaly_score = current_score 
                    causes_str = ", ".join(prediction.anomaly_responsibles)

                    return AlertDecision(
                        alert=True,  
                        timestamp=current_time,
                        message=f"Anomaly worsened significantly. Score increased to {current_score:.2f}. Causes: {causes_str}.",
                    )

                return AlertDecision(
                    alert=False,  
                    timestamp=current_time,
                    message="System already entered abnormal state earlier. Updating persistent anomaly timestamp.",
                )
            else:
                time_since_last_alert = current_time - self.last_alert_timestamp
                
                if time_since_last_alert >= timedelta(hours=self.hours_to_reset):
                    self.locked = False
                    self.last_alert_timestamp = None
                    self.anomaly_score = None 
                    self.consecutive_alerts = 0 

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
            self.consecutive_alerts += 1
            causes_str = ", ".join(prediction.anomaly_responsibles)

            if self.consecutive_alerts >= 2:
                self.last_alert_timestamp = current_time
                self.locked = True
                self.anomaly_score = prediction.anomaly_score            
                
                return AlertDecision(
                    alert=True,
                    timestamp=current_time,
                    message=f"Abnormal vibration confirmed. Initial score of {self.anomaly_score:.2f}. Causes: {causes_str}.",
                )
            else:
                return AlertDecision(
                    alert=False,
                    timestamp=current_time,
                    # Útil incluir aqui também para facilitar o debug caso o alarme não se confirme depois
                    message=f"First anomaly detected (Causes: {causes_str}). Waiting for consecutive confirmation to alert.",
                )

        if self.consecutive_alerts > 0:
            self.consecutive_alerts = 0
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
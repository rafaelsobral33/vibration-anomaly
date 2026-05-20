from pathlib import Path

import numpy as np
import yaml

from .interface import ModelParams, PredictOutput, TimeSeries, Weights

DEFAULT_PARAMS_PATH = Path("hyperparameters/model_hyperparams.yaml")

def load_model_params(path: Path = DEFAULT_PARAMS_PATH) -> ModelParams:
    """Load model hyperparameters from YAML. Falls back to ModelParams defaults if file missing."""
    if not path.exists():
        return ModelParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ModelParams(**data)

class AnomalyModel:
    def __init__(self, params_path: Path | None = None):
        self.weights = Weights()
        self.params = load_model_params(params_path or DEFAULT_PARAMS_PATH)

    def _featuring(self, samples: TimeSeries) -> np.ndarray:
        if not samples.data:
            return np.array([], dtype=float)

        # Gets only data from when the machine is running
        valid_samples = [p for p in samples.data if p.uptime]

        # If all data is from when machine isn't running, returns an empty array
        if not valid_samples:
            return np.array([], dtype=float)

        # Makes sure that the array is ordered according to the timestamp of the samples
        ordered = sorted(valid_samples, key=lambda p: p.timestamp)

        # Extracts the features as an array
        features = np.array([
            (p.vel_x, p.vel_y, p.vel_z, p.acc_x, p.acc_y, p.acc_z) 
            for p in ordered
        ], dtype=float)

        # Returns the array of features
        return features

    def fit(self, fitting_samples: TimeSeries) -> None:
        # Extracts the array of features
        X = self._featuring(fitting_samples)
        
        # Computes the mean and inverse covariance matrix of the fit data
        mean_vec = np.mean(X, axis=0)
        inv_cov = np.linalg.pinv(np.cov(X, rowvar=False))

        # Saves the values as weights for the predict
        self.weights = Weights(
            fitted=True,
            mean_vector=mean_vec.tolist(),
            inv_covariance=inv_cov.tolist()
        )

    def predict(self, samples: TimeSeries) -> PredictOutput:
        if not self.weights.fitted:
            raise RuntimeError("Model not fitted")
        if not samples.data:
            raise ValueError("Cannot predict on empty TimeSeries")

        # Extracts the features from the samples
        X = self._featuring(samples)

        # If half or more of the sample is from when the machine wasn't working, doesn't predict anything
        if len(X) <= len(samples.data)/2:
            return PredictOutput(
                anomaly_status=False,
                anomaly_score=0.0,
                anomaly_responsibles=[],
                timestamp=samples.data[-1].timestamp,
            )

        # Loads the matrices precalculated in the fit phase
        mean_vector = np.array(self.weights.mean_vector)
        inv_covariance = np.array(self.weights.inv_covariance)

        # Computes the Mahalanobis Distance (MD)
        delta = X - mean_vector
        distances = np.sqrt(np.sum(np.dot(delta, inv_covariance)*delta, axis=1))
        
        # Saves the average MD of the sample to indicate the gravity of the fault
        window_score = float(np.mean(distances))
        
        # Triggers as anomalous behaviour if more than half of the valid samples are considered anomalous
        is_anomalous = np.sum(distances > self.params.mahalanobis_mean_threshold)/len(X)>=self.params.window_anomaly_ratio
        
        # If considered anomalous, extract which features are the main responsible for the faulty behaviour
        responsibles = []
        if is_anomalous:
            # Computes the Mahalanobis Distance squared for each sampleh
            contributions_squared = delta*np.dot(delta, inv_covariance)

            # Computes the mean contribution of each variable to the MD
            mean_contributions = np.mean(contributions_squared, axis=0)

            # Computes the sum of contributions
            total_d2 = np.sum(mean_contributions) 
            
            # Computes the importance of each feature according to the total distance
            importance_percentages = (mean_contributions/total_d2)*100 if total_d2>0 else np.zeros(6)
            
            # Saves the name of the features which contribute the most for the trigger
            feature_names = ["vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"]
            for i, name in enumerate(feature_names):
                if importance_percentages[i] >= self.params.fault_threshold:
                    responsibles.append(name)

        # Returns the anomalou status with the score for the window and the responsibles for the anomaly, if any
        return PredictOutput(
            anomaly_status=is_anomalous,
            anomaly_score=window_score,
            anomaly_responsibles=responsibles,
            timestamp=samples.data[-1].timestamp,
        )
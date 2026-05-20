from pathlib import Path

import numpy as np
import yaml

from .interface import ModelParams, PipelineParams, PredictOutput, TimeSeries, Weights

DEFAULT_PARAMS_PATH = Path("hyperparameters/model_hyperparams.yaml")
DEFAULT_PIPELINE_PARAMS_PATH = Path("hyperparameters/pipeline_hyperparams.yaml")


def load_pipeline_params(path: Path = DEFAULT_PIPELINE_PARAMS_PATH) -> PipelineParams:
    """Load pipeline hyperparameters from YAML. Falls back to PipelineParams defaults if file missing."""
    if not path.exists():
        return PipelineParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return PipelineParams(**data)


def load_model_params(path: Path = DEFAULT_PARAMS_PATH) -> ModelParams:
    """Load model hyperparameters from YAML. Falls back to ModelParams defaults if file missing."""
    if not path.exists():
        return ModelParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ModelParams(**data)

class AnomalyModel:
    def __init__(self,
        pipeline_params: PipelineParams | None = None,
        params_path: Path | None = None):
        
        self.weights = Weights()
        self.params = load_model_params(params_path or DEFAULT_PARAMS_PATH)
        self.pipeline_params = load_pipeline_params(pipeline_params or DEFAULT_PIPELINE_PARAMS_PATH)

    def _featuring(self, samples: TimeSeries) -> np.ndarray:
        if not samples.data:
            return np.empty((0, 6), dtype=float)

        valid_samples = [p for p in samples.data if p.uptime]
        if not valid_samples:
            return np.empty((0, 6), dtype=float)

        ordered = sorted(valid_samples, key=lambda p: p.timestamp)

        features = np.array([
            (p.vel_x, p.vel_y, p.vel_z, p.acc_x, p.acc_y, p.acc_z) 
            for p in ordered
        ], dtype=float)

        return features

    def fit(self, fitting_samples: TimeSeries) -> None:
        X = self._featuring(fitting_samples)
        
        if len(X) < 2:
            raise ValueError("Dados operacionais insuficientes no arquivo de fit para calibração.")

        mean_vec = np.mean(X, axis=0)
        inv_cov = np.linalg.pinv(np.cov(X, rowvar=False))
        print("Médias:", mean_vec)

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

        X = self._featuring(samples)
        
        if len(X) <= len(samples.data)/2:
            return PredictOutput(
                anomaly_status=False,
                timestamp=samples.data[-1].timestamp,
                anomaly_score=0.0
            )

        mean_vector = np.array(self.weights.mean_vector)
        inv_covariance = np.array(self.weights.inv_covariance)

        delta = X - mean_vector
        distances = np.sqrt(np.sum(np.dot(delta, inv_covariance) * delta, axis=1))
        #print("distances", np.shape(distances))
        window_score = float(np.mean(distances))
        
        #is_anomalous = window_score > self.params.mahalanobis_mean_threshold
        is_anomalous = np.sum(distances > self.params.mahalanobis_mean_threshold)/len(X)>=0.5
        
        
        if is_anomalous:
            responsibles = []
            rotated_delta = np.dot(delta, inv_covariance)

            contributions_sq = delta * rotated_delta
            mean_contributions = np.mean(contributions_sq, axis=0)
            total_d2 = np.sum(mean_contributions) 
            
            importance_percentages = (mean_contributions / total_d2) * 100 if total_d2 > 0 else np.zeros(6)
            
            feature_names = ["vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"]
            fault_threshold = 25.0 # Limiar de corte
            
            for i, name in enumerate(feature_names):
                if importance_percentages[i] >= fault_threshold:
                    responsibles.append(name)

        #print(samples.data[-1].timestamp)
        #print("distances_mean", np.sqrt(np.abs(np.sum(np.dot(delta, inv_covariance) * delta, axis=0))))   

        #if is_anomalous:
         #   print("sample_size:", len(X))
         #   print("score:", window_score)
         #   print("timestamp:", samples.data[-1].timestamp)

        return PredictOutput(
            anomaly_status=is_anomalous,
            anomaly_score=window_score,
            anomaly_responsibles=responsibles,
            timestamp=samples.data[-1].timestamp,
        )
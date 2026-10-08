"""MLflow는 Isolation Forest 기록과 버전만 담당한다. 학습 실행과 서빙은 이 프로세스가 한다.

단일 모델 버전 조회·승격은 모델 목록으로 바뀌어 뺐다.
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.ensemble import IsolationForest

from .config import get_settings
from .featureset import FEATURE_KEYS, vector

log = logging.getLogger("analytics.registry")


class ModelError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    return MlflowClient(tracking_uri=get_settings().mlflow_tracking_uri)


def train_isolation(rows: list[dict], mlflow_name: str, params: dict) -> dict:
    """특징값으로 Isolation Forest를 학습하고 해당 등록 이름의 production으로 둔다."""
    contamination = float(params["contamination"])
    if not 0.001 <= contamination <= 0.5:
        raise ModelError("contamination은 0.001~0.5")
    matrix = [v for row in rows if (v := vector(row)) is not None]
    if len(matrix) < 30:
        raise ModelError(f"학습 샘플이 {len(matrix)}개입니다. 정상 파형을 30개 이상 모은 뒤 다시 학습하세요.")
    x = np.asarray(matrix, dtype=np.float64)
    model = IsolationForest(
        n_estimators=int(params["nEstimators"]),
        contamination=contamination,
        random_state=int(params["randomState"]),
        n_jobs=1,
    )
    model.fit(x)
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment("vibration")
    try:
        with mlflow.start_run(run_name=mlflow_name) as run:
            mlflow.log_params(
                {
                    "contamination": contamination,
                    "n_estimators": int(params["nEstimators"]),
                    "random_state": int(params["randomState"]),
                    "rows": len(matrix),
                    "features": ",".join(FEATURE_KEYS),
                }
            )
            info = mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=mlflow_name)
            run_id = run.info.run_id
    except Exception as e:
        raise ModelError(f"MLflow 기록 실패: {e}", 503) from e
    version = str(info.registered_model_version)
    client = _client()
    client.set_registered_model_alias(mlflow_name, "production", version)
    log.info("모델 %s v%s 학습 (run %s, %d rows)", mlflow_name, version, run_id, len(matrix))
    return {"version": version, "runId": run_id, "rows": len(matrix)}


def load_named(mlflow_name: str):
    """등록 이름의 production 가중치를 불러 실시간 판정에 쓴다."""
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    client = _client()
    current = client.get_model_version_by_alias(mlflow_name, "production")
    model = mlflow.sklearn.load_model(f"models:/{mlflow_name}@production")
    return current.version, model



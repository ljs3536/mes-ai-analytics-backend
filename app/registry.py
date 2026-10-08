"""MLflow는 실험 기록과 모델 버전만 담당한다. 학습 실행과 서빙은 이 프로세스가 한다."""

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


def list_versions() -> dict:
    s = get_settings()
    client = _client()
    try:
        versions = client.search_model_versions(f"name='{s.model_name}'")
    except Exception as e:
        raise ModelError(f"MLflow에 연결할 수 없습니다: {e}", 503) from e
    production = None
    try:
        production = client.get_model_version_by_alias(s.model_name, "production").version
    except Exception:
        production = None
    return {
        "name": s.model_name,
        "production": production,
        "versions": [
            {
                "version": v.version,
                "runId": v.run_id,
                "status": v.status,
                "createdAt": v.creation_timestamp,
            }
            for v in sorted(versions, key=lambda v: int(v.version), reverse=True)
        ],
    }


def train(rows: list[dict], contamination: float) -> dict:
    if not 0.001 <= contamination <= 0.5:
        raise ModelError("contamination은 0.001~0.5")
    matrix = [v for row in rows if (v := vector(row)) is not None]
    if len(matrix) < 30:
        raise ModelError(f"학습 샘플이 {len(matrix)}개입니다. 정상 파형을 30개 이상 모은 뒤 다시 학습하세요.")
    x = np.asarray(matrix, dtype=np.float64)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=7,
        n_jobs=1,
    )
    model.fit(x)
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment("vibration")
    try:
        with mlflow.start_run(run_name="isolation-forest") as run:
            mlflow.log_params(
                {
                    "contamination": contamination,
                    "rows": len(matrix),
                    "features": ",".join(FEATURE_KEYS),
                }
            )
            info = mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=s.model_name,
            )
            run_id = run.info.run_id
    except Exception as e:
        raise ModelError(f"MLflow 기록 실패: {e}", 503) from e
    version = str(info.registered_model_version)
    client = _client()
    client.set_registered_model_alias(s.model_name, "production", version)
    log.info("모델 v%s 학습, production 지정 (run %s, %d rows)", version, run_id, len(matrix))
    return {"version": version, "runId": run_id, "rows": len(matrix), "production": version}


def promote(version: str) -> dict:
    s = get_settings()
    client = _client()
    try:
        client.get_model_version(s.model_name, version)
        client.set_registered_model_alias(s.model_name, "production", version)
    except Exception as e:
        raise ModelError(f"버전을 지정할 수 없습니다: {e}", 404) from e
    return {"name": s.model_name, "production": version}


def load_production():
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    client = _client()
    current = client.get_model_version_by_alias(s.model_name, "production")
    model = mlflow.sklearn.load_model(f"models:/{s.model_name}@production")
    return current.version, model

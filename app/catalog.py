"""만든 모델 목록. 종류, 파라미터, 실시간 적용 여부를 저장한다."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings

KINDS = [
    {
        "id": "isolation_forest",
        "label": "Isolation Forest",
        "summary": "특징값(rms, peak, crest, kurtosis, 1x, 2x)으로 정상 분포 밖을 찾는다. 지금 운영에 쓰이던 모델이다.",
        "params": [
            {"key": "nEstimators", "label": "트리 개수", "type": "int", "min": 50, "max": 500, "step": 1, "default": 200},
            {"key": "contamination", "label": "contamination", "type": "float", "min": 0.001, "max": 0.5, "step": 0.01, "default": 0.05},
            {"key": "randomState", "label": "random state", "type": "int", "min": 0, "max": 9999, "step": 1, "default": 7},
        ],
    },
    {
        "id": "ssad_protonet",
        "label": "SSAD ProtoNet",
        "summary": "CNN으로 파형 특징을 뽑고, few-shot으로 정상 파형 몇 개의 기준점과 거리를 비교한다. 학습과 판정은 worker가 한다.",
        "params": [
            {"key": "nFilters", "label": "필터 수", "type": "int", "min": 4, "max": 64, "step": 1, "default": 20},
            {"key": "nSupport", "label": "support", "type": "int", "min": 1, "max": 20, "step": 1, "default": 5},
            {"key": "nQuery", "label": "query", "type": "int", "min": 1, "max": 20, "step": 1, "default": 8},
            {"key": "epochs", "label": "epochs", "type": "int", "min": 1, "max": 50, "step": 1, "default": 10},
            {"key": "learningRate", "label": "learning rate", "type": "float", "min": 1e-6, "max": 1e-2, "step": 0.00001, "default": 5e-5},
            {"key": "augmentN", "label": "증강 연산 수", "type": "int", "min": 1, "max": 7, "step": 1, "default": 7},
            {"key": "augmentM", "label": "증강 세기", "type": "int", "min": 1, "max": 10, "step": 1, "default": 6},
            {"key": "threshold", "label": "거리 임계값 (0이면 학습 때 계산)", "type": "float", "min": 0, "max": 100, "step": 0.01, "default": 0},
        ],
    },
    {
        "id": "hoko",
        "label": "HOKO",
        "summary": "센서 파형을 임의의 회전수(torch.ones 기반)로 HOKO 포락선에 올린 뒤, CNN few-shot이 기준점과 threshold를 만든다.",
        "params": [
            {"key": "nFilters", "label": "필터 수", "type": "int", "min": 4, "max": 64, "step": 1, "default": 20},
            {"key": "nSupport", "label": "support", "type": "int", "min": 1, "max": 20, "step": 1, "default": 5},
            {"key": "nQuery", "label": "query", "type": "int", "min": 1, "max": 20, "step": 1, "default": 8},
            {"key": "epochs", "label": "epochs", "type": "int", "min": 1, "max": 50, "step": 1, "default": 10},
            {"key": "learningRate", "label": "learning rate", "type": "float", "min": 1e-6, "max": 1e-2, "step": 0.00001, "default": 5e-5},
            {"key": "augmentN", "label": "증강 연산 수", "type": "int", "min": 1, "max": 7, "step": 1, "default": 7},
            {"key": "augmentM", "label": "증강 세기", "type": "int", "min": 1, "max": 10, "step": 1, "default": 6},
            {"key": "threshold", "label": "거리 임계값 (0이면 학습 때 계산)", "type": "float", "min": 0, "max": 100, "step": 0.01, "default": 0},
        ],
    },
]


def _connect() -> sqlite3.Connection:
    path = Path(get_settings().model_dir)
    path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path / "models.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_catalog() -> None:
    """모델 목록 테이블을 만들고, 비어 있으면 기존 Isolation Forest 한 줄을 넣는다."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                params TEXT NOT NULL,
                realtime INTEGER NOT NULL,
                machine TEXT,
                mlflow_name TEXT,
                version TEXT,
                status TEXT NOT NULL,
                message TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        if count == 0:
            kind = _kind("isolation_forest")
            now = _now()
            conn.execute(
                """
                INSERT INTO models (name, kind, params, realtime, machine, mlflow_name, version, status, message, updated_at)
                VALUES (?, ?, ?, 1, '', ?, NULL, 'saved', '기존 운영 모델. 학습본이 있으면 실시간 판정에 쓴다.', ?)
                """,
                ("진동 Isolation Forest", "isolation_forest", json.dumps(_defaults(kind)), get_settings().model_name, now),
            )


def kinds() -> list[dict]:
    """화면에서 고를 모델 종류와 파라미터 정의를 돌려준다."""
    return KINDS


def list_models() -> list[dict]:
    """저장된 모델 목록을 돌려준다."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY id").fetchall()
    return [_out(row) for row in rows]


def get_model(model_id: int) -> dict:
    """모델 한 건의 종류, 파라미터, 학습 상태를 돌려준다."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if row is None:
        raise KeyError(model_id)
    return _out(row)


def realtime_models() -> list[dict]:
    """실시간 적용이 켜져 있고 학습 중이 아닌 모델만 골라 판정에 쓴다."""
    return [row for row in list_models() if row["realtime"] and row["status"] != "training"]


def create_model(name: str, kind: str, params: dict, realtime: bool, machine: str | None) -> dict:
    """종류와 파라미터, 실시간 적용 여부를 검증해 모델 한 건을 등록한다."""
    spec = _kind(kind)
    clean = _check_params(spec, params)
    now = _now()
    with _connect() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO models (name, kind, params, realtime, machine, mlflow_name, version, status, message, updated_at)
                VALUES (?, ?, ?, ?, ?, NULL, NULL, 'saved', '', ?)
                """,
                (name.strip(), kind, json.dumps(clean), int(realtime), machine or "", now),
            )
        except sqlite3.IntegrityError as e:
            raise ValueError("같은 이름의 모델이 있습니다.") from e
        model_id = int(cur.lastrowid)
        if kind == "isolation_forest":
            conn.execute("UPDATE models SET mlflow_name = ? WHERE id = ?", (f"mes-model-{model_id}", model_id))
    return get_model(model_id)


def update_model(model_id: int, *, name: str, params: dict, realtime: bool) -> dict:
    """이름, 파라미터, 실시간 적용 여부를 바꾼다. 종류는 유지한다."""
    current = get_model(model_id)
    spec = _kind(current["kind"])
    clean = _check_params(spec, params)
    now = _now()
    with _connect() as conn:
        try:
            conn.execute(
                """
                UPDATE models
                SET name = ?, params = ?, realtime = ?, updated_at = ?
                WHERE id = ?
                """,
                (name.strip(), json.dumps(clean), int(realtime), now, model_id),
            )
        except sqlite3.IntegrityError as e:
            raise ValueError("같은 이름의 모델이 있습니다.") from e
    return get_model(model_id)


def mark(model_id: int, *, status: str, message: str = "", version: str | None = None, params: dict | None = None) -> None:
    """학습 진행, 실패, 완료와 계산된 threshold를 목록에 기록한다."""
    fields = ["status = ?", "message = ?", "updated_at = ?"]
    values: list = [status, message, _now()]
    if version is not None:
        fields.append("version = ?")
        values.append(version)
    if params is not None:
        fields.append("params = ?")
        values.append(json.dumps(params))
    values.append(model_id)
    with _connect() as conn:
        conn.execute(f"UPDATE models SET {', '.join(fields)} WHERE id = ?", values)


def delete_model(model_id: int) -> None:
    """모델 목록에서 한 건을 지운다."""
    with _connect() as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))


def _kind(kind_id: str) -> dict:
    for kind in KINDS:
        if kind["id"] == kind_id:
            return kind
    raise ValueError("지원하지 않는 모델 종류입니다.")


def _defaults(kind: dict) -> dict:
    return {item["key"]: item["default"] for item in kind["params"]}


def _check_params(kind: dict, params: dict) -> dict:
    clean = _defaults(kind)
    for item in kind["params"]:
        if item["key"] not in params or params[item["key"]] is None:
            continue
        value = params[item["key"]]
        if item["type"] == "int":
            value = int(value)
        else:
            value = float(value)
        if value < item["min"] or value > item["max"]:
            raise ValueError(f"{item['label']}은 {item['min']}~{item['max']}입니다.")
        clean[item["key"]] = value
    return clean


def _out(row: sqlite3.Row) -> dict:
    kind = _kind(row["kind"])
    return {
        "id": row["id"],
        "name": row["name"],
        "kind": row["kind"],
        "kindLabel": kind["label"],
        "params": json.loads(row["params"]),
        "realtime": bool(row["realtime"]),
        "machine": row["machine"] or "",
        "mlflowName": row["mlflow_name"],
        "version": row["version"],
        "status": row["status"],
        "message": row["message"] or "",
        "updatedAt": row["updated_at"],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

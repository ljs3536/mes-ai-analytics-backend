"""진동 이상 탐지 API.

현재 기능:
- 모델을 여러 개 등록한다. 종류는 Isolation Forest, SSAD ProtoNet, HOKO few-shot이다.
- 화면에서 종류별 파라미터와 실시간 적용 여부를 바꾼다.
- 실시간 적용이 켜진 모델만 MQTT 데이터에 판정하고, 연속 이상이면 MES 설비 보류를 호출한다.
- SSAD는 CNN few-shot worker가 학습하고 판정한다.
- HOKO는 임의의 회전수로 센서 파형을 포락선에 올린 뒤 CNN few-shot으로 threshold를 만든다.
- 베어링 사전학습 HOKO 분류기와 가중치는 센서 틀과 달라 뺐다.
"""

import logging
import threading
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from .catalog import create_model, delete_model, get_model, init_catalog, kinds, list_models, mark, update_model
from .config import get_settings
from .registry import ModelError, train_isolation
from .scorer import Scorer, decode_samples
from .workers.pool import WorkerPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

scorer: Scorer | None = None


class Schema(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ModelIn(Schema):
    name: str = Field(min_length=1, max_length=64)
    kind: str
    params: dict = Field(default_factory=dict)
    realtime: bool = False
    machine: str | None = None


class ModelUpdate(Schema):
    name: str = Field(min_length=1, max_length=64)
    params: dict = Field(default_factory=dict)
    realtime: bool = False


class TrainIn(Schema):
    machine: str
    minutes: int = Field(default=30, ge=1, le=1440)
    channel: str | None = None


train_pool = WorkerPool("app.workers.ssad_stdin", 1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global scorer
    init_catalog()
    scorer = Scorer()
    scorer.start()
    yield
    scorer.stop()
    train_pool.stop()


app = FastAPI(title="MES Analytics Backend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ModelError)
async def model_error(_: Request, exc: ModelError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/healthz")
def healthz():
    return {"status": "ok", "mqtt": bool(scorer and scorer.connected())}


@app.get("/api/models")
def models():
    """모델 종류 정의와 등록된 모델 목록을 돌려준다."""
    return {"kinds": kinds(), "models": list_models()}


@app.post("/api/models", status_code=201)
def add_model(body: ModelIn):
    """모델을 등록한다. 실시간 적용 여부는 이 시점에 고른다."""
    try:
        return create_model(body.name, body.kind, body.params, body.realtime, body.machine)
    except ValueError as e:
        raise ModelError(str(e), 422) from e


@app.put("/api/models/{model_id}")
def edit_model(model_id: int, body: ModelUpdate):
    """파라미터와 실시간 적용 여부를 저장한다."""
    try:
        return update_model(model_id, name=body.name, params=body.params, realtime=body.realtime)
    except KeyError as e:
        raise ModelError("모델을 찾을 수 없습니다.", 404) from e
    except ValueError as e:
        raise ModelError(str(e), 422) from e


@app.delete("/api/models/{model_id}", status_code=204)
def remove_model(model_id: int):
    """모델 목록에서 한 건을 지운다."""
    try:
        get_model(model_id)
    except KeyError as e:
        raise ModelError("모델을 찾을 수 없습니다.", 404) from e
    delete_model(model_id)


@app.get("/api/live")
def live():
    """설비별 최신 판정을 화면 폴링에 돌려준다."""
    return {"predictions": scorer.latest if scorer else {}}


@app.post("/api/models/{model_id}/train")
def train_model(model_id: int, body: TrainIn):
    """종류에 맞는 학습을 시작한다. Isolation Forest는 특징값, SSAD와 HOKO는 파형이다."""
    try:
        row = get_model(model_id)
    except KeyError as e:
        raise ModelError("모델을 찾을 수 없습니다.", 404) from e
    if row["status"] == "training":
        raise ModelError("이미 학습 중입니다.", 409)
    if row["kind"] == "isolation_forest":
        return _train_forest(row, body)
    if row["kind"] == "ssad_protonet":
        return _train_ssad(row, body)
    if row["kind"] == "hoko":
        return _train_hoko(row, body)
    raise ModelError("지원하지 않는 모델 종류입니다.", 422)


def _features(body: TrainIn) -> list[dict]:
    params = {"machine": body.machine, "minutes": body.minutes}
    if body.channel:
        params["channel"] = body.channel
    try:
        res = httpx.get(f"{get_settings().collector_url}/api/features", params=params, timeout=30)
        res.raise_for_status()
    except httpx.HTTPError as e:
        raise ModelError(f"수집기에서 특징값을 가져오지 못했습니다: {e}", 503) from e
    return res.json()


def _train_forest(row: dict, body: TrainIn) -> dict:
    """수집기 특징값으로 Isolation Forest를 학습하고 MLflow 버전을 기록한다."""
    mark(row["id"], status="training", message="특징값으로 학습 중")
    try:
        result = train_isolation(_features(body), row["mlflowName"], row["params"])
    except Exception as e:
        mark(row["id"], status="error", message=str(e))
        raise
    mark(row["id"], status="trained", message=f"{result['rows']}개 샘플", version=result["version"])
    return {**result, "realtime": row["realtime"]}


def _train_ssad(row: dict, body: TrainIn) -> dict:
    """최신 파형으로 SSAD few-shot 학습을 백그라운드에 맡긴다."""
    query = f"machine={body.machine}"
    if body.channel:
        query += f"&channel={body.channel}"
    try:
        res = httpx.get(f"{get_settings().collector_url}/api/waveform?{query}", timeout=30)
        res.raise_for_status()
    except httpx.HTTPError as e:
        raise ModelError(f"수집기에서 파형을 가져오지 못했습니다: {e}", 503) from e
    wave = res.json()
    raw = wave["samples"]
    samples = raw if isinstance(raw, list) else decode_samples(raw).tolist()
    mark(row["id"], status="training", message="SSAD worker 학습 중")
    out = f"{get_settings().model_dir}/{row['id']}"
    params = dict(row["params"])
    model_id = row["id"]

    def job() -> None:
        result = train_pool.process({"action": "train", "samples": [samples], "outDir": out, "params": params})
        if not result.get("ok"):
            mark(model_id, status="error", message=result.get("error") or "SSAD 학습 실패")
            return
        learned = {**params, "threshold": result["threshold"]}
        mark(model_id, status="trained", message="worker 학습 완료", version=str(result["version"]), params=learned)

    threading.Thread(target=job, name=f"ssad-train-{model_id}", daemon=True).start()
    return {"version": row["version"] or "", "realtime": row["realtime"], "status": "training"}


def _latest_wave(body: TrainIn) -> dict:
    query = f"machine={body.machine}"
    if body.channel:
        query += f"&channel={body.channel}"
    try:
        res = httpx.get(f"{get_settings().collector_url}/api/waveform?{query}", timeout=30)
        res.raise_for_status()
    except httpx.HTTPError as e:
        raise ModelError(f"수집기에서 파형을 가져오지 못했습니다: {e}", 503) from e
    return res.json()


def _train_hoko(row: dict, body: TrainIn) -> dict:
    """최신 파형으로 HOKO 포락선 few-shot 학습을 백그라운드에 맡긴다."""
    if scorer is None:
        raise ModelError("판정 프로세스가 아직 없습니다.", 503)
    wave = _latest_wave(body)
    raw = wave["samples"]
    samples = raw if isinstance(raw, list) else decode_samples(raw).tolist()
    mark(row["id"], status="training", message="HOKO 포락선 few-shot 학습 중")
    out = f"{get_settings().model_dir}/{row['id']}"
    params = dict(row["params"])
    model_id = row["id"]

    def job() -> None:
        result = scorer.hoko_pool.process(
            {
                "action": "train",
                "samples": [samples],
                "sampleRate": wave["sampleRate"],
                "outDir": out,
                "params": params,
            }
        )
        if not result.get("ok"):
            mark(model_id, status="error", message=result.get("error") or "HOKO 학습 실패")
            return
        learned = {**params, "threshold": result["threshold"]}
        mark(
            model_id,
            status="trained",
            message=f"threshold {result['threshold']} · 임의 rpm {result.get('rpm')}",
            version=str(result["version"]),
            params=learned,
        )

    threading.Thread(target=job, name=f"hoko-train-{model_id}", daemon=True).start()
    return {"version": row["version"] or "", "realtime": row["realtime"], "status": "training"}

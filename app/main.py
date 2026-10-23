"""진동 이상 탐지 API.

현재 기능:
- 모델을 여러 개 등록한다. 종류는 Isolation Forest, SSAD ProtoNet, HOKO few-shot이다. 조회와 변경은 관리자만 한다.
- 화면에서 종류별 파라미터와 실시간 적용 여부를 바꾼다.
- 실시간 적용이 켜진 모델만 MQTT 데이터에 판정하고, 연속 이상이면 내부 토큰으로 MES 설비 보류를 호출한다. 계정이 있으면 그 계정으로 접속한다.
- 파형 이력에서 고른 블록을 이름 있는 세트로 저장하고, 그 세트로 학습한다.
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

from .auth import Admin
from .catalog import create_model, delete_model, get_model, init_catalog, kinds, list_models, mark, update_model
from .datasets import create_dataset, delete_dataset, get_dataset, init_datasets, list_datasets
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
    machine: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    minutes: int = Field(default=30, ge=1, le=1440)
    channel: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    dataset_id: int | None = None


class DatasetItemIn(Schema):
    machine: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    channel: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    ts: str
    rpm: float | None = None
    n: int | None = None
    unit: str = ""
    preset: str = ""


class DatasetIn(Schema):
    name: str
    items: list[DatasetItemIn]


train_pool = WorkerPool("app.workers.ssad_stdin", 1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global scorer
    init_catalog()
    init_datasets()
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
def models(_: Admin):
    """모델 종류 정의와 등록된 모델 목록을 돌려준다."""
    return {"kinds": kinds(), "models": list_models()}


@app.post("/api/models", status_code=201)
def add_model(body: ModelIn, _: Admin):
    """모델을 등록한다. 실시간 적용 여부는 이 시점에 고른다."""
    try:
        return create_model(body.name, body.kind, body.params, body.realtime, body.machine)
    except ValueError as e:
        raise ModelError(str(e), 422) from e


@app.put("/api/models/{model_id}")
def edit_model(model_id: int, body: ModelUpdate, _: Admin):
    """파라미터와 실시간 적용 여부를 저장한다."""
    try:
        return update_model(model_id, name=body.name, params=body.params, realtime=body.realtime)
    except KeyError as e:
        raise ModelError("모델을 찾을 수 없습니다.", 404) from e
    except ValueError as e:
        raise ModelError(str(e), 422) from e


@app.delete("/api/models/{model_id}", status_code=204)
def remove_model(model_id: int, _: Admin):
    """모델 목록에서 한 건을 지운다."""
    try:
        get_model(model_id)
    except KeyError as e:
        raise ModelError("모델을 찾을 수 없습니다.", 404) from e
    delete_model(model_id)


@app.get("/api/datasets")
def datasets(_: Admin):
    """저장된 파형 세트와 그 안의 블록 목록을 돌려준다."""
    return {"datasets": list_datasets()}


@app.post("/api/datasets", status_code=201)
def add_dataset(body: DatasetIn, _: Admin):
    """이력에서 고른 파형 블록을 이름으로 저장한다. 샘플 원본은 저장하지 않는다."""
    try:
        return create_dataset(body.name, [item.model_dump() for item in body.items])
    except ValueError as e:
        raise ModelError(str(e), 422) from e


@app.delete("/api/datasets/{dataset_id}", status_code=204)
def remove_dataset(dataset_id: int, _: Admin):
    try:
        delete_dataset(dataset_id)
    except KeyError as e:
        raise ModelError("세트를 찾을 수 없습니다.", 404) from e


@app.get("/api/live")
def live(_: Admin):
    """설비별 최신 판정을 화면 폴링에 돌려준다."""
    return {"predictions": scorer.latest if scorer else {}}


@app.post("/api/models/{model_id}/train")
def train_model(model_id: int, body: TrainIn, _: Admin):
    """종류에 맞는 학습을 시작한다. Isolation Forest는 특징값, SSAD와 HOKO는 파형이다."""
    try:
        row = get_model(model_id)
    except KeyError as e:
        raise ModelError("모델을 찾을 수 없습니다.", 404) from e
    if row["status"] == "training":
        raise ModelError("이미 학습 중입니다.", 409)
    if row["kind"] == "isolation_forest":
        return _train_forest(row, body)
    waves = _dataset_waves(body.dataset_id) if body.dataset_id else None
    if row["kind"] == "ssad_protonet":
        return _train_ssad(row, body, waves)
    if row["kind"] == "hoko":
        return _train_hoko(row, body, waves)
    raise ModelError("지원하지 않는 모델 종류입니다.", 422)


def _collector_query(body: TrainIn, **extra: int) -> dict:
    params: dict = {"machine": body.machine, **extra}
    if body.channel:
        params["channel"] = body.channel
    return params


def _features(body: TrainIn) -> list[dict]:
    try:
        res = httpx.get(
            f"{get_settings().collector_url}/api/features",
            params=_collector_query(body, minutes=body.minutes),
            timeout=30,
        )
        res.raise_for_status()
    except httpx.HTTPError as e:
        raise ModelError(f"수집기에서 특징값을 가져오지 못했습니다: {e}", 503) from e
    return res.json()


def _dataset_waves(dataset_id: int) -> list[dict]:
    """세트에 들어 있는 시각의 파형 샘플을 수집기에서 가져온다."""
    try:
        items = get_dataset(dataset_id)["items"]
    except KeyError as e:
        raise ModelError("세트를 찾을 수 없습니다.", 404) from e
    if not items:
        raise ModelError("세트가 비어 있습니다.", 422)
    waves = []
    for item in items:
        try:
            res = httpx.get(
                f"{get_settings().collector_url}/api/waveform",
                params={"machine": item["machine"], "channel": item["channel"], "at": item["ts"], "with_fft": False},
                timeout=30,
            )
            res.raise_for_status()
        except httpx.HTTPError as e:
            raise ModelError(f"세트 파형을 가져오지 못했습니다: {e}", 503) from e
        waves.append(res.json())
    rates = {wave["sampleRate"] for wave in waves}
    if len(rates) != 1:
        raise ModelError("한 세트의 샘플레이트는 같아야 합니다.", 422)
    return waves


def _dataset_features(dataset_id: int) -> list[dict]:
    try:
        items = get_dataset(dataset_id)["items"]
    except KeyError as e:
        raise ModelError("세트를 찾을 수 없습니다.", 404) from e
    rows = []
    for item in items:
        try:
            res = httpx.get(
                f"{get_settings().collector_url}/api/features/at",
                params={"machine": item["machine"], "channel": item["channel"], "at": item["ts"]},
                timeout=30,
            )
            res.raise_for_status()
        except httpx.HTTPError as e:
            raise ModelError(f"세트 특징값을 가져오지 못했습니다: {e}", 503) from e
        rows.append(res.json())
    return rows


def _samples(wave: dict) -> list[float]:
    raw = wave["samples"]
    return raw if isinstance(raw, list) else decode_samples(raw).tolist()


def _train_forest(row: dict, body: TrainIn) -> dict:
    """수집기 특징값으로 Isolation Forest를 학습하고 MLflow 버전을 기록한다."""
    mark(row["id"], status="training", message="특징값으로 학습 중")
    try:
        rows = _dataset_features(body.dataset_id) if body.dataset_id else _features(body)
        result = train_isolation(rows, row["mlflowName"], row["params"])
    except Exception as e:
        mark(row["id"], status="error", message=str(e))
        raise
    mark(row["id"], status="trained", message=f"{result['rows']}개 샘플", version=result["version"])
    return {**result, "realtime": row["realtime"]}


def _train_ssad(row: dict, body: TrainIn, waves: list[dict] | None) -> dict:
    """최신 파형 또는 세트의 파형으로 SSAD few-shot 학습을 백그라운드에 맡긴다."""
    if waves is None:
        try:
            res = httpx.get(
                f"{get_settings().collector_url}/api/waveform",
                params=_collector_query(body),
                timeout=30,
            )
            res.raise_for_status()
        except httpx.HTTPError as e:
            raise ModelError(f"수집기에서 파형을 가져오지 못했습니다: {e}", 503) from e
        waves = [res.json()]
    samples = [_samples(wave) for wave in waves]
    mark(row["id"], status="training", message="SSAD worker 학습 중")
    out = f"{get_settings().model_dir}/{row['id']}"
    params = dict(row["params"])
    model_id = row["id"]

    def job() -> None:
        result = train_pool.process({"action": "train", "samples": samples, "outDir": out, "params": params})
        if not result.get("ok"):
            mark(model_id, status="error", message=result.get("error") or "SSAD 학습 실패")
            return
        learned = {**params, "threshold": result["threshold"]}
        mark(model_id, status="trained", message="worker 학습 완료", version=str(result["version"]), params=learned)

    threading.Thread(target=job, name=f"ssad-train-{model_id}", daemon=True).start()
    return {"version": row["version"] or "", "realtime": row["realtime"], "status": "training"}


def _train_hoko(row: dict, body: TrainIn, waves: list[dict] | None) -> dict:
    """최신 파형 또는 세트의 파형으로 HOKO 포락선 few-shot 학습을 백그라운드에 맡긴다."""
    if scorer is None:
        raise ModelError("판정 프로세스가 아직 없습니다.", 503)
    if waves is None:
        try:
            res = httpx.get(
                f"{get_settings().collector_url}/api/waveform",
                params=_collector_query(body),
                timeout=30,
            )
            res.raise_for_status()
        except httpx.HTTPError as e:
            raise ModelError(f"수집기에서 파형을 가져오지 못했습니다: {e}", 503) from e
        waves = [res.json()]
    samples = [_samples(wave) for wave in waves]
    mark(row["id"], status="training", message="HOKO 포락선 few-shot 학습 중")
    out = f"{get_settings().model_dir}/{row['id']}"
    params = dict(row["params"])
    model_id = row["id"]

    def job() -> None:
        result = scorer.hoko_pool.process(
            {
                "action": "train",
                "samples": samples,
                "sampleRate": waves[0]["sampleRate"],
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

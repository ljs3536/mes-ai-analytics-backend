"""진동 이상 탐지 API.

현재 기능:
- 수집기의 특징값으로 Isolation Forest를 학습하고 MLflow에 등록
- 운영 버전 지정
- MQTT로 들어온 특징값을 점수화하고, 연속 이상이면 MES 설비 보류를 호출
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from .config import get_settings
from .registry import ModelError, list_versions, promote, train
from .scorer import Scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

scorer: Scorer | None = None


class Schema(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class TrainIn(Schema):
    machine: str
    minutes: int = Field(default=30, ge=1, le=1440)
    channel: str | None = None
    contamination: float = 0.05


class PromoteIn(Schema):
    version: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    global scorer
    scorer = Scorer()
    scorer.start()
    yield
    scorer.stop()


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
    return {"status": "ok", "mqtt": bool(scorer and scorer.connected()), "model": scorer._version if scorer else None}


@app.get("/api/models")
def models():
    return list_versions()


@app.get("/api/live")
def live():
    return {"predictions": scorer.latest if scorer else {}}


@app.post("/api/train")
def train_model(body: TrainIn):
    params = {"machine": body.machine, "minutes": body.minutes}
    if body.channel:
        params["channel"] = body.channel
    try:
        res = httpx.get(f"{get_settings().collector_url}/api/features", params=params, timeout=30)
        res.raise_for_status()
    except httpx.HTTPError as e:
        raise ModelError(f"수집기에서 특징값을 가져오지 못했습니다: {e}", 503) from e
    return train(res.json(), body.contamination)


@app.post("/api/models/production")
def set_production(body: PromoteIn):
    return promote(body.version)

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import paho.mqtt.client as mqtt

from .config import get_settings
from .featureset import vector
from .registry import load_production

log = logging.getLogger("analytics.score")


class Scorer:
    def __init__(self) -> None:
        s = get_settings()
        self.prefix = s.mqtt_topic_prefix
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"analytics-{socket.gethostname()}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = lambda *_: log.warning("MQTT 연결 끊김, 재연결 대기")
        self.client.reconnect_delay_set(1, 30)
        self._lock = threading.Lock()
        self._model = None
        self._version: str | None = None
        self._streak: dict[str, int] = {}
        self._hold_until: dict[str, float] = {}
        self.latest: dict[str, dict] = {}

    def start(self) -> None:
        s = get_settings()
        self.client.connect_async(s.mqtt_host, s.mqtt_port, keepalive=30)
        self.client.loop_start()
        threading.Thread(target=self._reload_loop, name="model-reload", daemon=True).start()

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def connected(self) -> bool:
        return self.client.is_connected()

    def _on_connect(self, client, _u, _f, reason_code, _p):
        if reason_code.is_failure:
            log.error("MQTT 연결 거부: %s", reason_code)
            return
        client.subscribe(f"{self.prefix}/machines/+/features", qos=0)
        log.info("특징값 토픽 구독")

    def _reload_loop(self) -> None:
        while True:
            try:
                version, model = load_production()
                if version != self._version:
                    self._version, self._model = version, model
                    log.info("운영 모델 로드 v%s", version)
            except Exception:
                if self._model is None:
                    log.info("운영 모델 없음. 학습 후 production alias가 생기면 예측을 시작합니다.")
            time.sleep(10)

    def _on_message(self, _c, _u, msg):
        parts = msg.topic.split("/")
        if parts[-1] != "features":
            return
        machine = parts[-2]
        try:
            payload = json.loads(msg.payload)
        except ValueError:
            return
        values = vector(payload)
        model = self._model
        version = self._version
        if values is None or model is None or version is None:
            return
        x = np.asarray([values], dtype=np.float64)
        raw = float(model.decision_function(x)[0])
        anomaly = int(model.predict(x)[0]) == -1
        score = round(-raw, 6)
        body = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "model": get_settings().model_name,
            "version": version,
            "score": score,
            "isAnomaly": anomaly,
            "channel": payload.get("channel"),
        }
        self.latest[machine] = body
        self.client.publish(f"{self.prefix}/machines/{machine}/anomaly", json.dumps(body), qos=1)
        self._maybe_hold(machine, anomaly, score, version)

    def _maybe_hold(self, machine: str, anomaly: bool, score: float, version: str) -> None:
        if not anomaly:
            self._streak[machine] = 0
            return
        streak = self._streak.get(machine, 0) + 1
        self._streak[machine] = streak
        need = get_settings().anomaly_streak
        if streak < need or time.monotonic() < self._hold_until.get(machine, 0.0):
            return
        self._hold_until[machine] = time.monotonic() + 120
        self._streak[machine] = 0
        reason = f"AI 이상 감지 v{version} score={score:.3f}"
        url = f"{get_settings().mes_url}/api/machines/by-code/{machine}/hold"
        try:
            res = httpx.post(url, params={"reason": reason}, timeout=5)
        except httpx.HTTPError as e:
            log.warning("%s MES 정지 호출 실패: %s", machine, e)
            return
        if res.status_code >= 400:
            log.info("%s MES 정지 거부 %s %s", machine, res.status_code, res.text[:200])
            return
        log.info("%s MES 정지 요청 (%s)", machine, reason)

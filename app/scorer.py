"""실시간 판정.

실시간 적용이 켜진 모델만 구독 데이터에 돌린다.
Isolation Forest는 특징값 토픽, SSAD와 HOKO few-shot은 파형 토픽을 worker로 보낸다.
"""

from __future__ import annotations

import base64
import json
import logging
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import paho.mqtt.client as mqtt

from .catalog import realtime_models
from .config import get_settings
from .featureset import vector
from .registry import load_named
from .workers.pool import WorkerPool

log = logging.getLogger("analytics.score")


def decode_samples(b64: str) -> np.ndarray:
    """MQTT 파형 본문의 base64 float32를 샘플 배열로 푼다."""
    return np.frombuffer(base64.b64decode(b64), dtype="<f4").copy()


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
        self._forest: dict[int, tuple[str, object]] = {}
        self._ssad: dict[int, dict] = {}
        self._hoko: dict[int, dict] = {}
        self._streak: dict[str, int] = {}
        self._hold_until: dict[str, float] = {}
        self.latest: dict[str, dict[str, dict]] = {}
        self.ssad_pool = WorkerPool("app.workers.ssad_stdin", 1)
        self.hoko_pool = WorkerPool("app.workers.hoko_stdin", 1)

    def start(self) -> None:
        """MQTT 구독과 모델 목록 갱신을 시작한다."""
        s = get_settings()
        self.client.connect_async(s.mqtt_host, s.mqtt_port, keepalive=30)
        self.client.loop_start()
        threading.Thread(target=self._reload_loop, name="model-reload", daemon=True).start()

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
        self.ssad_pool.stop()
        self.hoko_pool.stop()

    def connected(self) -> bool:
        return self.client.is_connected()

    def _on_connect(self, client, _u, _f, reason_code, _p):
        if reason_code.is_failure:
            log.error("MQTT 연결 거부: %s", reason_code)
            return
        client.subscribe(f"{self.prefix}/machines/+/features", qos=0)
        client.subscribe(f"{self.prefix}/machines/+/waveform", qos=0)
        log.info("특징값·파형 토픽 구독. 실시간 적용 모델만 판정한다.")

    def _reload_loop(self) -> None:
        while True:
            forests: dict[int, tuple[str, object]] = {}
            ssad: dict[int, dict] = {}
            hoko: dict[int, dict] = {}
            for row in realtime_models():
                if row["kind"] == "isolation_forest" and row["mlflowName"]:
                    try:
                        version, model = load_named(row["mlflowName"])
                        forests[row["id"]] = (version, model)
                    except Exception:
                        continue
                elif row["kind"] == "ssad_protonet":
                    ssad[row["id"]] = row
                elif row["kind"] == "hoko":
                    hoko[row["id"]] = row
            with self._lock:
                self._forest = forests
                self._ssad = ssad
                self._hoko = hoko
            time.sleep(10)

    def _on_message(self, _c, _u, msg):
        parts = msg.topic.split("/")
        if len(parts) < 2:
            return
        kind = parts[-1]
        machine = parts[-2]
        try:
            payload = json.loads(msg.payload)
        except ValueError:
            return
        if kind == "features":
            self._score_forest(machine, payload)
        elif kind == "waveform":
            self._score_waveform(machine, payload)

    def _score_forest(self, machine: str, payload: dict) -> None:
        """특징값 토픽을 실시간 Isolation Forest에 넣어 이상 여부를 낸다."""
        values = vector(payload)
        if values is None:
            return
        x = np.asarray([values], dtype=np.float64)
        with self._lock:
            models = dict(self._forest)
        for model_id, (version, model) in models.items():
            raw = float(model.decision_function(x)[0])
            anomaly = int(model.predict(x)[0]) == -1
            self._publish(machine, model_id, "Isolation Forest", version, round(-raw, 6), anomaly, payload.get("channel"))

    def _score_waveform(self, machine: str, payload: dict) -> None:
        """파형 토픽을 학습된 SSAD·HOKO few-shot에 넣어 기준점과의 거리를 낸다."""
        if "samples" not in payload:
            return
        try:
            samples = decode_samples(payload["samples"]).tolist()
        except Exception:
            return
        with self._lock:
            models = dict(self._ssad)
        for model_id, row in models.items():
            out = Path(get_settings().model_dir) / str(model_id)
            if not (out / "model.pt").exists():
                continue
            result = self.ssad_pool.process(
                {
                    "action": "infer",
                    "samples": samples,
                    "outDir": str(out),
                    "threshold": row["params"].get("threshold") or 0,
                    "params": row["params"],
                }
            )
            if not result.get("ok"):
                continue
            self._publish(
                machine,
                model_id,
                row["name"],
                row["version"] or "1",
                float(result["score"]),
                bool(result["isAnomaly"]),
                payload.get("channel"),
            )
        with self._lock:
            hoko_models = dict(self._hoko)
        for model_id, row in hoko_models.items():
            out = Path(get_settings().model_dir) / str(model_id)
            if not (out / "model.pt").exists():
                continue
            result = self.hoko_pool.process(
                {
                    "action": "infer",
                    "samples": samples,
                    "sampleRate": payload.get("sampleRate") or 0,
                    "outDir": str(out),
                    "threshold": row["params"].get("threshold") or 0,
                    "params": row["params"],
                }
            )
            if not result.get("ok"):
                continue
            self._publish(
                machine,
                model_id,
                row["name"],
                str(result.get("version") or row["version"] or ""),
                float(result["score"]),
                bool(result["isAnomaly"]),
                payload.get("channel"),
            )

    def _publish(self, machine, model_id, name, version, score, anomaly, channel) -> None:
        """판정 결과를 최신 상태와 anomaly 토픽에 올린다."""
        body = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "modelId": model_id,
            "model": name,
            "version": version,
            "score": score,
            "isAnomaly": anomaly,
            "channel": channel,
        }
        bucket = self.latest.setdefault(machine, {})
        bucket[str(model_id)] = body
        self.client.publish(f"{self.prefix}/machines/{machine}/anomaly", json.dumps(body), qos=1)
        self._maybe_hold(machine, model_id, anomaly, score, version, name)

    def _maybe_hold(self, machine: str, model_id: int, anomaly: bool, score: float, version: str, name: str) -> None:
        """같은 설비·모델에서 이상이 연속되면 MES 설비 보류를 호출한다."""
        key = f"{machine}:{model_id}"
        if not anomaly:
            self._streak[key] = 0
            return
        streak = self._streak.get(key, 0) + 1
        self._streak[key] = streak
        need = get_settings().anomaly_streak
        if streak < need or time.monotonic() < self._hold_until.get(machine, 0.0):
            return
        self._hold_until[machine] = time.monotonic() + 120
        self._streak[key] = 0
        reason = f"AI 이상 감지 {name} v{version} score={score:.3f}"
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

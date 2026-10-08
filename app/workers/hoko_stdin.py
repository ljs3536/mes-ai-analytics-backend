"""HOKO 이름으로 등록된 few-shot worker.

실제 특징 추출기는 DualViewKoopman이 아니다.
센서 파형을 임의 회전수 1800rpm으로 각속도 포락선에 올린 뒤, 그 1차원 신호를 SSAD CNN(Classifier)에 넣는다.
학습이 만드는 것은 model.pt의 CNN 가중치, prototype.npy의 정상 임베딩 평균, 거리 threshold다.
루트의 HokoClassifier(믹서와 Koopman trunk, orders=torch.ones((1,)))는 이 worker가 부르지 않는다.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def _reply(body: dict) -> None:
    print(json.dumps(body, ensure_ascii=False), flush=True)


def _signals(payload: dict) -> tuple[np.ndarray, float]:
    """수집 파형을 CNN이 먹을 1차원 포락선으로 바꾼다.

    반환 신호 길이는 1024로 고정된다. rpm은 포락선의 각도 축을 만들 때만 쓰고, 모델 입력 텐서에는 들어가지 않는다.
    """
    from app.hoko.front import to_signal

    rows = payload["samples"]
    rate = float(payload["sampleRate"])
    if rows and isinstance(rows[0], list):
        converted = [to_signal(row, rate) for row in rows]
        return np.stack([item[0] for item in converted]), converted[0][1]
    return to_signal(rows, rate)


def _train(payload: dict) -> dict:
    """포락선으로 CNN few-shot을 학습한다.

    support 구간의 임베딩 평균이 정상 기준점이다.
    기준점과의 거리 평균에 3표준편차를 더한 값을 threshold로 저장한다. 화면에서 threshold를 0보다 크게 주면 그 값을 쓴다.
    """
    import torch
    import lightning.pytorch as pl

    from app.ssad.trainer import BaseTrainer

    params = payload["params"]
    normal, rpm = _signals(payload)
    if normal.ndim == 1:
        normal = normal[None, :]
    need = int(params.get("nSupport", 5)) + int(params.get("nQuery", 8))
    if len(normal) < need:
        normal = np.tile(normal, ((need // len(normal)) + 1, 1))
    trainer_model = BaseTrainer(
        n_filters=int(params.get("nFilters", 20)),
        n_support=int(params.get("nSupport", 5)),
        n_query=int(params.get("nQuery", 8)),
        n=int(params.get("augmentN", 7)),
        m=int(params.get("augmentM", 6)),
        lr=float(params.get("learningRate", 5e-5)),
        normal_signals=normal,
    )
    trainer = pl.Trainer(
        logger=False,
        max_epochs=int(params.get("epochs", 10)),
        accelerator="cpu",
        devices=1,
        deterministic=True,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(trainer_model)
    trainer_model.eval()
    with torch.no_grad():
        emb = trainer_model.model(torch.from_numpy(normal).float())
        prototype = emb.mean(0).detach().cpu()
        dists = torch.norm(emb.cpu() - prototype, dim=1).numpy()
    valid = [float(v) for v in dists.tolist() if math.isfinite(float(v))] or [0.0]
    threshold = float(params.get("threshold") or 0)
    if threshold <= 0:
        threshold = max(0.01, float(np.mean(valid) + 3 * np.std(valid)))
    out = Path(payload["outDir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": trainer_model.model.state_dict(),
            "n_filters": int(params.get("nFilters", 20)),
            "rpm": rpm,
        },
        out / "model.pt",
    )
    np.save(out / "prototype.npy", prototype.numpy())
    return {"ok": True, "threshold": round(threshold, 6), "version": "1", "rpm": rpm}


def _infer(payload: dict) -> dict:
    """저장한 CNN으로 포락선을 임베딩하고, 기준점까지 유클리드 거리가 threshold를 넘으면 이상으로 본다."""
    import torch
    from app.ssad.model import Classifier

    out = Path(payload["outDir"])
    ckpt = out / "model.pt"
    proto_path = out / "prototype.npy"
    if not ckpt.exists() or not proto_path.exists():
        return {"ok": False, "error": "학습된 HOKO few-shot 가중치가 없습니다."}
    signal, _rpm = _signals(payload)
    bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = Classifier(n_filters=int(bundle.get("n_filters", 20)))
    model.load_state_dict(bundle["state_dict"], strict=False)
    model.eval()
    prototype = torch.from_numpy(np.load(proto_path).astype(np.float32)).float().squeeze(0)
    with torch.no_grad():
        emb = model(torch.from_numpy(signal[None, :]).float()).squeeze(0)
        score = float(torch.norm(emb - prototype).item())
    if not math.isfinite(score):
        score = 0.0
    threshold = float(payload.get("threshold") or 0)
    return {"ok": True, "score": round(score, 6), "isAnomaly": bool(threshold > 0 and score > threshold)}


def main() -> None:
    if "--stdin" not in sys.argv:
        raise SystemExit("stdin JSON worker입니다.")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            action = payload.get("action")
            if action == "train":
                _reply(_train(payload))
            elif action == "infer":
                _reply(_infer(payload))
            else:
                _reply({"ok": False, "error": "action은 train 또는 infer입니다."})
        except Exception as exc:
            _reply({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()

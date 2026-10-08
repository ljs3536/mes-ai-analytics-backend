"""HOKO와 few-shot을 결합한 worker.

센서 파형에 임의의 회전수로 HOKO 포락선을 만들고, CNN few-shot이 기준점과 threshold를 저장한다.
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
    """센서 파형을 HOKO 포락선 신호와 임의 rpm으로 바꾼다."""
    from app.hoko.front import to_signal

    rows = payload["samples"]
    rate = float(payload["sampleRate"])
    if rows and isinstance(rows[0], list):
        converted = [to_signal(row, rate) for row in rows]
        return np.stack([item[0] for item in converted]), converted[0][1]
    return to_signal(rows, rate)


def _train(payload: dict) -> dict:
    """포락선 신호로 CNN few-shot을 학습하고 model.pt, prototype.npy, threshold를 저장한다."""
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
    """저장한 HOKO few-shot 기준점과 포락선 임베딩의 거리를 threshold와 비교한다."""
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

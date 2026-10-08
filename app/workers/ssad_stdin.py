"""SSAD ProtoNet worker.

ldt_ssad의 진동 학습 클래스에 화면에서 받은 파라미터를 넣어 학습하고,
저장한 기준점과의 거리로 판정한다. fast_server처럼 JSON 한 줄을 받고 JSON 한 줄을 낸다.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def _reply(body: dict) -> None:
    print(json.dumps(body, ensure_ascii=False), flush=True)


def _train(payload: dict) -> dict:
    """정상 파형으로 CNN few-shot을 학습하고 model.pt, prototype.npy, threshold를 저장한다."""
    import torch
    import lightning.pytorch as pl

    from app.ssad.trainer import BaseTrainer

    params = payload["params"]
    samples = [np.asarray(row, dtype=np.float32) for row in payload["samples"]]
    if not samples:
        raise ValueError("학습할 파형이 없습니다.")
    width = min(len(row) for row in samples)
    normal = np.stack([row[:width] for row in samples])
    need = int(params["nSupport"]) + int(params["nQuery"])
    if len(normal) < need:
        normal = np.tile(normal, ((need // len(normal)) + 1, 1))

    trainer_model = BaseTrainer(
        n_filters=int(params["nFilters"]),
        n_support=int(params["nSupport"]),
        n_query=int(params["nQuery"]),
        n=int(params["augmentN"]),
        m=int(params["augmentM"]),
        lr=float(params["learningRate"]),
        normal_signals=normal,
    )
    trainer = pl.Trainer(
        logger=False,
        max_epochs=int(params["epochs"]),
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
        {"state_dict": trainer_model.model.state_dict(), "n_filters": int(params["nFilters"])},
        out / "model.pt",
    )
    np.save(out / "prototype.npy", prototype.numpy())
    return {"ok": True, "threshold": round(threshold, 6), "version": "1"}


def _infer(payload: dict) -> dict:
    """저장한 기준점과 파형 임베딩의 거리를 threshold와 비교한다."""
    import torch
    from app.ssad.model import Classifier

    out = Path(payload["outDir"])
    ckpt = out / "model.pt"
    proto_path = out / "prototype.npy"
    if not ckpt.exists() or not proto_path.exists():
        return {"ok": False, "error": "학습된 SSAD 가중치가 없습니다."}
    bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
    n_filters = int(bundle.get("n_filters", payload["params"].get("nFilters", 20)))
    model = Classifier(n_filters=n_filters)
    model.load_state_dict(bundle["state_dict"], strict=False)
    model.eval()
    prototype = torch.from_numpy(np.load(proto_path).astype(np.float32)).float().squeeze(0)
    signal = np.asarray(payload["samples"], dtype=np.float32)
    with torch.no_grad():
        emb = model(torch.from_numpy(signal[None, :]).float()).squeeze(0)
        score = float(torch.norm(emb - prototype).item())
    if not math.isfinite(score):
        score = 0.0
    threshold = float(payload.get("threshold") or 0)
    anomaly = bool(threshold > 0 and score > threshold)
    return {"ok": True, "score": round(score, 6), "isAnomaly": anomaly}


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

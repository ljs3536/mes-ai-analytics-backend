"""센서 파형을 HOKO 포락선으로 바꾼다.

회전수는 센서에 없으므로 torch.ones 로 만든 임의의 값을 쓴다.
베어링 종류 가중치는 쓰지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).with_name("hust.json")
SIGNAL_LENGTH = 1024


def arbitrary_rpm() -> float:
    """센서에 없는 회전수를 torch.ones 기반 고정값으로 만든다."""
    import torch

    return float(torch.ones(1) * 1800)


def to_signal(samples, sample_rate: float) -> tuple[np.ndarray, float]:
    """파형을 HOKO 포락선 평균으로 접어 CNN 입력 길이로 맞춘다."""
    from .adapters import sensor_envelopes

    rpm = arbitrary_rpm()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    envelopes = sensor_envelopes(
        np.asarray(samples, dtype=np.float64),
        sample_rate=float(sample_rate),
        shaft_frequency_hz=rpm / 60.0,
        config=config,
    )
    trace = envelopes.mean(axis=1).reshape(-1).astype(np.float64)
    if len(trace) < 2:
        raise ValueError("포락선 길이가 너무 짧습니다.")
    grid = np.linspace(0.0, 1.0, SIGNAL_LENGTH)
    source = np.linspace(0.0, 1.0, len(trace))
    return np.interp(grid, source, trace).astype(np.float32), rpm

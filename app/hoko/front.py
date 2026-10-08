"""센서 파형을 CNN few-shot이 읽는 1차원 신호로 바꾼다.

임의 회전수는 torch.ones(1) * 1800 이다. 이 값은 각속도 포락선의 축을 만들 뿐이고,
Koopman trunk의 orders 텐서로 전달되지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).with_name("hust.json")
SIGNAL_LENGTH = 1024


def arbitrary_rpm() -> float:
    """포락선 각도 축에 쓸 고정 회전수.

    torch.ones(1)에 1800을 곱한다. 1rpm이면 1024샘플, 2560Hz 블록 안에 4회전 창이 생기지 않는다.
    """
    import torch

    return float(torch.ones(1) * 1800)


def to_signal(samples, sample_rate: float) -> tuple[np.ndarray, float]:
    """대역 포락선을 평균 내어 길이 1024의 1차원 신호로 다시 샘플링한다.

    sensor_envelopes 결과는 (1, 대역 수, 각도 좌표)다. 대역 축을 평균하면 CNN Classifier가 받는 (샘플 길이,) 파형이 된다.
    """
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

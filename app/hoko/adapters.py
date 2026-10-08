"""센서 파형을 HOKO 각속도 포락선으로 바꾼다.

베어링 사전학습 분류기와 51.2 kHz mat 변환은 센서 틀과 달라 뺐다.
"""

from __future__ import annotations

import numpy as np


def sensor_envelopes(
    waveform: np.ndarray, *, sample_rate: float, shaft_frequency_hz: float, config: dict
) -> np.ndarray:
    """시간 파형을 주파수 대역별 포락선으로 만든 뒤, 축 회전에 맞춰 각도 좌표로 다시 찍는다.

    1. 평균을 빼 FFT한다.
    2. rfft 빈을 spectral_atoms(32)개 구간으로 나누고, 각 구간만 역변환한 절대값이 그 대역의 포락선이다.
    3. 블록 길이 * 축주파수 / 샘플레이트가 회전 수다. window_units(4)보다 짧으면 각도 창을 만들 수 없어 실패한다.
    4. 회전당 samples_per_unit(64)개 좌표로 포락선을 보간한다.
    반환 shape는 (1, 대역 수, 각도 좌표)다. Koopman 믹서의 [batch, atom, coordinate]와 비슷해 보이지만, 이 배열은 CNN 입력으로 평균되기 전에 여기서 끝난다.
    """

    observation = config["observation"]
    atom_count = int(observation["spectral_atoms"])
    samples_per_unit = int(observation["samples_per_unit"])
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if len(values) < 64 or sample_rate <= 0 or shaft_frequency_hz <= 0:
        raise ValueError("파형, 샘플레이트, 회전수가 부족합니다.")
    block = values - values.mean()
    spectrum = np.fft.rfft(block)
    bins = spectrum.shape[0]
    edges = np.linspace(0, bins, atom_count + 1, dtype=np.int64)
    envelopes = np.zeros((1, atom_count, len(block)), dtype=np.float32)
    for atom, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        local = np.zeros(len(block), dtype=np.complex64)
        local[int(left) : int(right)] = spectrum[int(left) : int(right)]
        start = 1 if int(left) == 0 else int(left)
        local[start : int(right)] *= 2.0
        envelopes[0, atom] = np.abs(np.fft.ifft(local))
    revolutions = len(block) * shaft_frequency_hz / sample_rate
    units = int(np.floor(revolutions))
    if units < int(np.ceil(float(observation["window_units"]))) + 1:
        raise ValueError("한 블록 안의 회전 수가 부족해 HOKO 창을 만들 수 없습니다.")
    coordinate_count = units * samples_per_unit
    positions = np.arange(coordinate_count) * sample_rate / (samples_per_unit * shaft_frequency_hz)
    positions = np.clip(positions, 0, len(block) - 1.001)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, len(block) - 1)
    fraction = (positions - left).astype(np.float32)
    return envelopes[..., left] * (1.0 - fraction) + envelopes[..., right] * fraction
